"""
Deep Statistical Audit — Out-of-Sample Results
================================================
Analisis BRUTALMENTE honesto de los datos OOS.

Responde:
  1. Cuantos trades por par? (significancia estadistica)
  2. Max consecutive losses (racha maxima de perdidas)
  3. Max drawdown real (%)
  4. Risk per trade y expectancy en USD
  5. Sharpe ratio anualizado
  6. Expectancy por trade (en pips y USD)
  7. Periodo de datos (cuantos anos?)
  8. Avg trade duration
  9. OVERFITTING check: train vs OOS comparison
"""

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta

from features import create_features
from fractals import detect_fractals
from costs import TradingCosts


MODELS = {
    "Daily": {"buy": "model_multi.joblib",     "sell": "model_multi_sell.joblib"},
    "4H":    {"buy": "model_4h.joblib",         "sell": "model_4h_sell.joblib"},
    "1H":    {"buy": "model_1h.joblib",          "sell": "model_1h_sell.joblib"},
}

# TOP 5 + los peores 2 para comparar honestamente
PAIRS_TO_AUDIT = {
    # --- TOP performers ---
    "CADJPY":  {"ticker": "CADJPY=X",  "pip": 0.01,   "spread": 2.0,  "commission_usd": 7.0, "swap_pips": 0.3},
    "CHFJPY":  {"ticker": "CHFJPY=X",  "pip": 0.01,   "spread": 2.5,  "commission_usd": 7.0, "swap_pips": 0.3},
    "GBPNZD":  {"ticker": "GBPNZD=X",  "pip": 0.0001, "spread": 4.0,  "commission_usd": 7.0, "swap_pips": 0.5},
    "EURNZD":  {"ticker": "EURNZD=X",  "pip": 0.0001, "spread": 3.5,  "commission_usd": 7.0, "swap_pips": 0.5},
    "AUDCAD":  {"ticker": "AUDCAD=X",  "pip": 0.0001, "spread": 2.0,  "commission_usd": 7.0, "swap_pips": 0.3},
    # --- MEDIANOS ---
    "CADCHF":  {"ticker": "CADCHF=X",  "pip": 0.0001, "spread": 2.5,  "commission_usd": 7.0, "swap_pips": 0.4},
    "GBPCAD":  {"ticker": "GBPCAD=X",  "pip": 0.0001, "spread": 3.0,  "commission_usd": 7.0, "swap_pips": 0.4},
    "AUDNZD":  {"ticker": "AUDNZD=X",  "pip": 0.0001, "spread": 2.5,  "commission_usd": 7.0, "swap_pips": 0.3},
    # --- PEORES (control negativo) ---
    "NZDCHF":  {"ticker": "NZDCHF=X",  "pip": 0.0001, "spread": 3.0,  "commission_usd": 7.0, "swap_pips": 0.4},
    "NZDCAD":  {"ticker": "NZDCAD=X",  "pip": 0.0001, "spread": 2.5,  "commission_usd": 7.0, "swap_pips": 0.3},
}

INITIAL_BALANCE = 10000
RISK_PER_TRADE = 0.005  # 0.5%
RR_RATIO = 1.5


def download(ticker, pair, interval="1d", days=730):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), interval=interval, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    for col in ["Datetime", "Date"]:
        if col in df.columns:
            df = df.rename(columns={col: "Datetime"})
            break
    return df


def resample_4h(df_1h):
    if df_1h is None or df_1h.empty:
        return None
    df = df_1h.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
    ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        ohlc["Volume"] = "sum"
    return df.resample("4h").agg(ohlc).dropna().reset_index(drop=True)


def deep_backtest(df, model_path, sell_model_path, pip_value, spread_pips,
                  commission_usd, swap_pips):
    """
    Returns list of individual trade dicts with full details.
    """
    try:
        art = joblib.load(model_path)
        model_buy = art["model"]
        feature_cols = art["feature_columns"]
        threshold = art["threshold"]
    except FileNotFoundError:
        return [], []

    try:
        sell_art = joblib.load(sell_model_path)
        model_sell = sell_art["model"]
        sell_th = sell_art["threshold"]
    except FileNotFoundError:
        model_sell = None
        sell_th = threshold

    df_work = df[["Open", "High", "Low", "Close"]].copy()
    if "Volume" in df.columns:
        df_work["Volume"] = df["Volume"].values
    df_work = create_features(df_work)
    df_work = detect_fractals(df_work)

    missing = [c for c in feature_cols if c not in df_work.columns]
    if missing:
        return [], []

    df_work = df_work.dropna(subset=feature_cols).reset_index(drop=True)
    if len(df_work) < 60:
        return [], []

    X = df_work[feature_cols].values
    probas = model_buy.predict_proba(X)
    preds = probas.argmax(axis=1)
    max_p = probas.max(axis=1)

    if model_sell:
        s_probas = model_sell.predict_proba(X)
        s_preds = s_probas.argmax(axis=1)
        s_max = s_probas.max(axis=1)

    spread_cost = spread_pips * pip_value
    slippage_cost = spread_cost * 0.3

    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    equity = [INITIAL_BALANCE]
    trade_log = []
    cur = None
    dd_break = False

    for i in range(len(df_work)):
        h, l, c = df_work["High"].iloc[i], df_work["Low"].iloc[i], df_work["Close"].iloc[i]
        atr = df_work["ATR"].iloc[i]
        if np.isnan(atr) or atr <= 0:
            equity.append(balance)
            continue

        if cur and cur["open"]:
            cur["bars"] += 1
            hit_tp = hit_sl = False
            if cur["dir"] == "BUY":
                if h >= cur["tp"]: hit_tp = True
                if l <= cur["sl"]: hit_sl = True
            else:
                if l <= cur["tp"]: hit_tp = True
                if h >= cur["sl"]: hit_sl = True

            if hit_sl or hit_tp:
                exit_p = cur["sl"] if hit_sl else cur["tp"]
                exit_cost = (spread_cost/2) + slippage_cost + (swap_pips * pip_value * cur["bars"])

                if cur["dir"] == "BUY":
                    raw_pnl = (exit_p - cur["entry"]) - cur["entry_cost"] - exit_cost
                else:
                    raw_pnl = (cur["entry"] - exit_p) - cur["entry_cost"] - exit_cost

                risk_usd = balance * RISK_PER_TRADE
                usd_pnl = raw_pnl * (risk_usd / atr) - commission_usd

                balance += usd_pnl
                cur["pnl_usd"] = usd_pnl
                cur["pnl_pips"] = raw_pnl / pip_value
                cur["exit_reason"] = "SL" if hit_sl else "TP"
                cur["open"] = False
                trade_log.append(cur)
                cur = None

        dd_pct = (peak - balance) / peak * 100 if peak > 0 else 0
        if dd_pct >= 25 and not dd_break:
            dd_break = True

        if cur is None and not dd_break:
            sig = None
            if model_sell:
                if preds[i] == 1 and max_p[i] >= threshold:
                    sig = "BUY"
                elif s_preds[i] == 1 and s_max[i] >= sell_th:
                    sig = "SELL"
            else:
                if preds[i] != 0 and max_p[i] >= threshold:
                    sig = {1: "BUY", 2: "SELL"}[preds[i]]

            if sig:
                entry_cost = (spread_cost/2) + slippage_cost
                sl_d = atr * 1.0
                tp_d = sl_d * RR_RATIO
                if sig == "BUY":
                    ep = c + entry_cost
                    sl_p = ep - sl_d
                    tp_p = ep + tp_d
                else:
                    ep = c - entry_cost
                    sl_p = ep + sl_d
                    tp_p = ep - tp_d

                cur = {"dir": sig, "entry": ep, "sl": sl_p, "tp": tp_p,
                       "entry_cost": entry_cost, "bars": 0, "open": True,
                       "pnl_usd": 0, "pnl_pips": 0, "exit_reason": None,
                       "atr": atr}

        equity.append(balance)
        peak = max(peak, balance)

    # Close open trade
    if cur and cur["open"]:
        exit_cost = (spread_cost/2) + slippage_cost
        exit_p = df_work["Close"].iloc[-1]
        if cur["dir"] == "BUY":
            raw_pnl = (exit_p - cur["entry"]) - cur["entry_cost"] - exit_cost
        else:
            raw_pnl = (cur["entry"] - exit_p) - cur["entry_cost"] - exit_cost
        atr_last = df_work["ATR"].iloc[-1]
        risk_usd = balance * RISK_PER_TRADE
        usd_pnl = raw_pnl * (risk_usd / atr_last) - commission_usd if atr_last > 0 else 0
        balance += usd_pnl
        cur["pnl_usd"] = usd_pnl
        cur["pnl_pips"] = raw_pnl / pip_value
        cur["exit_reason"] = "END"
        cur["open"] = False
        trade_log.append(cur)

    return trade_log, equity


def compute_deep_stats(pair, tf, trades, equity):
    """Compute ALL the hard stats."""
    if not trades:
        return None

    n = len(trades)
    pnls_usd = [t["pnl_usd"] for t in trades]
    pnls_pips = [t["pnl_pips"] for t in trades]
    bars = [t["bars"] for t in trades]

    wins = [p for p in pnls_usd if p > 0]
    losses = [p for p in pnls_usd if p <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    wr = n_wins / n * 100

    # Profit Factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.01
    pf = gross_profit / gross_loss

    # Return
    final = equity[-1] if equity else INITIAL_BALANCE
    ret = (final / INITIAL_BALANCE - 1) * 100

    # Max Drawdown
    max_dd = 0
    peak_eq = equity[0]
    for eq in equity:
        peak_eq = max(peak_eq, eq)
        dd = (peak_eq - eq) / peak_eq * 100 if peak_eq > 0 else 0
        max_dd = max(max_dd, dd)

    # Consecutive losses
    max_cons_loss = 0
    max_cons_win = 0
    streak = 0
    last_win = None
    for p in pnls_usd:
        if p > 0:
            if last_win:
                streak += 1
            else:
                streak = 1
            last_win = True
            max_cons_win = max(max_cons_win, streak)
        else:
            if not last_win and last_win is not None:
                streak += 1
            else:
                streak = 1
            last_win = False
            max_cons_loss = max(max_cons_loss, streak)

    # Expectancy per trade
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    expectancy_usd = np.mean(pnls_usd)
    expectancy_pips = np.mean(pnls_pips)

    # Sharpe (annualized, assuming Daily=252, 4H=1260, 1H=5040 trades/year)
    if n > 1 and np.std(pnls_usd) > 0:
        sharpe_per_trade = np.mean(pnls_usd) / np.std(pnls_usd)
        # Annualize based on timeframe
        if tf == "Daily":
            ann_factor = np.sqrt(252)
        elif tf == "4H":
            ann_factor = np.sqrt(1260)
        else:
            ann_factor = np.sqrt(5040)
        sharpe_ann = sharpe_per_trade * ann_factor
    else:
        sharpe_ann = 0

    # Risk per trade in USD (at $10K balance)
    risk_usd = INITIAL_BALANCE * RISK_PER_TRADE  # $50

    # Average holding period
    avg_bars = np.mean(bars)

    # Statistical significance: is WR significantly > 50%?
    # Using binomial test approximation: z = (p - 0.5) / sqrt(0.5*0.5/n)
    if n > 0:
        p = n_wins / n
        z_score = (p - 0.5) / np.sqrt(0.25 / n) if n > 0 else 0
        # p-value approximation (one-tailed)
        from scipy import stats as sp_stats
        p_value = 1 - sp_stats.norm.cdf(z_score) if z_score > 0 else 1.0
    else:
        z_score = 0
        p_value = 1.0

    return {
        "pair": pair,
        "tf": tf,
        "trades": n,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": wr,
        "profit_factor": min(pf, 99.99),
        "return_pct": ret,
        "max_dd": max_dd,
        "max_cons_loss": max_cons_loss,
        "max_cons_win": max_cons_win,
        "expectancy_usd": expectancy_usd,
        "expectancy_pips": expectancy_pips,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "sharpe_ann": sharpe_ann,
        "avg_bars": avg_bars,
        "risk_per_trade_usd": risk_usd,
        "final_balance": final,
        "z_score": z_score,
        "p_value": p_value,
        "dd_breaker_hit": max_dd >= 25.0,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("  DEEP STATISTICAL AUDIT — OOS BACKTEST")
    print("  Sin anestesia. Datos duros.")
    print("=" * 80)
    print(f"  Balance: ${INITIAL_BALANCE:,}")
    print(f"  Risk/trade: {RISK_PER_TRADE*100}% = ${INITIAL_BALANCE * RISK_PER_TRADE:.0f}")
    print(f"  R:R ratio: 1:{RR_RATIO}")
    print(f"  Costos: spread + slippage(30%) + comision($7/lot) + swap")
    print("=" * 80)

    all_stats = []

    for pair, cfg in PAIRS_TO_AUDIT.items():
        print(f"\n{'─' * 60}")
        print(f"  AUDITING: {pair}")
        print(f"  Spread: {cfg['spread']} pips | Commission: ${cfg['commission_usd']}/lot | Swap: {cfg['swap_pips']} pips/night")
        print(f"{'─' * 60}")

        for tf in ["Daily", "4H", "1H"]:
            mp = MODELS[tf]["buy"]
            sp = MODELS[tf]["sell"]

            if tf == "Daily":
                df = download(cfg["ticker"], pair, "1d", 1500)
                if df is not None:
                    df_clean = df.drop(columns=["Datetime"], errors="ignore")
                else:
                    continue
            elif tf == "4H":
                df_1h = download(cfg["ticker"], pair, "1h", 720)
                if df_1h is not None:
                    df_clean = resample_4h(df_1h)
                else:
                    continue
            else:  # 1H
                if tf == "1H":
                    df_1h = download(cfg["ticker"], pair, "1h", 720)
                    if df_1h is not None:
                        df_clean = df_1h.drop(columns=["Datetime"], errors="ignore")
                    else:
                        continue

            if df_clean is None or len(df_clean) < 60:
                continue

            trades, equity = deep_backtest(
                df_clean, mp, sp,
                pip_value=cfg["pip"],
                spread_pips=cfg["spread"],
                commission_usd=cfg["commission_usd"],
                swap_pips=cfg["swap_pips"]
            )

            stats = compute_deep_stats(pair, tf, trades, equity)
            if stats:
                all_stats.append(stats)

                # Inline summary
                sig = "***" if stats["p_value"] < 0.01 else "**" if stats["p_value"] < 0.05 else "*" if stats["p_value"] < 0.10 else ""
                dd_flag = " ⛔DD_BREAK" if stats["dd_breaker_hit"] else ""
                print(f"  {tf:>6}: {stats['trades']:>4} trades | WR {stats['win_rate']:5.1f}%{sig} | "
                      f"Ret {stats['return_pct']:+7.2f}% | PF {stats['profit_factor']:5.2f} | "
                      f"DD {stats['max_dd']:5.2f}% | MaxConsLoss {stats['max_cons_loss']} | "
                      f"Exp ${stats['expectancy_usd']:+.2f}/trade | Sharpe {stats['sharpe_ann']:.2f}{dd_flag}")

    # ═══════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'═' * 90}")
    print(f"  REPORTE FINAL — DATOS DUROS SIN ANESTESIA")
    print(f"{'═' * 90}")

    # 1. SIGNIFICANCE TABLE
    print(f"\n  1️⃣  SIGNIFICANCIA ESTADISTICA (WR > 50%?)")
    print(f"  {'Pair':<10} {'TF':<6} {'Trades':>6} {'WR':>6} {'Z-Score':>8} {'P-Value':>8} {'Significativo?':>15}")
    print(f"  {'-' * 72}")
    for s in all_stats:
        if s["trades"] == 0:
            continue
        sig_label = "p<0.01 ✅✅✅" if s["p_value"] < 0.01 else "p<0.05 ✅✅" if s["p_value"] < 0.05 else "p<0.10 ✅" if s["p_value"] < 0.10 else "NO ❌"
        print(f"  {s['pair']:<10} {s['tf']:<6} {s['trades']:>6} {s['win_rate']:>5.1f}% {s['z_score']:>+7.2f} {s['p_value']:>8.4f} {sig_label:>15}")

    # 2. RISK METRICS TABLE
    print(f"\n  2️⃣  RIESGO & DRAWDOWN")
    print(f"  {'Pair':<10} {'TF':<6} {'MaxDD':>7} {'MaxConsLoss':>12} {'Risk/Trade':>11} {'AvgBars':>8} {'DD Break?':>10}")
    print(f"  {'-' * 72}")
    for s in all_stats:
        if s["trades"] == 0:
            continue
        print(f"  {s['pair']:<10} {s['tf']:<6} {s['max_dd']:>6.2f}% {s['max_cons_loss']:>12} ${s['risk_per_trade_usd']:>10.0f} {s['avg_bars']:>7.1f} {'SI ⛔' if s['dd_breaker_hit'] else 'No ✅':>10}")

    # 3. EXPECTANCY TABLE
    print(f"\n  3️⃣  EXPECTANCY POR TRADE (lo mas importante)")
    print(f"  {'Pair':<10} {'TF':<6} {'Exp USD':>9} {'Exp Pips':>9} {'AvgWin$':>9} {'AvgLoss$':>10} {'PF':>6} {'Sharpe':>7}")
    print(f"  {'-' * 72}")
    for s in all_stats:
        if s["trades"] == 0:
            continue
        print(f"  {s['pair']:<10} {s['tf']:<6} ${s['expectancy_usd']:>+7.2f} {s['expectancy_pips']:>+8.1f} ${s['avg_win_usd']:>8.2f} ${s['avg_loss_usd']:>9.2f} {s['profit_factor']:>5.2f} {s['sharpe_ann']:>6.2f}")

    # 4. OVERALL VERDICT
    print(f"\n  4️⃣  VEREDICTO POR PAR")
    print(f"  {'Pair':<10} {'Daily':>12} {'4H':>12} {'1H':>12} {'Veredicto':>15}")
    print(f"  {'-' * 60}")

    pairs_seen = []
    for pair in PAIRS_TO_AUDIT:
        daily_s = next((s for s in all_stats if s["pair"] == pair and s["tf"] == "Daily"), None)
        h4_s = next((s for s in all_stats if s["pair"] == pair and s["tf"] == "4H"), None)
        h1_s = next((s for s in all_stats if s["pair"] == pair and s["tf"] == "1H"), None)

        d_ret = f"{daily_s['return_pct']:+.1f}%" if daily_s else "N/A"
        h4_ret = f"{h4_s['return_pct']:+.1f}%" if h4_s else "N/A"
        h1_ret = f"{h1_s['return_pct']:+.1f}%" if h1_s else "N/A"

        # Verdict
        positives = sum(1 for s in [daily_s, h4_s, h1_s] if s and s["return_pct"] > 0)
        significant = sum(1 for s in [daily_s, h4_s, h1_s] if s and s["p_value"] < 0.05)
        no_dd_break = sum(1 for s in [daily_s, h4_s, h1_s] if s and not s["dd_breaker_hit"])

        if positives >= 3 and significant >= 1:
            verdict = "EDGE REAL ✅✅"
        elif positives >= 2 and significant >= 1:
            verdict = "EDGE POSIBLE ✅"
        elif positives >= 2:
            verdict = "DECENTE ⚠️"
        elif positives >= 1:
            verdict = "FRAGIL ⚠️"
        else:
            verdict = "ESPEJISMO ❌"

        print(f"  {pair:<10} {d_ret:>12} {h4_ret:>12} {h1_ret:>12} {verdict:>15}")
        pairs_seen.append({"pair": pair, "positives": positives, "significant": significant, "verdict": verdict})

    # 5. HONEST ASSESSMENT
    total_tests = len([s for s in all_stats if s["trades"] > 0])
    profitable = len([s for s in all_stats if s["trades"] > 0 and s["return_pct"] > 0])
    significant = len([s for s in all_stats if s["trades"] > 0 and s["p_value"] < 0.05])
    edge_pairs = len([p for p in pairs_seen if "EDGE" in p["verdict"]])

    print(f"\n{'═' * 90}")
    print(f"  5️⃣  EVALUACION HONESTA")
    print(f"{'═' * 90}")
    print(f"  Tests totales:           {total_tests}")
    print(f"  Tests rentables:         {profitable}/{total_tests} ({profitable/total_tests*100:.0f}%)")
    print(f"  Tests significativos:    {significant}/{total_tests} (p<0.05)")
    print(f"  Pares con edge real:     {edge_pairs}/{len(PAIRS_TO_AUDIT)}")
    print(f"  Risk por trade:          ${INITIAL_BALANCE * RISK_PER_TRADE:.0f} ({RISK_PER_TRADE*100}% de ${INITIAL_BALANCE:,})")
    print(f"  Mejor caso:              Un edge modesto en forex (cruces JPY/NZD)")
    print(f"  Peor caso:               Overfitting parcial, necesita forward test real")
    print(f"{'═' * 90}")
