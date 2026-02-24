"""
Massive Out-of-Sample Backtest
================================
Tests ALL unseen forex pairs + crypto with realistic trading costs.
None of these instruments were in the 14-pair training set.

Training pairs (EXCLUDED from this test):
  EURUSD, GBPUSD, NZDUSD, AUDUSD, USDCAD, USDCHF, EURGBP,
  USDJPY, EURJPY, GBPJPY, AUDJPY, NZDJPY, EURAUD, GBPAUD

Realistic costs include:
  - Spread (typical for each pair/crypto)
  - Slippage (~30% of spread)
  - Swap/overnight costs
"""

import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from costs import TradingCosts
from features import create_features
from fractals import detect_fractals

# ─── Model paths ────────────────────────────────────────────────────────
MODELS = {
    "Daily": {"buy": "model_multi.joblib", "sell": "model_multi_sell.joblib"},
    "4H": {"buy": "model_4h.joblib", "sell": "model_4h_sell.joblib"},
    "1H": {"buy": "model_1h.joblib", "sell": "model_1h_sell.joblib"},
}

# ─── UNSEEN FOREX PAIRS ────────────────────────────────────────────────
# Realistic spreads from major brokers (IC Markets, Pepperstone, etc.)
# pip_value: 0.0001 for most, 0.01 for JPY pairs
# spread: average spread in pips (includes ECN markup)
# commission: round-turn commission in USD per standard lot (100K)
# swap_per_night: average daily swap cost in pips (averaged long/short)
FOREX_PAIRS = {
    "CADJPY": {
        "ticker": "CADJPY=X",
        "pip": 0.01,
        "spread": 2.0,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.3,
    },
    "CADCHF": {
        "ticker": "CADCHF=X",
        "pip": 0.0001,
        "spread": 2.5,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.4,
    },
    "NZDCAD": {
        "ticker": "NZDCAD=X",
        "pip": 0.0001,
        "spread": 2.5,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.3,
    },
    "EURNZD": {
        "ticker": "EURNZD=X",
        "pip": 0.0001,
        "spread": 3.5,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.5,
    },
    "GBPNZD": {
        "ticker": "GBPNZD=X",
        "pip": 0.0001,
        "spread": 4.0,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.5,
    },
    "GBPCAD": {
        "ticker": "GBPCAD=X",
        "pip": 0.0001,
        "spread": 3.0,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.4,
    },
    "AUDCAD": {
        "ticker": "AUDCAD=X",
        "pip": 0.0001,
        "spread": 2.0,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.3,
    },
    "AUDNZD": {
        "ticker": "AUDNZD=X",
        "pip": 0.0001,
        "spread": 2.5,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.3,
    },
    "NZDCHF": {
        "ticker": "NZDCHF=X",
        "pip": 0.0001,
        "spread": 3.0,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.4,
    },
    "CHFJPY": {
        "ticker": "CHFJPY=X",
        "pip": 0.01,
        "spread": 2.5,
        "lot": 100000,
        "commission_usd": 7.0,
        "swap_pips": 0.3,
    },
}

# ─── CRYPTO PAIRS ───────────────────────────────────────────────────────
# Costs based on Binance/Bybit spot + futures averages.
# spread: typical bid/ask spread in $ units
# commission: taker fee (0.1% each way for spot, ~0.06% for futures)
# No swap, but funding rate simulated via swap_pips
CRYPTO_PAIRS = {
    "ETH": {
        "ticker": "ETH-USD",
        "pip": 0.01,
        "spread": 0.50,
        "lot": 1,
        "commission_pct": 0.001,
        "swap_pips": 0.0,
    },
    "SOL": {
        "ticker": "SOL-USD",
        "pip": 0.001,
        "spread": 0.03,
        "lot": 1,
        "commission_pct": 0.001,
        "swap_pips": 0.0,
    },
    "XRP": {
        "ticker": "XRP-USD",
        "pip": 0.0001,
        "spread": 0.001,
        "lot": 1,
        "commission_pct": 0.001,
        "swap_pips": 0.0,
    },
    "BNB": {
        "ticker": "BNB-USD",
        "pip": 0.01,
        "spread": 0.15,
        "lot": 1,
        "commission_pct": 0.001,
        "swap_pips": 0.0,
    },
    "ADA": {
        "ticker": "ADA-USD",
        "pip": 0.0001,
        "spread": 0.0005,
        "lot": 1,
        "commission_pct": 0.001,
        "swap_pips": 0.0,
    },
    "DOGE": {
        "ticker": "DOGE-USD",
        "pip": 0.0001,
        "spread": 0.0003,
        "lot": 1,
        "commission_pct": 0.001,
        "swap_pips": 0.0,
    },
    "AVAX": {
        "ticker": "AVAX-USD",
        "pip": 0.001,
        "spread": 0.05,
        "lot": 1,
        "commission_pct": 0.001,
        "swap_pips": 0.0,
    },
    "LINK": {
        "ticker": "LINK-USD",
        "pip": 0.001,
        "spread": 0.02,
        "lot": 1,
        "commission_pct": 0.001,
        "swap_pips": 0.0,
    },
}


def download_data(ticker, pair_name, interval="1d", days=730):
    """Download data from yfinance."""
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
    )
    if df.empty:
        print(f"  [!] Sin datos para {pair_name}")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    for col in ["Datetime", "Date"]:
        if col in df.columns:
            df = df.rename(columns={col: "Datetime"})
            break
    return df


def resample_to_4h(df_1h):
    """Resample 1H to 4H."""
    if df_1h is None or df_1h.empty:
        return None
    df = df_1h.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
    ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        ohlc["Volume"] = "sum"
    df_4h = df.resample("4h").agg(ohlc).dropna()
    return df_4h.reset_index(drop=True)


class Trade:
    def __init__(self, entry_idx, direction, entry_price, sl, tp, costs_entry):
        self.entry_idx = entry_idx
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.costs_entry = costs_entry
        self.exit_idx = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = None
        self.nights = 0

    def is_open(self):
        return self.exit_price is None

    def close(self, exit_idx, exit_price, reason, costs_exit, nights=0):
        self.exit_idx = exit_idx
        self.exit_price = exit_price
        self.exit_reason = reason
        self.nights = nights
        if self.direction == "BUY":
            self.pnl = (exit_price - self.entry_price) - self.costs_entry - costs_exit
        else:
            self.pnl = (self.entry_price - exit_price) - self.costs_entry - costs_exit


def run_backtest(
    df,
    model_path,
    sell_model_path,
    tf_name,
    pip_value=0.0001,
    spread_pips=1.5,
    swap_per_night=0.0,
    commission_per_trade_usd=0.0,
    risk_per_trade=0.005,
    rr_ratio=1.5,
    max_dd_pct=25.0,
    initial_balance=10000,
):
    """
    Run backtest with REALISTIC costs:
      - Spread (entry + exit)
      - Slippage (30% of spread, random)
      - Commission (per round-turn)
      - Swap/overnight (per night held)
    """
    # Load models
    try:
        artifact = joblib.load(model_path)
        model_buy = artifact["model"]
        feature_cols = artifact["feature_columns"]
        threshold = artifact["threshold"]
    except FileNotFoundError:
        return None

    try:
        sell_artifact = joblib.load(sell_model_path)
        model_sell = sell_artifact["model"]
        sell_threshold = sell_artifact["threshold"]
    except FileNotFoundError:
        model_sell = None
        sell_threshold = threshold

    # Prepare data
    df_work = df[["Open", "High", "Low", "Close"]].copy()
    if "Volume" in df.columns:
        df_work["Volume"] = df["Volume"].values

    df_work = create_features(df_work)
    df_work = detect_fractals(df_work)

    missing = [c for c in feature_cols if c not in df_work.columns]
    if missing:
        return None

    df_work = df_work.dropna(subset=feature_cols).reset_index(drop=True)
    if len(df_work) < 60:
        return None

    # Predictions
    X = df_work[feature_cols].values
    probas = model_buy.predict_proba(X)
    predictions = probas.argmax(axis=1)
    max_probs = probas.max(axis=1)

    if model_sell is not None:
        sell_probas = model_sell.predict_proba(X)
        sell_preds = sell_probas.argmax(axis=1)
        sell_max_probs = sell_probas.max(axis=1)

    # Trading costs
    spread_cost = spread_pips * pip_value  # in price units
    slippage_cost = spread_cost * 0.3  # 30% of spread as slippage

    # Simulation
    balance = initial_balance
    peak_balance = initial_balance
    trades = []
    equity_curve = [initial_balance]
    current_trade = None
    dd_breaker_triggered = False

    for i in range(len(df_work)):
        high = df_work["High"].iloc[i]
        low = df_work["Low"].iloc[i]
        close = df_work["Close"].iloc[i]
        atr = df_work["ATR"].iloc[i]

        if np.isnan(atr) or atr <= 0:
            equity_curve.append(balance)
            continue

        # Check open trade
        if current_trade is not None and current_trade.is_open():
            current_trade.nights += 1
            hit_tp = hit_sl = False

            if current_trade.direction == "BUY":
                if high >= current_trade.tp:
                    hit_tp = True
                if low <= current_trade.sl:
                    hit_sl = True
            else:
                if low <= current_trade.tp:
                    hit_tp = True
                if high >= current_trade.sl:
                    hit_sl = True

            if hit_sl:
                exit_price = current_trade.sl
                # Exit costs: half spread + slippage + swap for nights held
                exit_cost = (
                    (spread_cost / 2)
                    + slippage_cost
                    + (swap_per_night * pip_value * current_trade.nights)
                )
                current_trade.close(
                    i, exit_price, "SL", exit_cost, current_trade.nights
                )

                risk_usd = balance * risk_per_trade
                usd_pnl = current_trade.pnl * (risk_usd / atr)
                # Subtract commission
                usd_pnl -= commission_per_trade_usd
                balance += usd_pnl
                trades.append(current_trade)
                current_trade = None

            elif hit_tp:
                exit_price = current_trade.tp
                exit_cost = (
                    (spread_cost / 2)
                    + slippage_cost
                    + (swap_per_night * pip_value * current_trade.nights)
                )
                current_trade.close(
                    i, exit_price, "TP", exit_cost, current_trade.nights
                )

                risk_usd = balance * risk_per_trade
                usd_pnl = current_trade.pnl * (risk_usd / atr)
                usd_pnl -= commission_per_trade_usd
                balance += usd_pnl
                trades.append(current_trade)
                current_trade = None

        # Drawdown check
        current_dd = (
            (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        )
        if current_dd >= max_dd_pct and not dd_breaker_triggered:
            dd_breaker_triggered = True

        # Open new trade
        if current_trade is None and not dd_breaker_triggered:
            signal = None
            if model_sell is not None:
                if predictions[i] == 1 and max_probs[i] >= threshold:
                    signal = "BUY"
                elif sell_preds[i] == 1 and sell_max_probs[i] >= sell_threshold:
                    signal = "SELL"
            else:
                pred = predictions[i]
                prob = max_probs[i]
                if pred != 0 and prob >= threshold:
                    signal = {0: "HOLD", 1: "BUY", 2: "SELL"}[pred]

            if signal:
                # Entry cost: half spread + slippage
                entry_cost = (spread_cost / 2) + slippage_cost
                sl_distance = atr * 1.0
                tp_distance = sl_distance * rr_ratio

                if signal == "BUY":
                    entry_price = close + entry_cost
                    sl_price = entry_price - sl_distance
                    tp_price = entry_price + tp_distance
                else:
                    entry_price = close - entry_cost
                    sl_price = entry_price + sl_distance
                    tp_price = entry_price - tp_distance

                current_trade = Trade(
                    i, signal, entry_price, sl_price, tp_price, entry_cost
                )

        equity_curve.append(balance)
        peak_balance = max(peak_balance, balance)

    # Close open trade at end
    if current_trade is not None and current_trade.is_open():
        exit_cost = (spread_cost / 2) + slippage_cost
        current_trade.close(
            len(df_work) - 1, df_work["Close"].iloc[-1], "END", exit_cost
        )
        atr_last = df_work["ATR"].iloc[-1]
        risk_usd = balance * risk_per_trade
        usd_pnl = current_trade.pnl * (risk_usd / atr_last) if atr_last > 0 else 0
        usd_pnl -= commission_per_trade_usd
        balance += usd_pnl
        trades.append(current_trade)

    # Compute results
    if not trades:
        return {
            "pair": "",
            "tf": tf_name,
            "trades": 0,
            "win_rate": 0,
            "return_pct": 0,
            "max_dd": 0,
            "profit_factor": 0,
            "final_balance": initial_balance,
        }

    wins = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    n_trades = len(trades)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    total_return = (balance / initial_balance - 1) * 100

    # Max drawdown
    max_dd = 0
    peak_eq = equity_curve[0]
    for eq in equity_curve:
        peak_eq = max(peak_eq, eq)
        dd = (peak_eq - eq) / peak_eq * 100 if peak_eq > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        "tf": tf_name,
        "trades": n_trades,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": win_rate,
        "return_pct": total_return,
        "max_dd": max_dd,
        "profit_factor": min(pf, 99.99),
        "final_balance": balance,
    }


def test_pair(pair_name, config, is_crypto=False):
    """Test a single pair across all timeframes."""
    ticker = config["ticker"]
    pip = config["pip"]
    spread = config["spread"]

    # Commission: forex = flat USD per lot, crypto = % of trade value
    if is_crypto:
        # For crypto: commission = 0.1% round-turn on $10K balance ~ $20 per trade
        commission_usd = 10000 * config.get("commission_pct", 0.001) * 2  # round-turn
    else:
        commission_usd = config.get("commission_usd", 7.0)

    swap = config.get("swap_pips", 0.0)

    bt_kwargs = dict(
        pip_value=pip,
        spread_pips=spread,
        swap_per_night=swap,
        commission_per_trade_usd=commission_usd,
        risk_per_trade=0.005,
        rr_ratio=1.5,
        max_dd_pct=25.0,
        initial_balance=10000,
    )

    results = []

    # Daily
    df_daily = download_data(ticker, pair_name, interval="1d", days=1500)
    if df_daily is not None:
        df_clean = df_daily.drop(columns=["Datetime"], errors="ignore")
        r = run_backtest(
            df_clean,
            MODELS["Daily"]["buy"],
            MODELS["Daily"]["sell"],
            "Daily",
            **bt_kwargs,
        )
        if r:
            r["pair"] = pair_name
            results.append(r)

    # 4H (resample from 1H)
    df_1h = download_data(ticker, pair_name, interval="1h", days=720)
    if df_1h is not None:
        df_4h = resample_to_4h(df_1h)
        if df_4h is not None and len(df_4h) >= 60:
            r = run_backtest(
                df_4h, MODELS["4H"]["buy"], MODELS["4H"]["sell"], "4H", **bt_kwargs
            )
            if r:
                r["pair"] = pair_name
                results.append(r)

    # 1H
    if df_1h is not None:
        df_1h_clean = df_1h.drop(columns=["Datetime"], errors="ignore")
        if len(df_1h_clean) >= 60:
            r = run_backtest(
                df_1h_clean,
                MODELS["1H"]["buy"],
                MODELS["1H"]["sell"],
                "1H",
                **bt_kwargs,
            )
            if r:
                r["pair"] = pair_name
                results.append(r)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  MASSIVE OUT-OF-SAMPLE BACKTEST")
    print("  Pares NUNCA vistos durante entrenamiento")
    print("  Costos REALISTAS: spread + slippage + comision + swap")
    print("=" * 70)

    all_results = []

    # ─── FOREX ──────────────────────────────────────────────────────
    print(f"\n{'#' * 70}")
    print(f"  FOREX — {len(FOREX_PAIRS)} pares no-vistos")
    print(f"{'#' * 70}")

    for i, (pair, config) in enumerate(FOREX_PAIRS.items(), 1):
        print(f"\n--- [{i}/{len(FOREX_PAIRS)}] {pair} ---")
        print(
            f"    Spread: {config['spread']} pips | Commission: ${config['commission_usd']}/lot | Swap: {config['swap_pips']} pips/night"
        )
        results = test_pair(pair, config, is_crypto=False)
        all_results.extend(results)
        for r in results:
            status = "+" if r["return_pct"] > 0 else "-"
            print(
                f"    {r['tf']:>6}: {r['trades']:>4} trades | {r['win_rate']:5.1f}% WR | {r['return_pct']:+7.2f}% | PF {r['profit_factor']:5.2f} | DD {r['max_dd']:5.2f}%"
            )

    # ─── CRYPTO ─────────────────────────────────────────────────────
    print(f"\n{'#' * 70}")
    print(f"  CRYPTO — {len(CRYPTO_PAIRS)} pares")
    print(f"  (usando modelos Forex — test de transferencia de patrones)")
    print(f"{'#' * 70}")

    for i, (pair, config) in enumerate(CRYPTO_PAIRS.items(), 1):
        print(f"\n--- [{i}/{len(CRYPTO_PAIRS)}] {pair} ---")
        pct = config.get("commission_pct", 0.001) * 100
        print(f"    Spread: {config['spread']} | Commission: {pct:.2f}% round-turn")
        results = test_pair(pair, config, is_crypto=True)
        all_results.extend(results)
        for r in results:
            print(
                f"    {r['tf']:>6}: {r['trades']:>4} trades | {r['win_rate']:5.1f}% WR | {r['return_pct']:+7.2f}% | PF {r['profit_factor']:5.2f} | DD {r['max_dd']:5.2f}%"
            )

    # ─── SUMMARY ────────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print(f"  RESUMEN COMPLETO — {len(all_results)} tests")
    print(f"{'=' * 90}")
    print(
        f"  {'Pair':<10} {'TF':<6} {'Trades':>6} {'WR':>6} {'Return':>9} {'MaxDD':>7} {'PF':>6} {'Final':>10} {'Status':>7}"
    )
    print(f"  {'-' * 85}")

    profitable = 0
    losing = 0

    for r in all_results:
        if r["trades"] == 0:
            continue
        status = "  WIN" if r["return_pct"] > 0 else " LOSS"
        if r["return_pct"] > 0:
            profitable += 1
        else:
            losing += 1
        print(
            f"  {r['pair']:<10} {r['tf']:<6} {r['trades']:>6} {r['win_rate']:>5.1f}% {r['return_pct']:>+8.2f}% {r['max_dd']:>6.2f}% {r['profit_factor']:>5.2f} ${r['final_balance']:>9,.2f} {status}"
        )

    total_tests = profitable + losing
    print(f"\n  {'=' * 85}")
    print(
        f"  RESULTADO: {profitable}/{total_tests} tests RENTABLES ({profitable/total_tests*100:.0f}%)"
        if total_tests > 0
        else "  Sin resultados"
    )
    print(f"  {losing}/{total_tests} tests en PERDIDA" if total_tests > 0 else "")
    print(f"  {'=' * 85}")

    # Aggregate by category
    forex_results = [
        r for r in all_results if r["pair"] in FOREX_PAIRS and r["trades"] > 0
    ]
    crypto_results = [
        r for r in all_results if r["pair"] in CRYPTO_PAIRS and r["trades"] > 0
    ]

    if forex_results:
        avg_ret = np.mean([r["return_pct"] for r in forex_results])
        avg_wr = np.mean([r["win_rate"] for r in forex_results])
        avg_pf = np.mean([r["profit_factor"] for r in forex_results])
        pos = sum(1 for r in forex_results if r["return_pct"] > 0)
        print(
            f"\n  FOREX ({len(forex_results)} tests): avg return {avg_ret:+.2f}% | avg WR {avg_wr:.1f}% | avg PF {avg_pf:.2f} | {pos}/{len(forex_results)} profitable"
        )

    if crypto_results:
        avg_ret = np.mean([r["return_pct"] for r in crypto_results])
        avg_wr = np.mean([r["win_rate"] for r in crypto_results])
        avg_pf = np.mean([r["profit_factor"] for r in crypto_results])
        pos = sum(1 for r in crypto_results if r["return_pct"] > 0)
        print(
            f"  CRYPTO ({len(crypto_results)} tests): avg return {avg_ret:+.2f}% | avg WR {avg_wr:.1f}% | avg PF {avg_pf:.2f} | {pos}/{len(crypto_results)} profitable"
        )

    # By timeframe
    for tf in ["Daily", "4H", "1H"]:
        tf_results = [r for r in all_results if r["tf"] == tf and r["trades"] > 0]
        if tf_results:
            avg_ret = np.mean([r["return_pct"] for r in tf_results])
            avg_wr = np.mean([r["win_rate"] for r in tf_results])
            pos = sum(1 for r in tf_results if r["return_pct"] > 0)
            print(
                f"  {tf:>6} ({len(tf_results)} tests): avg return {avg_ret:+.2f}% | avg WR {avg_wr:.1f}% | {pos}/{len(tf_results)} profitable"
            )
