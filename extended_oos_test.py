"""
True Out-of-Sample (OOS) Test Framework — v15 (Phase 1.5 Restored)
===================================================================
Framework profesional de evaluacion de robustez.
Condiciones Estrictas:
1. Datos 2022-2026 jamas vistos por el modelo.
2. Split temporal Train(10-18), Val(19-21), OOS(22-26).
3. Costos estocasticos (Slippage normal N(0.1R, 0.05R)).
4. Monte Carlo 1000 permutaciones.
5. Metricas: CAGR, Sharpe, Sortino, Calmar, Ulcer, PF, Expectancy.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys
import joblib
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from features import create_features
from fractals import detect_fractals

# --- CONFIGURACION ---
PAIRS = {
    # Forex Base
    "EURUSD": {"ticker": "EURUSD=X", "pip": 0.0001, "spread": 3.0, "commission": 15.0, "swap": 0.3},
    "GBPUSD": {"ticker": "GBPUSD=X", "pip": 0.0001, "spread": 4.0, "commission": 15.0, "swap": 0.5},
    "NZDUSD": {"ticker": "NZDUSD=X", "pip": 0.0001, "spread": 3.0, "commission": 15.0, "swap": 0.5},
    "AUDUSD": {"ticker": "AUDUSD=X", "pip": 0.0001, "spread": 3.0, "commission": 15.0, "swap": 0.5},
    "USDCAD": {"ticker": "USDCAD=X", "pip": 0.0001, "spread": 4.0, "commission": 15.0, "swap": 0.5},
    "USDCHF": {"ticker": "USDCHF=X", "pip": 0.0001, "spread": 5.0, "commission": 15.0, "swap": 0.5},
    "EURGBP": {"ticker": "EURGBP=X", "pip": 0.0001, "spread": 5.0, "commission": 15.0, "swap": 0.5},
    # Nuevos Forex
    "USDJPY": {"ticker": "USDJPY=X", "pip": 0.01,   "spread": 3.0, "commission": 15.0, "swap": 0.5},
    "EURJPY": {"ticker": "EURJPY=X", "pip": 0.01,   "spread": 4.0, "commission": 15.0, "swap": 0.5},
    "GBPJPY": {"ticker": "GBPJPY=X", "pip": 0.01,   "spread": 5.0, "commission": 15.0, "swap": 0.5},
    # Acciones Globales
    "AAPL":   {"ticker": "AAPL",   "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "MSFT":   {"ticker": "MSFT",   "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "AMZN":   {"ticker": "AMZN",   "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "NVDA":   {"ticker": "NVDA",   "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "GOOGL":  {"ticker": "GOOGL",  "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "META":   {"ticker": "META",   "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "TSLA":   {"ticker": "TSLA",   "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "BRK-B":  {"ticker": "BRK-B",  "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "JPM":    {"ticker": "JPM",    "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "KO":     {"ticker": "KO",     "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "JNJ":    {"ticker": "JNJ",    "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "PG":     {"ticker": "PG",     "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "XOM":    {"ticker": "XOM",    "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "BABA":   {"ticker": "BABA",   "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0},
    "SAN":    {"ticker": "SAN",    "pip": 0.01, "spread": 5.0, "commission": 1.0, "swap": 0.0}
}

SPLITS = {
    "TRAIN": ("2010-02-01", "2018-12-31"),
    "VAL":   ("2019-01-01", "2021-12-31"),
    "OOS":   ("2022-01-01", "2026-02-22"),
}

INITIAL_BALANCE        = 10000
RISK_PER_TRADE         = 0.005   # 0.5% base
MAX_RISK_PCT           = 0.01    # 1.0% cap
ALPHA_SIZING           = 0.5
LOOKAHEAD              = 5       # Extreme exit sensitivity test (5 bars)
MonteCarloPermutations = 1000

def get_data(ticker):
    print(f"    📥 Descargando {ticker}...")
    df = yf.download(ticker, start="2010-01-01", end="2026-02-22",
                     interval="1d", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df.rename(columns={"Date": "Datetime"})

def apply_stochastic_pnl(raw_pnl_usd, risk_usd):
    slippage = np.random.normal(0.10 * risk_usd, 0.05 * risk_usd)
    return raw_pnl_usd - slippage

def calculate_metrics(trades, equity, years):
    if not trades or len(trades) < 2:
        return None
    pnls = np.array([t['pnl_usd'] for t in trades])
    returns = np.array(equity)

    total_ret = returns[-1] / returns[0]
    cagr = (total_ret ** (1 / years)) - 1 if years > 0 else 0

    daily_returns = np.diff(returns) / returns[:-1]
    sharpe   = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
    neg_ret  = daily_returns[daily_returns < 0]
    sortino  = (np.mean(daily_returns) / np.std(neg_ret) * np.sqrt(252)) if len(neg_ret) > 1 and np.std(neg_ret) > 0 else 0

    max_dd = 0
    peak = returns[0]
    drawdowns = []
    for r in returns:
        if r > peak:
            peak = r
        dd = (peak - r) / peak
        drawdowns.append(dd)
        max_dd = max(max_dd, dd)

    ulcer  = np.sqrt(np.mean(np.square(drawdowns))) * 100
    calmar = (cagr / max_dd) if max_dd > 0 else 0

    win_rate   = len(pnls[pnls > 0]) / len(pnls)
    avg_w      = np.mean(pnls[pnls > 0]) if any(pnls > 0) else 0
    avg_l      = abs(np.mean(pnls[pnls <= 0])) if any(pnls <= 0) else 1e-6
    expectancy = (win_rate * avg_w) - ((1 - win_rate) * avg_l)

    return {
        "CAGR":   cagr * 100,
        "MaxDD":  max_dd * 100,
        "Sharpe": sharpe,
        "Sortino":sortino,
        "Calmar": calmar,
        "Ulcer":  ulcer,
        "Exp":    expectancy,
        "WR":     win_rate * 100,
        "PF":     np.sum(pnls[pnls > 0]) / abs(np.sum(pnls[pnls <= 0])) if any(pnls <= 0) else 99,
        "n_trades": len(trades),
    }

def monte_carlo_sim(trades, initial_balance):
    pnls       = np.array([t['pnl_usd'] for t in trades])
    results    = []
    ruins      = 0
    for _ in range(MonteCarloPermutations):
        shuffled = np.random.permutation(pnls)
        bal = initial_balance
        ruined = False
        for p in shuffled:
            bal += p
            if bal <= initial_balance * 0.5:
                ruined = True
        results.append(bal)
        if ruined:
            ruins += 1
    results = np.sort(results)
    return {
        "p5":        results[int(0.05 * len(results))],
        "p95":       results[int(0.95 * len(results))],
        "mean":      np.mean(results),
        "prob_ruin": (ruins / MonteCarloPermutations) * 100,
    }

def run_backtest(df_feat, art, feature_cols, start_date, end_date, cfg):
    """
    Backtest bar-by-bar con TP/SL en OHLC real.
    Usa el modelo multi-clase (classes=[0=HOLD, 1=BUY, 2=SELL]).
    Sin lookahead. Features on-the-fly.
    """
    mask = (df_feat['Datetime'] >= start_date) & (df_feat['Datetime'] <= end_date)
    df_p = df_feat.loc[mask].copy().reset_index(drop=True)
    if len(df_p) < 50:
        return None, None

    missing = [c for c in feature_cols if c not in df_p.columns]
    if missing:
        print(f"    ⚠️  Columnas faltantes ({len(missing)}): {missing[:3]}")
        return None, None

    X = df_p[feature_cols].values
    model = art['model']
    probas = model.predict_proba(X)   # shape (N, 3): [P(HOLD), P(BUY), P(SELL)]

    th = art.get('threshold', 0.51)   # Threshold del artefacto (guardado por main.py)

    trades  = []
    balance = INITIAL_BALANCE
    equity  = [INITIAL_BALANCE]

    for i in range(len(df_p) - LOOKAHEAD):
        prob_buy  = probas[i, 1]
        prob_sell = probas[i, 2]

        buy_sig  = prob_buy  >= th
        sell_sig = prob_sell >= th

        if not buy_sig and not sell_sig:
            equity.append(balance)
            continue

        # Dirección
        if buy_sig and not sell_sig:
            sig_dir, conf = "BUY", prob_buy
        elif sell_sig and not buy_sig:
            sig_dir, conf = "SELL", prob_sell
        else:
            if (prob_buy - th) >= (prob_sell - th):
                sig_dir, conf = "BUY", prob_buy
            else:
                sig_dir, conf = "SELL", prob_sell

        # Sizing
        if conf > th and th < 1.0:
            scaled = RISK_PER_TRADE * ((conf - th) / (1.0 - th)) * ALPHA_SIZING + RISK_PER_TRADE
        else:
            scaled = RISK_PER_TRADE
        adj_risk_pct = min(scaled, MAX_RISK_PCT)
        risk_usd     = balance * adj_risk_pct

        # Setup TP/SL
        entry  = df_p['Close'].iloc[i]
        atr    = df_p['ATR'].iloc[i] if 'ATR' in df_p.columns and not np.isnan(df_p['ATR'].iloc[i]) else entry * 0.005
        sl_dist = max(atr * 1.0, entry * 0.001) # Mismo que labeling
        tp_dist = sl_dist * 1.5

        # Tracking multibar (max 20 bars)
        hit_tp = False
        hit_sl = False
        exit_idx = i + LOOKAHEAD
        
        for j in range(i + 1, i + 1 + LOOKAHEAD):
            if j >= len(df_p): break
            
            high = df_p['High'].iloc[j]
            low  = df_p['Low'].iloc[j]
            
            if sig_dir == "BUY":
                sl_touched = low <= (entry - sl_dist)
                tp_touched = high >= (entry + tp_dist)
                
                if sl_touched and tp_touched:
                    hit_sl = True # Worst case: SL hits first
                    exit_idx = j
                    break
                elif sl_touched:
                    hit_sl = True
                    exit_idx = j
                    break
                elif tp_touched:
                    hit_tp = True
                    exit_idx = j
                    break
                    
            else: # SELL
                sl_touched = high >= (entry + sl_dist)
                tp_touched = low <= (entry - tp_dist)
                
                if sl_touched and tp_touched:
                    hit_sl = True # Worst case
                    exit_idx = j
                    break
                elif sl_touched:
                    hit_sl = True
                    exit_idx = j
                    break
                elif tp_touched:
                    hit_tp = True
                    exit_idx = j
                    break

        # Result
        raw_pnl = risk_usd * 1.5 if hit_tp else -risk_usd
        # Si no tocó nada en 20 velas, cerramos a precio de mercado (timeout)
        if not hit_tp and not hit_sl:
            last_close = df_p['Close'].iloc[i + LOOKAHEAD]
            if sig_dir == "BUY":
                raw_pnl = ((last_close - entry) / sl_dist) * risk_usd
            else:
                raw_pnl = ((entry - last_close) / sl_dist) * risk_usd

        pnl_net = apply_stochastic_pnl(raw_pnl, risk_usd) - cfg['commission']
        balance = max(balance + pnl_net, 1.0)
        equity.append(balance)

        trades.append({
            'dir':     sig_dir,
            'conf':    round(conf, 4),
            'pnl_usd': pnl_net,
            'hit_tp':  hit_tp,
        })

    return trades, equity


def main():
    print(f"\n{'='*80}")
    print(f"  TRUE OOS TEST FRAMEWORK v15 — Phase 1.5 Restored")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Cargar modelo multi-clase de Phase 1.5 (classes=[0=HOLD, 1=BUY, 2=SELL])
    try:
        art          = joblib.load("model_multi.joblib")
        feature_cols = art['feature_columns']
        print(f"✅ model_multi.joblib cargado | classes=[0,1,2] | threshold={art.get('threshold', '?')} | features={len(feature_cols)}\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Asegurate de tener model_multi.joblib (generado por main.py).")
        return

    results = {}

    for name, cfg in PAIRS.items():
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")

        df_raw = get_data(cfg['ticker'])
        if df_raw is None:
            print(f"  ❌ Sin datos para {name}")
            continue

        # Construir features on-the-fly (idéntico al pipeline de entrenamiento)
        print(f"    ⚙️  Construyendo features...")
        try:
            df_feat = create_features(df_raw.copy())
            df_feat = detect_fractals(df_feat)
            df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
        except Exception as e:
            print(f"  ❌ Error features: {e}")
            continue

        if df_feat.empty:
            print(f"  ❌ DataFrame vacio tras features")
            continue

        # === VALIDATION (2019-2021) ===
        print(f"    🔬 VALIDATION 2019-2021...")
        v_trades, v_equity = run_backtest(
            df_feat, art, feature_cols,
            SPLITS['VAL'][0], SPLITS['VAL'][1], cfg
        )
        v_m = calculate_metrics(v_trades, v_equity, 3.0)

        # === OOS (2022-2026) — MOMENTO DE LA VERDAD ===
        print(f"    🎯 OOS 2022-2026... ← NUNCA VISTO POR EL MODELO")
        o_trades, o_equity = run_backtest(
            df_feat, art, feature_cols,
            SPLITS['OOS'][0], SPLITS['OOS'][1], cfg
        )
        o_m = calculate_metrics(o_trades, o_equity, 4.1)

        if not v_m or not o_m:
            print(f"  ⚠️ Sin trades suficientes para {name}")
            continue

        mc = monte_carlo_sim(o_trades, INITIAL_BALANCE)

        overfit  = (o_m['PF'] < v_m['PF'] * 0.60) and v_m['PF'] > 1.0
        fragile  = o_m['WR'] < v_m['WR'] - 15
        is_robust = o_m['PF'] > 1.3 and not overfit and not fragile

        results[name] = {
            "VAL":    v_m,
            "OOS":    o_m,
            "MC":     mc,
            "ROBUST": is_robust,
            "REASON": ("OVERFIT " if overfit else "") + ("FRAGILE" if fragile else "") or "STABLE",
        }

        print(f"\n  {'─'*50}")
        print(f"  VAL 2019-21 | PF: {v_m['PF']:>6.2f} | WR: {v_m['WR']:>5.1f}% | Sharpe: {v_m['Sharpe']:>5.2f} | Exp: ${v_m['Exp']:>7.2f} | Trades: {v_m['n_trades']}")
        print(f"  OOS 2022-26 | PF: {o_m['PF']:>6.2f} | WR: {o_m['WR']:>5.1f}% | Sharpe: {o_m['Sharpe']:>5.2f} | Exp: ${o_m['Exp']:>7.2f} | Trades: {o_m['n_trades']}")
        print(f"  Monte Carlo | Mean: ${mc['mean']:.0f} | P5: ${mc['p5']:.0f} | P95: ${mc['p95']:.0f} | P(Ruina): {mc['prob_ruin']:.1f}%")
        v = "✅ ROBUSTO" if is_robust else f"❌ {results[name]['REASON']}"
        print(f"  VEREDICTO   | {v}")

    # ── REPORTE FINAL ──────────────────────────────────────
    if not results:
        print("\n❌ Sin resultados. Revisar modelos y datos.")
        return

    print(f"\n\n{'='*80}")
    print(f"  RESUMEN FINAL: VALIDATION vs OOS")
    print(f"{'='*80}")
    print(f"{'Par':<10} | {'VAL PF':>7} | {'OOS PF':>7} | {'VAL WR':>7} | {'OOS WR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | Status")
    print(f"{'─'*80}")

    for name, res in results.items():
        v, o = res['VAL'], res['OOS']
        st = "✅ ROBUSTO" if res['ROBUST'] else f"❌ {res['REASON']}"
        print(f"{name:<10} | {v['PF']:>7.2f} | {o['PF']:>7.2f} | {v['WR']:>6.1f}% | {o['WR']:>6.1f}% | {o['MaxDD']:>6.1f}% | {o['Sharpe']:>6.2f} | {st}")

    robust_n = sum(1 for r in results.values() if r['ROBUST'])
    total_n  = len(results)
    print(f"\n  Robustez: {robust_n}/{total_n} pares superaron el OOS.")
    print(f"\n  ⚠️  RECORDATORIO DE PROTOCOLO:")
    print(f"  → Si OOS es positivo: Canary Live 30 días en Demo (mismo broker, mismo sizing).")
    print(f"  → Registrar: slippage real, fills, spread horario.")
    print(f"  → NO ajustar nada hasta completar el Canary Live.")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()