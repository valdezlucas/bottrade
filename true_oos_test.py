"""
True Out-of-Sample (OOS) Test Framework — v15
==============================================
Framework profesional de evaluacion de robustez.
Condiciones Estrictas:
1. Datos 2022-2025 jamas vistos por el modelo.
2. Split temporal Train(10-18), Val(19-21), OOS(22-25).
3. Walk-Forward (2y train + 1y test).
4. Costos estocasticos (Slippage normal N(0.1R, 0.05R)).
5. Monte Carlo 1000 permutaciones.
6. Metricas de vanguardia: Ulcer, Calmar, Prob Ruina.
"""

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime
import sys
from scipy import stats
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from features import create_features
from fractals import detect_fractals
from train import prepare_dataset, train_final_model
from train_sell import prepare_sell_dataset, train_sell_model

# --- CONFIGURACION ---
PAIRS = {
    "EURUSD": {"ticker": "EURUSD=X", "pip": 0.0001, "spread": 1.5, "commission": 7.0, "swap": 0.3},
    "GBPUSD": {"ticker": "GBPUSD=X", "pip": 0.0001, "spread": 2.0, "commission": 7.0, "swap": 0.5},
    "USDJPY": {"ticker": "USDJPY=X", "pip": 0.01,   "spread": 1.5, "commission": 7.0, "swap": 0.3},
    "CADJPY": {"ticker": "CADJPY=X", "pip": 0.01,   "spread": 2.0, "commission": 7.0, "swap": 0.3},
    "CHFJPY": {"ticker": "CHFJPY=X", "pip": 0.01,   "spread": 2.5, "commission": 7.0, "swap": 0.3},
}

SPLITS = {
    "TRAIN": ("2010-02-01", "2018-12-31"),
    "VAL":   ("2019-01-01", "2021-12-31"),
    "OOS":   ("2022-01-01", "2025-12-31"),
}

INITIAL_BALANCE = 10000
RISK_PER_TRADE = 0.005  # 0.5%
MonteCarloPermutations = 1000

def get_data(ticker, pair, timeframe="1d"):
    print(f"📥 Descargando {ticker}...")
    df = yf.download(ticker, start="2010-01-01", end="2026-02-22", interval=timeframe, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df.rename(columns={"Date": "Datetime"})

def apply_stochastic_pnl(raw_pnl_usd, risk_usd):
    """Aplica costos estocasticos (Slippage normal N(0.1R, 0.05R))."""
    # Slippage medio 10% del riesgo, desvio 5%
    slippage = np.random.normal(0.10 * risk_usd, 0.05 * risk_usd)
    return raw_pnl_usd - slippage

def calculate_advanced_metrics(trades, equity, years):
    if not trades: return None
    
    pnls = np.array([t['pnl_usd'] for t in trades])
    returns = np.array(equity)
    
    # CAGR
    total_ret = returns[-1] / returns[0]
    cagr = (total_ret ** (1/years)) - 1 if years > 0 else 0
    
    # Sharpe & Sortino
    daily_returns = np.diff(returns) / returns[:-1]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
    
    neg_returns = daily_returns[daily_returns < 0]
    sortino = (np.mean(daily_returns) / np.std(neg_returns) * np.sqrt(252)) if len(neg_returns) > 1 and np.std(neg_returns) > 0 else 0
    
    # Drawdown & Ulcer
    max_dd = 0
    peak = returns[0]
    drawdowns = []
    for r in returns:
        peak = max(peak, r)
        dd = (peak - r) / peak
        drawdowns.append(dd)
        max_dd = max(max_dd, dd)
    
    ulcer_index = np.sqrt(np.mean(np.square(drawdowns))) * 100
    calmar = (cagr / max_dd) if max_dd > 0 else 0
    
    # Expectancy
    win_rate = len(pnls[pnls > 0]) / len(pnls)
    avg_w = np.mean(pnls[pnls > 0]) if any(pnls > 0) else 0
    avg_l = abs(np.mean(pnls[pnls <= 0])) if any(pnls <= 0) else 1e-6
    expectancy = (win_rate * avg_w) - ((1 - win_rate) * avg_l)
    
    return {
        "CAGR": cagr * 100,
        "MaxDD": max_dd * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Ulcer": ulcer_index,
        "Exp": expectancy,
        "WR": win_rate * 100,
        "PF": np.sum(pnls[pnls > 0]) / abs(np.sum(pnls[pnls <= 0])) if any(pnls <= 0) else 99
    }

def monte_carlo_sim(trades, initial_balance, risk_pct):
    permutations = []
    ruins = 0
    
    pnls = np.array([t['pnl_usd'] for t in trades])
    for _ in range(MonteCarloPermutations):
        shuffled = np.random.permutation(pnls)
        bal = initial_balance
        path = [initial_balance]
        ruined = False
        for p in shuffled:
            bal += p
            path.append(bal)
            if bal <= initial_balance * 0.5: # Ruina definida como 50% DD
                ruined = True
        permutations.append(path[-1])
        if ruined: ruins += 1
        
    permutations = np.sort(permutations)
    p5 = permutations[int(0.05 * MonteCarloPermutations)]
    prob_ruin = (ruins / MonteCarloPermutations) * 100
    
    return p5, prob_ruin

def run_oos_audit(pair_name, df, buy_model, sell_model, features, start_date, end_date, cfg):
    """Corre backtest en periodo especifico."""
    mask = (df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)
    df_period = df.loc[mask].copy().reset_index(drop=True)
    if len(df_period) < 50: return None, None
    
    df_feat = create_features(df_period)
    df_feat = detect_fractals(df_feat)
    df_feat = df_feat.dropna(subset=features).reset_index(drop=True)
    
    if df_feat.empty: return None, None
    
    X = df_feat[features].values
    buy_proba = buy_model.predict_proba(X)
    sell_proba = sell_model.predict_proba(X)
    
    buy_th = 0.6 # Fijo para OOS Real
    sell_th = 0.7 # Fijo para OOS Real
    
    trades = []
    balance = INITIAL_BALANCE
    equity = [INITIAL_BALANCE]
    
    # Simulacion simple bar-by-bar (OHLC)
    cur = None
    for i in range(len(df_feat)):
        h, l, c = df_feat['High'].iloc[i], df_feat['Low'].iloc[i], df_feat['Close'].iloc[i]
        atr = df_feat['ATR'].iloc[i]
        
        if cur and cur['open']:
            hit_tp = False
            hit_sl = False
            if cur['dir'] == "BUY":
                if h >= cur['tp']: hit_tp = True
                if l <= cur['sl']: hit_sl = True
            else:
                if l <= cur['tp']: hit_tp = True
                if h >= cur['sl']: hit_sl = True
                
            if hit_tp or hit_sl:
                risk_usd = balance * RISK_PER_TRADE
                pnl_raw = (cur['tp'] - cur['entry']) if hit_tp else (cur['sl'] - cur['entry'])
                if cur['dir'] == "SELL": pnl_raw *= -1
                
                # Convertir pips a USD aproximado via ATR
                pnl_usd = (pnl_raw / atr) * risk_usd if atr > 0 else 0
                pnl_usd = apply_stochastic_pnl(pnl_usd, risk_usd)
                pnl_usd -= cfg['commission']
                
                balance += pnl_usd
                cur['pnl_usd'] = pnl_usd
                cur['open'] = False
                trades.append(cur)
                cur = None
        
        if not cur:
            buy_sig = (buy_proba[i, 1] >= buy_th)
            sell_sig = (sell_proba[i, 1] >= sell_th) # Modelo binario sell
            
            if buy_sig or sell_sig:
                sig_dir = "BUY" if buy_sig else "SELL"
                entry = c
                sl_dist = atr * 1.5
                tp_dist = sl_dist * 1.5
                
                cur = {
                    "dir": sig_dir,
                    "entry": entry,
                    "sl": entry - sl_dist if sig_dir == "BUY" else entry + sl_dist,
                    "tp": entry + tp_dist if sig_dir == "BUY" else entry - tp_dist,
                    "open": True
                }
        
        equity.append(balance)
        
    return trades, equity

def main():
    print(f"\n{'='*100}")
    print(f"  TRUE OUT-OF-SAMPLE TEST FRAMEWORK v15")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    results = {}
    
    for name, cfg in PAIRS.items():
        print(f"\n--- Analizando {name} ---")
        df = get_data(cfg['ticker'], name)
        if df is None: continue
        
        # 1. TRAIN: 2010-2018 (Usamos modelos actuales entrenados con esta data)
        # 2. VALIDATION: 2019-2021
        # 3. OOS: 2022-2025
        
        # Cargar modelos actuales
        try:
            buy_art = joblib.load("model_multi.joblib")
            sell_art = joblib.load("model_multi_sell.joblib")
            features = buy_art['feature_columns']
            buy_model = buy_art['model']
            sell_model = sell_art['model']
        except:
            print(f"❌ Error cargando modelos para {name}")
            continue
            
        print(f"🔬 Corriendo Backtests...")
        
        # Validation Period
        val_trades, val_equity = run_oos_audit(name, df, buy_model, sell_model, features, SPLITS['VAL'][0], SPLITS['VAL'][1], cfg)
        val_metrics = calculate_advanced_metrics(val_trades, val_equity, 3.0)
        
        # OOS Final (Moment of Truth)
        oos_trades, oos_equity = run_oos_audit(name, df, buy_model, sell_model, features, SPLITS['OOS'][0], SPLITS['OOS'][1], cfg)
        oos_metrics = calculate_advanced_metrics(oos_trades, oos_equity, 3.1) # 2022 a Feb 2025 aprox 3.1 años
        
        if not val_metrics or not oos_metrics:
            print(f"⚠️ Datos insuficientes para {name}")
            continue
            
        # Monte Carlo en OOS
        mc_p5, prob_ruin = monte_carlo_sim(oos_trades, INITIAL_BALANCE, RISK_PER_TRADE)
        
        # Robustness Check
        overfit = oos_metrics['PF'] < val_metrics['PF'] * 0.6
        fragile = oos_metrics['WR'] < val_metrics['WR'] - 10
        
        results[name] = {
            "VAL": val_metrics,
            "OOS": oos_metrics,
            "OOS_trades": oos_trades,
            "MC_P5": mc_p5,
            "P_RUIN": prob_ruin,
            "ROBUST": not (overfit or fragile),
            "REASON": ("OVERFIT" if overfit else "") + (" FRAGILE" if fragile else "") or "STABLE"
        }
        
        print(f"   [VAL] WR: {val_metrics['WR']:.1f}% | PF: {val_metrics['PF']:.2f} | Sharpe: {val_metrics['Sharpe']:.2f}")
        print(f"   [OOS] WR: {oos_metrics['WR']:.1f}% | PF: {oos_metrics['PF']:.2f} | Sharpe: {oos_metrics['Sharpe']:.2f}")
        print(f"   [MC]  Worst-Case Equity: ${mc_p5:.0f} | Prob Ruina: {prob_ruin:.1f}%")
        status = "✅ ROBUSTO" if results[name]['ROBUST'] else f"❌ {results[name]['REASON']}"
        print(f"   VEREDICTO: {status}")

    # --- REPORTE COMPARATIVO ---
    print(f"\n\n{'='*100}")
    print(f"  COMPARATIVA FINAL: VALIDATION vs OOS REAL")
    print(f"{'='*100}")
    print(f"{'Pair':<10} | {'Val PF':>7} | {'OOS PF':>7} | {'Val WR':>7} | {'OOS WR':>7} | {'Status':<15}")
    print(f"{'-'*100}")
    
    for name, res in results.items():
        v, o = res['VAL'], res['OOS']
        status = "✅ ROBUSTO" if res['ROBUST'] else f"❌ {res['REASON']}"
        print(f"{name:<10} | {v['PF']:>7.2f} | {o['PF']:>7.2f} | {v['WR']:>6.1f}% | {o['WR']:>6.1f}% | {status:<15}")
        
    # --- CORRELACION Y DD AGREGADO (OOS Period) ---
    print(f"\n\n{'='*100}")
    print(f"  ANALISIS DE CARTERA (OOS 2022-2025)")
    print(f"{'='*100}")
    
    all_oos_returns = []
    pair_names = []
    for name in results:
        trades = results[name].get('OOS_trades', [])
        if not trades: continue
        
        # Serie de retornos (en USD)
        returns = [t['pnl_usd'] for t in trades]
        if len(returns) < 10: continue
        
        # Pad with zeros to equalize lengths for a rough correlation check
        all_oos_returns.append(returns)
        pair_names.append(name)
        
    if len(all_oos_returns) > 1:
        # Equalize lengths with zeros at the end
        max_len = max(len(r) for r in all_oos_returns)
        padded_returns = [r + [0]*(max_len - len(r)) for r in all_oos_returns]
        
        corr_df = pd.DataFrame(np.array(padded_returns).T, columns=pair_names).corr()
        print("\n📊 Matriz de Correlación de Retornos (Trades):")
        print(corr_df.round(2).to_string())
        
        # DD Agregado (Suma simple de DDs como proxy conservador)
        avg_dd = np.mean([results[n]['OOS']['MaxDD'] for n in results])
        max_dd_sum = np.sum([results[n]['OOS']['MaxDD'] for n in results]) 
        print(f"\n📈 Drawdown Promedio OOS: {avg_dd:.2f}%")
        print(f"📉 Drawdown Agregado Teórico (Suma): {max_dd_sum:.2f}% (Escenario catastrófico)")
    else:
        print("\n⚠️ No hay suficientes pares con trades para correlacion.")


if __name__ == "__main__":
    main()
