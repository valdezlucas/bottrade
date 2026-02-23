"""
Crypto True OOS Test
Evaluates the 'model_crypto_{tf}.joblib' performance across TOP 15 cryptos
on recent unseen data.
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from features import create_features
from fractals import detect_fractals
from extended_oos_test import apply_stochastic_pnl, calculate_metrics, INITIAL_BALANCE, RISK_PER_TRADE, MAX_RISK_PCT, ALPHA_SIZING, LOOKAHEAD

CRYPTO_PAIRS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "TRX-USD",
    "LINK-USD", "BCH-USD", "LTC-USD", "MATIC-USD", "XLM-USD"
]

def download_data(ticker, tf, days=720):
    print(f"    📥 Descargando {ticker} [{tf.upper()}]...")
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        if tf == "1d":
            # For 1d, let's use last 2.5 years
            start = end - timedelta(days=900)
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", progress=False)
        else:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1h", progress=False)
            
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "Datetime"})
            
        if tf == "4h":
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
            ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            if "Volume" in df.columns: ohlc["Volume"] = "sum"
            df = df.resample("4h").agg(ohlc).dropna().reset_index()
            
        return df
    except Exception: return None

def run_crypto_backtest(df_feat, art, feature_cols, th_override=None):
    # Simulate recent 14 months OOS
    mask = (df_feat["Datetime"] >= "2024-01-01") 
    df_p = df_feat.loc[mask].copy().reset_index(drop=True)
    if len(df_p) < 50: return None, None
    X = df_p[feature_cols].astype(np.float32).values
    model = art["model"]
    probas = model.predict_proba(X)
    th = th_override if th_override else art.get("threshold", 0.45)
    lookahead = art.get("lookahead", 10)

    trades  = []
    balance = INITIAL_BALANCE
    equity  = [INITIAL_BALANCE]

    for i in range(len(df_p) - lookahead):
        prob_buy, prob_sell = probas[i, 1], probas[i, 2]
        buy_sig, sell_sig = prob_buy >= th, prob_sell >= th
        if not buy_sig and not sell_sig:
            equity.append(balance)
            continue
        
        if buy_sig and not sell_sig: sig_dir, conf = "BUY", prob_buy
        elif sell_sig and not buy_sig: sig_dir, conf = "SELL", prob_sell
        else:
            if (prob_buy - th) >= (prob_sell - th): sig_dir, conf = "BUY", prob_buy
            else: sig_dir, conf = "SELL", prob_sell

        if conf > th and th < 1.0: scaled = RISK_PER_TRADE * ((conf - th) / (1.0 - th)) * ALPHA_SIZING + RISK_PER_TRADE
        else: scaled = RISK_PER_TRADE
        risk_usd = balance * min(scaled, MAX_RISK_PCT)

        entry = df_p["Close"].iloc[i]
        atr = df_p["ATR"].iloc[i] if not np.isnan(df_p["ATR"].iloc[i]) else entry * 0.02
        sl_dist = max(atr * 2.0, entry * 0.01) # Crypto SL
        tp_dist = sl_dist * 1.5

        hit_tp, hit_sl = False, False
        for j in range(i + 1, i + 1 + lookahead):
            if j >= len(df_p): break
            high, low = df_p["High"].iloc[j], df_p["Low"].iloc[j]
            if sig_dir == "BUY":
                if low <= (entry - sl_dist): hit_sl = True; break
                elif high >= (entry + tp_dist): hit_tp = True; break
            else:
                if high >= (entry + sl_dist): hit_sl = True; break
                elif low <= (entry - tp_dist): hit_tp = True; break

        raw_pnl = risk_usd * 1.5 if hit_tp else -risk_usd
        if not hit_tp and not hit_sl:
            last_close = df_p["Close"].iloc[i + lookahead]
            if sig_dir == "BUY": raw_pnl = ((last_close - entry) / sl_dist) * risk_usd
            else: raw_pnl = ((entry - last_close) / sl_dist) * risk_usd

        # No commission explicitly mocked for crypto spot, but we add realistic friction
        pnl_net = apply_stochastic_pnl(raw_pnl, risk_usd)
        balance = max(balance + pnl_net, 1.0)
        equity.append(balance)
        trades.append({"dir": sig_dir, "pnl_usd": pnl_net})
    
    return trades, equity

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, choices=["1d", "4h", "1h", "ALL"], default="ALL")
    args = parser.parse_args()
    
    tfs = ["1d", "4h", "1h"] if args.tf == "ALL" else [args.tf]
    
    print(f"\n{'='*60}\n  TRUE OOS TEST: CRYPTO ECOSYSTEM (2024-2026)\n{'='*60}\n")
    
    for tf in tfs:
        model_name = f"model_crypto_{tf}.joblib"
        if not os.path.exists(model_name):
            print(f"❌ Modelo {model_name} no encontrado. Saltando.")
            continue
            
        print(f"\n\nEvaluando Modelo {tf.upper()}...")
        try:
            art = joblib.load(model_name)
            feature_cols = art["feature_columns"]
        except Exception as e: print(f"❌ Error: {e}"); continue
        
        results = {}
        for ticker in CRYPTO_PAIRS:
            df = download_data(ticker, tf)
            if df is None: continue
            
            try:
                df_feat = create_features(df.copy())
                df_feat = detect_fractals(df_feat)
                df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
            except Exception: continue
            
            trades, equity = run_crypto_backtest(df_feat, art, feature_cols)
            # Aproximar a años. 2024 a feb 2026 son ~1.15 años
            m = calculate_metrics(trades, equity, 1.15) 
            if m:
                results[ticker] = m
                print(f"  {ticker:10s} | PF: {m['PF']:>5.2f} | WR: {m['WR']:>5.1f}% | Sharpe: {m['Sharpe']:>5.2f} | Trades: {m['n_trades']}")
            else: print(f"  {ticker:10s} | ⚠️ Sin trades")

if __name__ == "__main__":
    main()
