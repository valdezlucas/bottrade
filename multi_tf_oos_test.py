"""
OOS Test for Lower Timeframes (4H / 1H)
Downloads intraday data (max 730 days via yfinance), resamples,
applies the model, and calculates OOS performance.
"""

import os
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from extended_oos_test import (
    ALPHA_SIZING,
    INITIAL_BALANCE,
    LOOKAHEAD,
    MAX_RISK_PCT,
    RISK_PER_TRADE,
    apply_stochastic_pnl,
    calculate_metrics,
)
from features import create_features
from fractals import detect_fractals

PAIRS = {
    # Forex Robusto
    "GBPUSD": {"ticker": "GBPUSD=X", "pip": 0.0001, "commission": 15.0},
    "NZDUSD": {"ticker": "NZDUSD=X", "pip": 0.0001, "commission": 15.0},
    "AUDUSD": {"ticker": "AUDUSD=X", "pip": 0.0001, "commission": 15.0},
    "USDCAD": {"ticker": "USDCAD=X", "pip": 0.0001, "commission": 15.0},
    "USDCHF": {"ticker": "USDCHF=X", "pip": 0.0001, "commission": 15.0},
    "USDJPY": {"ticker": "USDJPY=X", "pip": 0.01, "commission": 15.0},
    "EURJPY": {"ticker": "EURJPY=X", "pip": 0.01, "commission": 15.0},
    "GBPJPY": {"ticker": "GBPJPY=X", "pip": 0.01, "commission": 15.0},
    # Acciones Robustas
    "MSFT": {"ticker": "MSFT", "pip": 0.01, "commission": 1.0},
    "TSLA": {"ticker": "TSLA", "pip": 0.01, "commission": 1.0},
    "PG": {"ticker": "PG", "pip": 0.01, "commission": 1.0},
    "XOM": {"ticker": "XOM", "pip": 0.01, "commission": 1.0},
}


def download_1h_data(ticker, days=720):
    print(f"    📥 Descargando {ticker}...")
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1h",
            progress=False,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Datetime"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "Datetime"})
        return df
    except Exception:
        return None


def resample_to_4h(df_1h):
    if df_1h is None or df_1h.empty:
        return None
    df = df_1h.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
    ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        ohlc["Volume"] = "sum"
    df_4h = df.resample("4h").agg(ohlc).dropna().reset_index()
    return df_4h


def run_backtest_tf(df_feat, art, feature_cols, cfg):
    mask = df_feat["Datetime"] >= "2024-01-01"  # Simulate recent period OOS
    df_p = df_feat.loc[mask].copy().reset_index(drop=True)
    if len(df_p) < 50:
        return None, None
    X = df_p[feature_cols].values
    model = art["model"]
    probas = model.predict_proba(X)
    th = art.get("threshold", 0.5)

    trades = []
    balance = INITIAL_BALANCE
    equity = [INITIAL_BALANCE]

    for i in range(len(df_p) - LOOKAHEAD):
        prob_buy, prob_sell = probas[i, 1], probas[i, 2]
        buy_sig, sell_sig = prob_buy >= th, prob_sell >= th
        if not buy_sig and not sell_sig:
            equity.append(balance)
            continue

        if buy_sig and not sell_sig:
            sig_dir, conf = "BUY", prob_buy
        elif sell_sig and not buy_sig:
            sig_dir, conf = "SELL", prob_sell
        else:
            if (prob_buy - th) >= (prob_sell - th):
                sig_dir, conf = "BUY", prob_buy
            else:
                sig_dir, conf = "SELL", prob_sell

        if conf > th and th < 1.0:
            scaled = (
                RISK_PER_TRADE * ((conf - th) / (1.0 - th)) * ALPHA_SIZING
                + RISK_PER_TRADE
            )
        else:
            scaled = RISK_PER_TRADE
        risk_usd = balance * min(scaled, MAX_RISK_PCT)

        entry = df_p["Close"].iloc[i]
        atr = (
            df_p["ATR"].iloc[i] if not np.isnan(df_p["ATR"].iloc[i]) else entry * 0.005
        )
        sl_dist = max(atr * 1.5, entry * 0.001)  # Wider SL for intraday
        tp_dist = sl_dist * 1.5

        hit_tp, hit_sl = False, False
        for j in range(i + 1, i + 1 + LOOKAHEAD):
            if j >= len(df_p):
                break
            high, low = df_p["High"].iloc[j], df_p["Low"].iloc[j]
            if sig_dir == "BUY":
                if low <= (entry - sl_dist):
                    hit_sl = True
                    break
                elif high >= (entry + tp_dist):
                    hit_tp = True
                    break
            else:
                if high >= (entry + sl_dist):
                    hit_sl = True
                    break
                elif low <= (entry - tp_dist):
                    hit_tp = True
                    break

        raw_pnl = risk_usd * 1.5 if hit_tp else -risk_usd
        if not hit_tp and not hit_sl:
            last_close = df_p["Close"].iloc[i + LOOKAHEAD]
            if sig_dir == "BUY":
                raw_pnl = ((last_close - entry) / sl_dist) * risk_usd
            else:
                raw_pnl = ((entry - last_close) / sl_dist) * risk_usd

        pnl_net = apply_stochastic_pnl(raw_pnl, risk_usd) - cfg["commission"]
        balance = max(balance + pnl_net, 1.0)
        equity.append(balance)
        trades.append({"dir": sig_dir, "pnl_usd": pnl_net})

    return trades, equity


def main():
    print(f"\n{'='*60}\n  MULTI-TIMEFRAME RECENT OOS EVALUATION\n{'='*60}\n")
    for tf in ["4h", "1h"]:
        print(f"\n\nEvaluando Modelo {tf.upper()}...")
        try:
            art = joblib.load(f"model_{tf}.joblib")
            feature_cols = art["feature_columns"]
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        results = {}
        for name, cfg in PAIRS.items():
            df_1h = download_1h_data(cfg["ticker"])
            if df_1h is None:
                continue
            df = resample_to_4h(df_1h) if tf == "4h" else df_1h.copy()

            try:
                df_feat = create_features(df.copy())
                df_feat = detect_fractals(df_feat)
                df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
            except Exception:
                continue

            trades, equity = run_backtest_tf(df_feat, art, feature_cols, cfg)
            m = calculate_metrics(trades, equity, 1.0)  # Approx 1 year of recent OOS
            if m:
                results[name] = m
                print(
                    f"  {name:6s} | PF: {m['PF']:>5.2f} | WR: {m['WR']:>5.1f}% | Sharpe: {m['Sharpe']:>5.2f} | Trades: {m['n_trades']}"
                )
            else:
                print(f"  {name:6s} | ⚠️ Sin trades")


if __name__ == "__main__":
    main()
