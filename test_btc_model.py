"""Quick test of BTC-specific daily model vs forex-trained daily model."""
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from features import create_features
from fractals import detect_fractals


def quick_backtest(model_path, sell_model_path, label):
    # Download BTC daily
    end = datetime.now()
    start = end - timedelta(days=1500)
    df = yf.download("BTC-USD", start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index(drop=True)

    # Load models
    art = joblib.load(model_path)
    sell_art = joblib.load(sell_model_path)
    model_buy = art["model"]
    model_sell = sell_art["model"]
    feature_cols = art["feature_columns"]
    threshold = art["threshold"]
    sell_th = sell_art["threshold"]

    # Prepare
    df_work = df[["Open", "High", "Low", "Close"]].copy()
    if "Volume" in df.columns:
        df_work["Volume"] = df["Volume"].values
    df_work = create_features(df_work)
    df_work = detect_fractals(df_work)
    df_work = df_work.dropna(subset=feature_cols).reset_index(drop=True)

    X = df_work[feature_cols].values
    buy_proba = model_buy.predict_proba(X)
    buy_preds = buy_proba.argmax(axis=1)
    buy_max = buy_proba.max(axis=1)
    sell_proba = model_sell.predict_proba(X)
    sell_preds = sell_proba.argmax(axis=1)
    sell_max = sell_proba.max(axis=1)

    # Quick sim
    balance = 10000.0
    peak = 10000.0
    risk = 0.005
    rr = 1.5
    wins = 0
    losses = 0
    current = None

    for i in range(len(df_work)):
        h = df_work["High"].iloc[i]
        l = df_work["Low"].iloc[i]
        c = df_work["Close"].iloc[i]
        atr = df_work["ATR"].iloc[i]
        if np.isnan(atr) or atr <= 0:
            continue

        if current is not None:
            if current["dir"] == "BUY":
                if h >= current["tp"]:
                    pnl = (current["tp"] - current["entry"]) * (balance * risk / atr)
                    balance += pnl
                    wins += 1
                    current = None
                elif l <= current["sl"]:
                    pnl = (current["sl"] - current["entry"]) * (balance * risk / atr)
                    balance += pnl
                    losses += 1
                    current = None
            else:
                if l <= current["tp"]:
                    pnl = (current["entry"] - current["tp"]) * (balance * risk / atr)
                    balance += pnl
                    wins += 1
                    current = None
                elif h >= current["sl"]:
                    pnl = (current["entry"] - current["sl"]) * (balance * risk / atr)
                    balance += pnl
                    losses += 1
                    current = None

        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        if dd >= 25:
            break
        peak = max(peak, balance)

        if current is None:
            sig = None
            if buy_preds[i] == 1 and buy_max[i] >= threshold:
                sig = "BUY"
            elif sell_preds[i] == 1 and sell_max[i] >= sell_th:
                sig = "SELL"
            if sig:
                sl_d = atr
                tp_d = sl_d * rr
                if sig == "BUY":
                    current = {"dir": "BUY", "entry": c, "sl": c - sl_d, "tp": c + tp_d}
                else:
                    current = {"dir": "SELL", "entry": c, "sl": c + sl_d, "tp": c - tp_d}

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    ret = (balance / 10000 - 1) * 100

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Threshold: {threshold} | Candles: {len(df_work)}")
    print(f"  Trades: {total} | Wins: {wins} ({wr:.1f}%) | Losses: {losses}")
    print(f"  Return: {ret:+.2f}%")
    print(f"  Final: ${balance:,.2f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    print("=== BTC-USD Daily: Modelo Forex vs Modelo BTC-Especifico ===\n")

    # Test 1: Forex-trained model on BTC
    quick_backtest("model_multi.joblib", "model_multi_sell.joblib",
                   "BTC Daily — FOREX-TRAINED MODEL")

    # Test 2: BTC-specific model
    quick_backtest("model_btc_daily.joblib", "model_btc_daily_sell.joblib",
                   "BTC Daily — BTC-SPECIFIC MODEL")
