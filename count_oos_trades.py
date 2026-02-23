import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from true_oos_test import PAIRS, SPLITS, run_oos_audit

buy_art = joblib.load('model_multi.joblib')
sell_art = joblib.load('model_multi_sell.joblib')
features = buy_art['feature_columns']
buy_model = buy_art['model']
sell_model = sell_art['model']

total_oos = 0
print(f"{'Pair':<10} | {'OOS Trades':<10}")
print("-" * 25)

for name, cfg in PAIRS.items():
    ticker = cfg['ticker']
    df = yf.download(ticker, start="2010-01-01", end="2026-02-22", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={'Date': 'Datetime'})
    
    trades, _ = run_oos_audit(name, df, buy_model, sell_model, features, SPLITS['OOS'][0], SPLITS['OOS'][1], cfg)
    count = len(trades) if trades else 0
    print(f"{name:<10} | {count:<10}")
    total_oos += count

print("-" * 25)
print(f"{'TOTAL':<10} | {total_oos:<10}")
