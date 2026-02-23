"""
Crypto AI - Single Script Pipeline
Downloads Top 15 Crypto data, generates features, trains LightGBM GPU,
and saves the isolated model.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings

# Forzar utf-8 en Windows para evitar UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

from features import create_features
from fractals import detect_fractals

warnings.filterwarnings('ignore')

CRYPTO_PAIRS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "TRX-USD",
    "LINK-USD", "BCH-USD", "LTC-USD", "MATIC-USD", "XLM-USD"
]

def download_crypto_data(ticker, tf="1d"):
    print(f"    📥 Descargando {ticker} [{tf}]...")
    try:
        if tf == "1d":
            # Máxima historia posible para daily
            df = yf.download(ticker, start="2015-01-01", interval="1d", progress=False)
        else:
            # yfinance limita datos intraday a 730 días máximos
            end = datetime.now()
            start = end - timedelta(days=725)
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1h", progress=False)
            
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        
        # Unificar columna de tiempo
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "Datetime"})
            
        if tf == "4h":
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
            ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            df = df.resample("4h").agg(ohlc).dropna().reset_index()
            
        return df
    except Exception as e:
        print(f"      ❌ Error descargando {ticker}: {e}")
        return None

def label_data(df, lookahead=10):
    """
    Multiclass labeling dinámico:
    0 = HOLD
    1 = BUY (Alcanza TP antes que SL)
    2 = SELL (Alcanza TP antes que SL)
    """
    df = df.copy()
    df['target'] = 0
    df['potential_win'] = 0.0
    df['potential_loss'] = 0.0

    for i in range(len(df) - lookahead):
        entry = df['Close'].iloc[i]
        # Crypto ATRs son más grandes, usamos 2.0x SL por volatilidad crypto
        atr = df['ATR'].iloc[i] if not np.isnan(df['ATR'].iloc[i]) else entry * 0.02
        sl_dist = max(atr * 2.0, entry * 0.01)
        tp_dist = sl_dist * 1.5

        hit_buy_tp = False
        hit_buy_sl = False
        hit_sell_tp = False
        hit_sell_sl = False

        max_high = entry
        min_low = entry

        for j in range(i + 1, i + 1 + lookahead):
            high = df['High'].iloc[j]
            low = df['Low'].iloc[j]
            max_high = max(max_high, high)
            min_low = min(min_low, low)

            # BUY Eval
            if not hit_buy_sl and not hit_buy_tp:
                if low <= entry - sl_dist: hit_buy_sl = True
                elif high >= entry + tp_dist: hit_buy_tp = True

            # SELL Eval
            if not hit_sell_sl and not hit_sell_tp:
                if high >= entry + sl_dist: hit_sell_sl = True
                elif low <= entry - tp_dist: hit_sell_tp = True

            if (hit_buy_tp or hit_buy_sl) and (hit_sell_tp or hit_sell_sl):
                break

        # Determinación de Target (Multiclass)
        if hit_buy_tp and not hit_buy_sl:
            df.at[i, 'target'] = 1
            df.at[i, 'potential_win'] = tp_dist
        elif hit_sell_tp and not hit_sell_sl:
            df.at[i, 'target'] = 2
            df.at[i, 'potential_win'] = tp_dist
        else:
            df.at[i, 'target'] = 0
            df.at[i, 'potential_loss'] = sl_dist

    # Filtrar solo entradas que concuerden con fractales de la estrategia v16
    mask_buy_valid = (df['fractal_low'] == 1) & (df['target'] == 1)
    mask_sell_valid = (df['fractal_high'] == 1) & (df['target'] == 2)
    mask_hold = (df['target'] == 0)

    valid_idx = df[mask_buy_valid | mask_sell_valid | mask_hold].index
    df_filtered = df.loc[valid_idx].copy()

    return df_filtered

def balance_dataset(df):
    """
    Sub-muestreo masivo de HOLDs para equilibrar las clases.
    """
    buy_df = df[df['target'] == 1]
    sell_df = df[df['target'] == 2]
    hold_df = df[df['target'] == 0]

    min_class_size = min(len(buy_df), len(sell_df))
    if min_class_size == 0:
        return df # Fallback if empty

    hold_size = int(min_class_size * 1.5)

    print(f"    📏 Clases originales -> HOLD: {len(hold_df)}, BUY: {len(buy_df)}, SELL: {len(sell_df)}")
    
    if len(hold_df) > hold_size:
        hold_df = hold_df.sample(n=hold_size, random_state=42)
    
    balanced_df = pd.concat([hold_df, buy_df, sell_df]).sort_index()
    print(f"    ⚖️ Clases balanceadas -> HOLD: {len(hold_df)}, BUY: {len(buy_df)}, SELL: {len(sell_df)}\n")
    return balanced_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, choices=["1d", "4h", "1h", "ALL"], default="1d")
    args = parser.parse_args()

    # Timeframes to process
    tfs_to_run = ["1d", "4h", "1h"] if args.tf == "ALL" else [args.tf]

    print("="*60)
    print("  🚀 ENTRENAMIENTO AISLADO CRYPTO (TOP 15)")
    print("="*60)

    for tf in tfs_to_run:
        print(f"\n\n{'='*40}\n  INICIANDO PIPELINE -> {tf.upper()}\n{'='*40}")
        
        all_data = []
        lookahead = {"1d": 10, "4h": 20, "1h": 24}.get(tf, 10)

        for ticker in CRYPTO_PAIRS:
            df = download_crypto_data(ticker, tf)
            if df is None: continue
            
            try:
                df = create_features(df)
                df = detect_fractals(df)
            except Exception as e:
                print(f"      ❌ Error features {ticker}: {e}")
                continue
                
            df.dropna(inplace=True)
            df = label_data(df, lookahead=lookahead)
            df["Asset"] = ticker
            all_data.append(df)

        if not all_data:
            print("❌ No se descargaron datos.")
            return

        df_final = pd.concat(all_data, ignore_index=True)
        df_final = df_final.sort_values(by="Datetime").reset_index(drop=True)

        print(f"\n  ✅ Datos unidos: {len(df_final)} filas")

        # Balanceo
        df_balanced = balance_dataset(df_final)

        EXCLUDE_COLS = ["Datetime", "Date", "Open", "High", "Low", "Close", "Volume", "target", "potential_win", "potential_loss", "Asset"]
        feature_cols = [c for c in df_balanced.columns if c not in EXCLUDE_COLS]

        X = df_balanced[feature_cols].astype(np.float32)
        y = df_balanced["target"]

        print(f"  ⚙️ Entrenando modelo LightGBM GPU de {tf.upper()} (Clases: 0,1,2)...")
        # Base estimator
        lgbm = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            device='gpu'
        )

        calibrated_model = CalibratedClassifierCV(estimator=lgbm, method='isotonic', cv=3)
        calibrated_model.fit(X, y)

        # Baseline threshold conservative 
        artifact = {
            "model": calibrated_model,
            "feature_columns": feature_cols,
            "threshold": 0.45, 
            "lookahead": lookahead
        }

        save_path = f"model_crypto_{tf}.joblib"
        joblib.dump(artifact, save_path)
        print(f"  ✅ Modelo entrenado y calibrado -> {save_path}")

    print("\n🎉 Entrenamiento Crypto finalizado exitosamente!")

if __name__ == "__main__":
    main()
