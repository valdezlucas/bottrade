"""
Shitcoin Reversal Trainer (Phase 6)
Trains a dedicated Counter-Trend LightGBM GPU model designed to short parabolic pumps
and buy extreme dumps on volatile altcoins/memecoins.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

from features import create_features
from fractals import detect_fractals

# Cesta de Altcoins altamente volátiles y Memecoins recientes
SHITCOIN_PAIRS = [
    "WIF-USD", "PEPE-USD", "BONK-USD", "FLOKI-USD", "ORDI-USD",
    "INJ-USD", "RNDR-USD", "OP-USD", "ARB-USD", "SUI-USD",
    "APT-USD", "TIA-USD", "SEI-USD", "GALA-USD", "FET-USD"
]

def download_data(ticker, tf="1h"):
    print(f"    📥 Descargando {ticker} [{tf}]...")
    try:
        end = datetime.now()
        if tf == "1d":
            df = yf.download(ticker, start="2020-01-01", interval="1d", progress=False)
        else:
            start = end - timedelta(days=725)
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
            ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume":"sum"}
            df = df.resample("4h").agg(ohlc).dropna().reset_index()
            
        return df
    except Exception as e:
        print(f"      ❌ Error {ticker}: {e}")
        return None

def engineer_shitcoin_features(df):
    """
    Agrega métricas específicas para detectar Parábolas y Dumps.
    """
    df = df.copy()
    
    # Rendimiento acumulado en N periodos (Mide el Pump/Dump reciente)
    df["ret_24h"] = df["Close"].pct_change(periods=24) # Suponiendo 1H tf, para 4H seria pct_change(6) adaptado dinamicamente no importa, dejamos pct simple
    df["pump_4_bars"] = df["Close"].pct_change(periods=4)
    df["pump_8_bars"] = df["Close"].pct_change(periods=8)
    
    # Ratio de cuerpo vs mecha (muy importante en tops de shitcoins)
    df["body"] = abs(df["Close"] - df["Open"])
    df["range"] = df["High"] - df["Low"]
    df["upper_wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    
    # Prevención division por cero
    df["range"] = df["range"].replace(0, 1e-9)
    df["upper_wick_ratio"] = df["upper_wick"] / df["range"]
    df["lower_wick_ratio"] = df["lower_wick"] / df["range"]
    
    # Anomalia de Volumen (Spike)
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_spike"] = df["Volume"] / df["vol_sma_20"].replace(0, 1e-9)
    
    return df

def label_reversals(df, lookahead=12):
    """
    Etiquetado asimétrico exclusivo para Reversiones Violentas.
    Buscamos un Risk:Reward de 1:2. Riesgo corto (SL pegado al mechozine extremo).
    Target: 0 = HOLD, 1 = BUY REVERSAL (Atrapar cuchillo), 2 = SELL REVERSAL (Shortear Pump)
    """
    df = df.copy()
    df['target'] = 0

    for i in range(len(df) - lookahead):
        entry = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i] if 'ATR' in df.columns and not np.isnan(df['ATR'].iloc[i]) else entry * 0.03
        
        # SL apretado para reversión (1.5 ATR). TP amplio (3 ATR) para capturar el colapso.
        sl_dist = atr * 1.5 
        tp_dist = atr * 3.0

        hit_buy_tp, hit_buy_sl = False, False
        hit_sell_tp, hit_sell_sl = False, False

        for j in range(i + 1, i + 1 + lookahead):
            high = df['High'].iloc[j]
            low = df['Low'].iloc[j]

            # LONG (Comprar Dump)
            if not hit_buy_sl and not hit_buy_tp:
                if low <= entry - sl_dist: hit_buy_sl = True
                elif high >= entry + tp_dist: hit_buy_tp = True

            # SHORT (Shortear Pump)
            if not hit_sell_sl and not hit_sell_tp:
                if high >= entry + sl_dist: hit_sell_sl = True
                elif low <= entry - tp_dist: hit_sell_tp = True

            if (hit_buy_tp or hit_buy_sl) and (hit_sell_tp or hit_sell_sl): break

        # Validamos con la condición geométrica (Fractal / Exhaustion y Pump/Dump extremo previo)
        pump_magn = df["pump_4_bars"].iloc[i] if not np.isnan(df["pump_4_bars"].iloc[i]) else 0
        
        # BUY REVERSAL: Necesitamos que haya dumpeado (pump_4_bars < -0.05) y toque TP antes que SL
        if hit_buy_tp and not hit_buy_sl and pump_magn < -0.05:
            df.at[i, 'target'] = 1
            
        # SELL REVERSAL: Necesitamos que haya pumpeado (pump_4_bars > 0.05) y toque TP antes que SL
        elif hit_sell_tp and not hit_sell_sl and pump_magn > 0.05:
            df.at[i, 'target'] = 2

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, choices=["1d", "4h", "1h", "ALL"], default="1h")
    args = parser.parse_args()

    tfs_to_run = ["1d", "4h", "1h"] if args.tf == "ALL" else [args.tf]

    print("="*60)
    print("  🚀 ENTRENAMIENTO SHITCOIN PUMP & DUMP REVERSAL")
    print("="*60)

    for tf in tfs_to_run:
        print(f"\n\n{'='*40}\n  INICIANDO PIPELINE -> {tf.upper()}\n{'='*40}")
        all_data = []
        lookahead = {"1d": 5, "4h": 12, "1h": 24}.get(tf, 12)

        for ticker in SHITCOIN_PAIRS:
            df = download_data(ticker, tf)
            if df is None: continue
            
            try:
                # Features base v16
                df = create_features(df)
                df = detect_fractals(df)
                # Shitcoin features
                df = engineer_shitcoin_features(df)
            except Exception as e:
                print(f"      ❌ Error features {ticker}: {e}")
                continue
                
            df.dropna(inplace=True)
            df = label_reversals(df, lookahead=lookahead)
            df["Asset"] = ticker
            all_data.append(df)

        if not all_data:
            print("❌ No se descargaron datos.")
            return

        df_final = pd.concat(all_data, ignore_index=True)
        print(f"\n  ✅ Datos unidos: {len(df_final)} filas")

        # Balanceo asimétrico para reversiones raras
        buy_df = df_final[df_final['target'] == 1]
        sell_df = df_final[df_final['target'] == 2]
        hold_df = df_final[df_final['target'] == 0]
        
        min_class = min(len(buy_df), len(sell_df))
        if min_class == 0:
            print("❌ No hay suficientes ejemplos de reversión. Ajustando parámetros.")
            continue
            
        # Nos quedamos con 2x HOLDs vs señales activas para no matar el modelo a Falsos Positivos
        hold_sample = hold_df.sample(n=min_class * 2, random_state=42)
        df_balanced = pd.concat([hold_sample, buy_df, sell_df]).sort_index()
        
        print(f"    📏 Clases balanceadas -> HOLD: {len(hold_sample)}, BUY REV: {len(buy_df)}, SELL REV: {len(sell_df)}")

        EXCLUDE_COLS = ["Datetime", "Date", "Open", "High", "Low", "Close", "Volume", "target", "Asset", "vol_sma_20", "body", "range", "upper_wick", "lower_wick"]
        feature_cols = [c for c in df_balanced.columns if c not in EXCLUDE_COLS]

        X = df_balanced[feature_cols].astype(np.float32)
        y = df_balanced["target"]

        print(f"  ⚙️ Entrenando modelo LightGBM Reversión {tf.upper()} (GPU)...")
        lgbm = LGBMClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=128,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            device='gpu' # Requiere drivers NVIDIA instalados
        )

        calibrated_model = CalibratedClassifierCV(estimator=lgbm, method='isotonic', cv=3)
        calibrated_model.fit(X, y)

        artifact = {
            "model": calibrated_model,
            "feature_columns": feature_cols,
            "threshold": 0.50, # Umbral alto para confirmar reversiones claras
            "lookahead": lookahead
        }

        save_path = f"model_shitcoins_{tf}.joblib"
        joblib.dump(artifact, save_path)
        print(f"  ✅ Modelo entrenado y guardado -> {save_path}")

    print("\n🎉 Entrenamiento Shitcoin finalizado exitosamente!")

if __name__ == "__main__":
    main()
