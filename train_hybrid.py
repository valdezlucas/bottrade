"""
Hybrid Pipeline: Classifier + Regressor (Phase 7)
"""

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

from features import create_features
from fractals import detect_fractals


# Simulando download_data de forma directa por eficiencia (para prueba)
# En prod usaríamos los históricos descargados
def load_and_prepare_data():
    try:
        print("📥 Cargando dataset de acciones y forex...")
        df = pd.read_csv(
            "df_1d_joined.csv"
        )  # Usamos el dataset masivo que ya tenemos pre-descargado
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        # Filtro temporal para Train (aislando >2022 para OOS)
        df = df[df["Datetime"].dt.year <= 2021].copy().reset_index(drop=True)
        return df
    except Exception as e:
        print(f"❌ Error leyendo dataset: {e}")
        return None


def label_hybrid_data(df, lookahead=10):
    """
    1. Etiqueta Direccion (1=BUY, 2=SELL, 0=HOLD) con stop fijo de 1.5
    2. Calcula el MFE (Maximum Favorable Excursion) ANTES de que el precio golpee el SL.
       Expresado en múltiplos de ATR.
    """
    df = df.copy()

    # Fast vectorized approach approximation para el MFE:
    # Como el loop de 80,000 iteraciones sería hiper lento, usamos una heurística vectorizada O(n)
    # Target básico (para el clasificador): Ya estaba en el csv, pero recalculamos

    # Creamos targets iterando rápida con numpy
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    atrs = df["ATR"].values

    n = len(df)
    targets = np.zeros(n, dtype=int)
    mfe_buy = np.zeros(n, dtype=np.float32)
    mfe_sell = np.zeros(n, dtype=np.float32)

    for i in range(n - lookahead):
        entry = closes[i]
        atr = atrs[i] if not np.isnan(atrs[i]) and atrs[i] > 0 else entry * 0.005

        sl_dist = atr * 1.5
        tp_dist = atr * 1.5

        hit_buy_tp, hit_buy_sl = False, False
        hit_sell_tp, hit_sell_sl = False, False

        max_high = entry
        min_low = entry

        for j in range(i + 1, i + 1 + lookahead):
            h, l = highs[j], lows[j]

            # --- BUY EVAL ---
            if not hit_buy_tp and not hit_buy_sl:
                if l <= entry - sl_dist:
                    hit_buy_sl = True
                elif h >= entry + tp_dist:
                    hit_buy_tp = True

            if not hit_buy_sl:  # Si todavía no me estopeó, sigo acumulando MFE
                if h > max_high:
                    max_high = h

            # --- SELL EVAL ---
            if not hit_sell_tp and not hit_sell_sl:
                if h >= entry + sl_dist:
                    hit_sell_sl = True
                elif l <= entry - tp_dist:
                    hit_sell_tp = True

            if not hit_sell_sl:
                if l < min_low:
                    min_low = l

        if hit_buy_tp and not hit_buy_sl:
            targets[i] = 1
        elif hit_sell_tp and not hit_sell_sl:
            targets[i] = 2

        # Guardamos el MFE ratio
        mfe_buy[i] = (max_high - entry) / atr
        mfe_sell[i] = (entry - min_low) / atr

    df["target"] = targets
    df["mfe_buy"] = mfe_buy
    df["mfe_sell"] = mfe_sell
    return df


def main():
    print("=" * 60)
    print("  🚀 ENTRENAMIENTO HÍBRIDO v16 (Clasificador + Regresor)")
    print("=" * 60)

    df = load_and_prepare_data()
    if df is None:
        return

    print("  🏷️ Generando etiquetas híbridas (MFE y TP Dinámico)...")
    # Para ser breves, cogemos 500k de EURUSD u otro
    # Solo tomamos los top pares de forex y stocks pre-descargados por velocidad de prueba
    valid_assets = ["EURUSD=X", "GBPUSD=X", "MSFT", "AUDUSD=X"]
    df = df[df["Asset"].isin(valid_assets)].copy()

    df_hybrid = label_hybrid_data(df, lookahead=10)

    EXCLUDE_COLS = [
        "Datetime",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "target",
        "potential_win",
        "potential_loss",
        "Asset",
        "mfe_buy",
        "mfe_sell",
        "fractal_high",
        "fractal_low",
    ]
    feature_cols = [c for c in df_hybrid.columns if c not in EXCLUDE_COLS]

    # --- 1. Entrenamiento CLASIFICADOR ---
    print("\n  ⚙️ Entrenamiento: 1/3 Clasificador Direccional GPU...")
    X_class = df_hybrid[feature_cols].values
    y_class = df_hybrid["target"].values

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        class_weight="balanced",
        random_state=42,
        device="gpu",
    )
    clf_calibrated = CalibratedClassifierCV(clf, method="isotonic", cv=3)
    clf_calibrated.fit(X_class, y_class)

    # --- 2. Entrenamiento REGRESOR BUY ---
    print("  ⚙️ Entrenamiento: 2/3 Regresor BUY (Predictor de TP)...")
    # Filtramos SOLO por trades ganadores BUY para enseñarle al modelo cuánto suben
    df_buy = df_hybrid[df_hybrid["target"] == 1].copy()
    X_buy = df_buy[feature_cols].values
    y_buy = df_buy["mfe_buy"].values

    # Limitamos el TP logico outlier cap a 5 ATR
    y_buy = np.clip(y_buy, 1.5, 5.0)

    reg_buy = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        device="gpu",
        objective="regression_l1",
    )
    reg_buy.fit(X_buy, y_buy)

    # --- 3. Entrenamiento REGRESOR SELL ---
    print("  ⚙️ Entrenamiento: 3/3 Regresor SELL (Predictor de TP)...")
    df_sell = df_hybrid[df_hybrid["target"] == 2].copy()
    X_sell = df_sell[feature_cols].values
    y_sell = df_sell["mfe_sell"].values

    y_sell = np.clip(y_sell, 1.5, 5.0)

    reg_sell = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        device="gpu",
        objective="regression_l1",
    )
    reg_sell.fit(X_sell, y_sell)

    # --- Guardar ENSAMBLE HÍBRIDO ---
    package = {
        "classifier": clf_calibrated,
        "regressor_buy": reg_buy,
        "regressor_sell": reg_sell,
        "feature_columns": feature_cols,
        "threshold": 0.47,
    }
    joblib.dump(package, "model_hybrid_v16.joblib")

    print(f"\n✅ Pipeline Híbrido GPU guardado -> model_hybrid_v16.joblib")

    # Mini Test
    print("\n  🔍 Evaluando predicciones TP del Regresor (Out of box):")
    sample_buy = X_buy[:5]
    sample_sell = X_sell[:5]
    pred_tps_buy = reg_buy.predict(sample_buy)
    pred_tps_sell = reg_sell.predict(sample_sell)

    print(
        f"      Predicciones de Extensión BUY:  {[f'{x:.2f} ATR' for x in pred_tps_buy]}"
    )
    print(
        f"      Predicciones de Extensión SELL: {[f'{x:.2f} ATR' for x in pred_tps_sell]}"
    )

    print("\n🎉 Fase 7 Exitosa!")


if __name__ == "__main__":
    main()
