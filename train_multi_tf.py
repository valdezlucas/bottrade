"""
Multi-Timeframe Model Trainer
==============================
Downloads 1H data from yfinance, resamples to 4H,
and trains separate BUY+SELL models for each timeframe.

Usage:
    python train_multi_tf.py               # Train both 1H and 4H
    python train_multi_tf.py --tf 1h       # Train only 1H
    python train_multi_tf.py --tf 4h       # Train only 4H
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.ensemble import GradientBoostingClassifier

from features import create_features
from fractals import detect_fractals
from ml_dataset import label_data


# ─── Pares a entrenar ──────────────────────────────────────────────────
PAIRS = {
    "EURUSD": {"ticker": "EURUSD=X", "spread": 1.0, "pip": 0.0001},
    "GBPUSD": {"ticker": "GBPUSD=X", "spread": 1.2, "pip": 0.0001},
    "NZDUSD": {"ticker": "NZDUSD=X", "spread": 1.5, "pip": 0.0001},
    "AUDUSD": {"ticker": "AUDUSD=X", "spread": 1.2, "pip": 0.0001},
    "USDCAD": {"ticker": "USDCAD=X", "spread": 1.5, "pip": 0.0001},
    "USDCHF": {"ticker": "USDCHF=X", "spread": 1.5, "pip": 0.0001},
    "EURGBP": {"ticker": "EURGBP=X", "spread": 1.5, "pip": 0.0001},
    "USDJPY": {"ticker": "USDJPY=X", "spread": 1.0, "pip": 0.01},
    "EURJPY": {"ticker": "EURJPY=X", "spread": 1.5, "pip": 0.01},
    "GBPJPY": {"ticker": "GBPJPY=X", "spread": 2.0, "pip": 0.01},
    "AUDJPY": {"ticker": "AUDJPY=X", "spread": 2.0, "pip": 0.01},
    "NZDJPY": {"ticker": "NZDJPY=X", "spread": 2.5, "pip": 0.01},
    "EURAUD": {"ticker": "EURAUD=X", "spread": 2.0, "pip": 0.0001},
    "GBPAUD": {"ticker": "GBPAUD=X", "spread": 2.5, "pip": 0.0001},
}

# Features que NO son input del modelo
EXCLUDE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "target", "potential_win", "potential_loss",
    "fractal_high", "fractal_low",
    "Datetime",
]


def get_feature_columns(df):
    """Obtiene columnas de features válidas."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ─── Descarga de datos ─────────────────────────────────────────────────
def download_1h_data(ticker, days=720):
    """
    Descarga datos de 1H desde yfinance.
    yfinance permite hasta ~730 días de datos intraday.
    """
    print(f"  📥 Descargando {ticker} (1H, {days} días)...")
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
            print(f"  ⚠️  Sin datos para {ticker}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Guardar columna de datetime si existe
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Datetime"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "Datetime"})

        print(f"  ✅ {len(df)} velas de 1H descargadas")
        return df

    except Exception as e:
        print(f"  ❌ Error descargando {ticker}: {e}")
        return None


def resample_to_4h(df_1h):
    """
    Resamplea datos de 1H a 4H usando OHLC correcto.
    """
    if df_1h is None or df_1h.empty:
        return None

    df = df_1h.copy()

    # Asegurar que Datetime es datetime type
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")

    # Resample a 4H
    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }

    # Solo incluir Volume si existe
    if "Volume" in df.columns:
        ohlc["Volume"] = "sum"

    df_4h = df.resample("4h").agg(ohlc).dropna()
    df_4h = df_4h.reset_index()

    print(f"  📊 Resampled a {len(df_4h)} velas de 4H")
    return df_4h


# ─── Pipeline de preparación ───────────────────────────────────────────
def prepare_data(df, lookahead, rr):
    """Aplica features, fractales y labeling a un DataFrame OHLC."""
    if df is None or len(df) < 60:
        return None

    # Guardar Datetime para no perderla
    has_dt = "Datetime" in df.columns
    dt_col = df["Datetime"].values if has_dt else None

    # Crear features (necesita Open, High, Low, Close)
    df_work = df[["Open", "High", "Low", "Close"]].copy()
    if "Volume" in df.columns:
        df_work["Volume"] = df["Volume"].values

    df_work = create_features(df_work)
    df_work = detect_fractals(df_work)
    df_work = label_data(df_work, lookahead=lookahead, rr=rr)

    # Eliminar NaN en features
    feature_cols = get_feature_columns(df_work)
    df_work = df_work.dropna(subset=feature_cols).reset_index(drop=True)

    return df_work


# ─── Entrenamiento por timeframe ───────────────────────────────────────
def train_timeframe(tf_name, lookahead, rr):
    """
    Entrena modelos BUY+SELL para un timeframe específico.

    Args:
        tf_name: "1h" o "4h"
        lookahead: barras a mirar hacia adelante para labeling
        rr: ratio reward/risk para labeling
    """
    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO {tf_name.upper()}")
    print(f"  Lookahead: {lookahead} | R:R: {rr}")
    print(f"  Pares: {len(PAIRS)}")
    print(f"{'='*60}")

    all_data = []
    feature_cols = None

    for pair, config in PAIRS.items():
        print(f"\n📈 Procesando {pair}...")

        # Descargar 1H
        df_1h = download_1h_data(config["ticker"])

        if df_1h is None:
            continue

        # Para 4H, resamplear
        if tf_name == "4h":
            df = resample_to_4h(df_1h)
        else:
            df = df_1h.copy()

        if df is None:
            continue

        # Preparar (features + labels)
        df_prepared = prepare_data(df, lookahead=lookahead, rr=rr)

        if df_prepared is None or len(df_prepared) < 100:
            print(f"  ⚠️  {pair}: datos insuficientes ({len(df_prepared) if df_prepared is not None else 0} filas)")
            continue

        if feature_cols is None:
            feature_cols = get_feature_columns(df_prepared)

        all_data.append(df_prepared)

        counts = df_prepared["target"].value_counts().sort_index()
        labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
        dist = " | ".join([f"{labels.get(k, k)}: {v}" for k, v in counts.items()])
        print(f"  ✅ {pair}: {len(df_prepared)} filas | {dist}")

    if not all_data:
        print("❌ No se pudo generar datos para ningún par")
        return

    # Combinar todos los pares
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 DATOS COMBINADOS: {len(combined)} filas")

    counts_all = combined["target"].value_counts().sort_index()
    labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
    for val, count in counts_all.items():
        print(f"   {labels.get(val, val)}: {count} ({count/len(combined)*100:.1f}%)")

    # ─── Walk-Forward para encontrar threshold óptimo ──────────────
    print(f"\n🔄 Walk-Forward Validation...")

    X_all = combined[feature_cols].values
    y_all = combined["target"].values
    total = len(combined)

    best_th = 0.5
    best_exp = -999

    for fold in range(3):
        train_end = int(total * (0.5 + fold * 0.15))
        test_end = int(total * (0.65 + fold * 0.15))
        test_end = min(test_end, total)

        X_tr = X_all[:train_end]
        y_tr = y_all[:train_end]
        X_te = X_all[train_end:test_end]
        y_te = y_all[train_end:test_end]

        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        for th in [0.40, 0.45, 0.50, 0.55, 0.60]:
            max_probs = y_proba.max(axis=1)
            mask = (y_pred != 0) & (max_probs >= th)
            n_trades = mask.sum()

            if n_trades < 20:
                continue

            wins = losses = 0
            total_win = total_loss = 0.0

            for idx in np.where(mask)[0]:
                actual = y_te[idx]
                pred = y_pred[idx]
                pw = combined.iloc[train_end + idx]["potential_win"]
                pl = combined.iloc[train_end + idx]["potential_loss"]

                if pred == actual and actual != 0:
                    wins += 1
                    total_win += pw
                else:
                    losses += 1
                    total_loss += pl

            wr = wins / n_trades if n_trades > 0 else 0
            avg_w = total_win / wins if wins > 0 else 0
            avg_l = total_loss / losses if losses > 0 else 0
            exp = (wr * avg_w) - ((1 - wr) * avg_l)

            if exp > best_exp:
                best_exp = exp
                best_th = th

        print(f"  Fold {fold+1}: Train {train_end} | Test {test_end-train_end} → Exp: {best_exp:.6f}")

    print(f"\n🎯 Threshold óptimo: {best_th}")

    # ─── Entrenar modelos finales ──────────────────────────────────
    print(f"\n🏋️ Entrenando modelos finales con {len(combined)} filas...")

    # Modelo principal (multiclass: HOLD=0, BUY=1, SELL=2)
    main_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    main_model.fit(X_all, y_all)

    main_path = f"model_{tf_name}.joblib"
    joblib.dump({
        "model": main_model,
        "feature_columns": feature_cols,
        "threshold": best_th,
    }, main_path)
    print(f"  💾 Modelo BUY guardado: {main_path}")

    # Modelo SELL (binario)
    y_sell = (combined["target"] == 2).astype(int).values
    sell_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    sell_model.fit(X_all, y_sell)

    sell_path = f"model_{tf_name}_sell.joblib"
    joblib.dump({
        "model": sell_model,
        "feature_columns": feature_cols,
        "threshold": best_th,
    }, sell_path)
    print(f"  💾 Modelo SELL guardado: {sell_path}")

    # Feature importance
    importances = pd.Series(main_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print(f"\n📊 Top 10 features ({tf_name.upper()}):")
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 100)
        print(f"   {feat:25s} {imp:.4f} {bar}")

    print(f"\n✅ Modelos {tf_name.upper()} entrenados exitosamente!")


# ─── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Timeframe Model Trainer")
    parser.add_argument("--tf", type=str, default="all",
                        choices=["1h", "4h", "all"],
                        help="Timeframe a entrenar (default: all)")
    parser.add_argument("--lookahead-1h", type=int, default=12,
                        help="Lookahead para 1H (default: 12 barras = 12h)")
    parser.add_argument("--lookahead-4h", type=int, default=10,
                        help="Lookahead para 4H (default: 10 barras = 40h)")
    parser.add_argument("--rr", type=float, default=1.5,
                        help="Ratio Reward/Risk (default: 1.5)")
    args = parser.parse_args()

    # Lookahead ajustado por timeframe:
    # 1H: 12 barras = mira 12 horas adelante
    # 4H: 10 barras = mira 40 horas adelante
    # Daily (existente): 20 barras = mira 20 días adelante

    if args.tf in ["1h", "all"]:
        train_timeframe("1h", lookahead=args.lookahead_1h, rr=args.rr)

    if args.tf in ["4h", "all"]:
        train_timeframe("4h", lookahead=args.lookahead_4h, rr=args.rr)

    print("\n🎉 Entrenamiento multi-timeframe completado!")
