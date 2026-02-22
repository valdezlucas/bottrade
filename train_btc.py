"""
BTC-USD Daily Model Trainer
============================
Downloads BTC-USD daily data from yfinance (max history),
trains GradientBoosting BUY+SELL models specifically for Bitcoin.

Usage:
    python train_btc.py
"""

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.ensemble import GradientBoostingClassifier

from features import create_features
from fractals import detect_fractals
from ml_dataset import label_data


# Features que NO son input del modelo
EXCLUDE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "target", "potential_win", "potential_loss",
    "fractal_high", "fractal_low",
    "Datetime", "Date",
]


def get_feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def download_btc_daily(days=2500):
    """Descarga datos diarios de BTC-USD."""
    print(f"📥 Descargando BTC-USD (diario, {days} dias)...")
    end = datetime.now()
    start = end - timedelta(days=days)

    df = yf.download(
        "BTC-USD",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
    )

    if df.empty:
        print("⚠️  Sin datos")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    for col in ["Datetime", "Date"]:
        if col in df.columns:
            df = df.rename(columns={col: "Datetime"})

    print(f"✅ {len(df)} velas diarias descargadas")
    return df


def train_btc_daily():
    """Entrena modelos BUY+SELL para BTC-USD diario."""
    print("=" * 60)
    print("  ENTRENAMIENTO BTC-USD DAILY")
    print("  Lookahead: 20 | R:R: 1.5")
    print("=" * 60)

    # Descargar datos
    df = download_btc_daily(days=2500)
    if df is None:
        return

    # Preparar features + labels
    df_work = df[["Open", "High", "Low", "Close"]].copy()
    if "Volume" in df.columns:
        df_work["Volume"] = df["Volume"].values

    df_work = create_features(df_work)
    df_work = detect_fractals(df_work)
    df_work = label_data(df_work, lookahead=20, rr=1.5)

    feature_cols = get_feature_columns(df_work)
    df_work = df_work.dropna(subset=feature_cols).reset_index(drop=True)

    print(f"\n📊 DATOS PREPARADOS: {len(df_work)} velas")
    counts = df_work["target"].value_counts().sort_index()
    labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
    for val, count in counts.items():
        print(f"   {labels.get(val, val)}: {count} ({count/len(df_work)*100:.1f}%)")

    X_all = df_work[feature_cols].values
    y_all = df_work["target"].values
    total = len(df_work)

    # ─── Walk-Forward ───────────────────────────────────────────────
    print(f"\n🔄 Walk-Forward Validation (3 folds)...")

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
            min_samples_leaf=10,
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

            if n_trades < 10:
                continue

            wins = losses = 0
            total_win = total_loss = 0.0

            for idx in np.where(mask)[0]:
                actual = y_te[idx]
                pred = y_pred[idx]
                pw = df_work.iloc[train_end + idx]["potential_win"]
                pl = df_work.iloc[train_end + idx]["potential_loss"]

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
    print(f"\n🏋️ Entrenando modelos finales con {total} velas...")

    # Modelo principal (multiclass)
    main_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    main_model.fit(X_all, y_all)

    joblib.dump({
        "model": main_model,
        "feature_columns": feature_cols,
        "threshold": best_th,
    }, "model_btc_daily.joblib")
    print(f"  💾 Modelo BUY guardado: model_btc_daily.joblib")

    # Modelo SELL (binario)
    y_sell = (df_work["target"] == 2).astype(int).values
    sell_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    sell_model.fit(X_all, y_sell)

    joblib.dump({
        "model": sell_model,
        "feature_columns": feature_cols,
        "threshold": best_th,
    }, "model_btc_daily_sell.joblib")
    print(f"  💾 Modelo SELL guardado: model_btc_daily_sell.joblib")

    # Feature importance
    importances = pd.Series(main_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print(f"\n📊 Top 10 features (BTC Daily):")
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 100)
        print(f"   {feat:25s} {imp:.4f} {bar}")

    print(f"\n✅ Modelos BTC-USD Daily entrenados exitosamente!")


if __name__ == "__main__":
    train_btc_daily()
