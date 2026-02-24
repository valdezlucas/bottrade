"""
Backtest Hub — Genera historial de operaciones 'fantasma'
=========================================================
Corre el modelo actual sobre los últimos N meses de datos reales y marca
cada trade con su resultado (TP/SL), guardando todo en Firebase.

Los usuarios pueden ver estos resultados con /history en Telegram.

Uso:
    python backtest_hub.py --months 6
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from features import create_features
from fractals import detect_fractals
from ml_dataset import label_data
from train import get_feature_columns

try:
    from firebase_manager import db

    if db:
        from google.cloud.firestore_v1.base_query import FieldFilter

        HAS_FIREBASE = True
    else:
        HAS_FIREBASE = False
except:
    HAS_FIREBASE = False

# Pares que usamos en backtesting (solo los pip=0.0001 para simplificar)
BACKTEST_PAIRS = {
    "EURUSD": {"ticker": "EURUSD=X", "spread": 1.0, "pip": 0.0001},
    "GBPUSD": {"ticker": "GBPUSD=X", "spread": 1.2, "pip": 0.0001},
    "AUDUSD": {"ticker": "AUDUSD=X", "spread": 1.2, "pip": 0.0001},
    "NZDUSD": {"ticker": "NZDUSD=X", "spread": 1.5, "pip": 0.0001},
    "USDCAD": {"ticker": "USDCAD=X", "spread": 1.5, "pip": 0.0001},
    "USDCHF": {"ticker": "USDCHF=X", "spread": 1.5, "pip": 0.0001},
    "EURGBP": {"ticker": "EURGBP=X", "spread": 1.5, "pip": 0.0001},
}


def simulate_trade(row, signal, rr=1.5, lookahead=20, df_future=None):
    """Simula un trade hacia adelante en los datos reales."""
    entry = row["Close"]

    if signal == "BUY":
        sl = entry - row.get("potential_loss", entry * 0.01)
        tp = entry + row.get("potential_win", entry * 0.015)
    else:  # SELL
        sl = entry + row.get("potential_loss", entry * 0.01)
        tp = entry - row.get("potential_win", entry * 0.015)

    if df_future is None or len(df_future) == 0:
        return {"result": "PENDING", "exit_price": entry}

    for _, future_row in df_future.iterrows():
        if signal == "BUY":
            if future_row["Low"] <= sl:
                return {"result": "SL", "exit_price": sl}
            if future_row["High"] >= tp:
                return {"result": "TP", "exit_price": tp}
        else:
            if future_row["High"] >= sl:
                return {"result": "SL", "exit_price": sl}
            if future_row["Low"] <= tp:
                return {"result": "TP", "exit_price": tp}

    # Si no toca ni SL ni TP, cerramos al precio actual
    return {"result": "EXPIRED", "exit_price": df_future.iloc[-1]["Close"]}


def run_backtest_hub(months=6):
    """Corre el modelo sobre los últimos N meses y genera historial."""
    print("=" * 70)
    print(f"  BACKTEST HUB — Últimos {months} meses")
    print("=" * 70)

    # Cargar modelos
    model_data = joblib.load("model_multi.joblib")
    model = model_data["model"]
    feature_cols = model_data["feature_columns"]
    threshold = model_data.get("threshold", 0.5)

    sell_model_data = joblib.load("model_multi_sell.joblib")
    sell_model = sell_model_data["model"]

    print(f"  Modelo cargado | Threshold: {threshold}")
    print(f"  Features: {len(feature_cols)}")

    all_trades = []

    for pair, config in BACKTEST_PAIRS.items():
        print(f"\n{'─'*70}")
        print(f"  {pair} — Descargando últimos {months} meses...")

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30)

            df = yf.download(
                config["ticker"],
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
            )

            if df.empty:
                print(f"  [!] Sin datos para {pair}")
                continue

            # Limpiar
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            rename = {}
            for c in df.columns:
                cl = c.lower()
                if "open" in cl:
                    rename[c] = "Open"
                elif "high" in cl:
                    rename[c] = "High"
                elif "low" in cl:
                    rename[c] = "Low"
                elif "close" in cl:
                    rename[c] = "Close"
                elif "volume" in cl:
                    rename[c] = "Volume"
            df = df.rename(columns=rename)
            df = df.reset_index(drop=True)

            # Features
            df = create_features(df)
            df = detect_fractals(df)
            df = label_data(df, lookahead=20, rr=1.5)
            df = df.dropna(subset=feature_cols).reset_index(drop=True)

            if len(df) < 50:
                print(f"  [!] Datos insuficientes ({len(df)})")
                continue

            # Predecir
            X = df[feature_cols].values
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)

            # Predicciones SELL del modelo dedicado
            y_sell_proba = sell_model.predict_proba(X)[:, 1]

            n_signals = 0
            pair_trades = []

            for i in range(len(df) - 20):  # Dejar margen para simular
                pred = y_pred[i]
                prob = y_proba[i].max()

                if pred == 0 or prob < threshold:
                    continue

                signal = "BUY" if pred == 1 else "SELL"

                # Para SELL, verificar modelo dedicado
                if signal == "SELL" and y_sell_proba[i] < 0.5:
                    continue

                row = df.iloc[i]
                future = df.iloc[i + 1 : i + 21]

                result = simulate_trade(row, signal, df_future=future)

                trade = {
                    "pair": pair,
                    "signal": signal,
                    "entry": float(row["Close"]),
                    "sl": (
                        float(
                            row["Close"]
                            - row.get("potential_loss", row["Close"] * 0.01)
                        )
                        if signal == "BUY"
                        else float(
                            row["Close"]
                            + row.get("potential_loss", row["Close"] * 0.01)
                        )
                    ),
                    "tp": (
                        float(
                            row["Close"]
                            + row.get("potential_win", row["Close"] * 0.015)
                        )
                        if signal == "BUY"
                        else float(
                            row["Close"]
                            - row.get("potential_win", row["Close"] * 0.015)
                        )
                    ),
                    "confidence": float(prob),
                    "result": result["result"],
                    "exit_price": float(result["exit_price"]),
                    "candle_index": i,
                    "source": "backtest_hub",
                }

                pair_trades.append(trade)
                n_signals += 1

            # Stats
            wins = sum(1 for t in pair_trades if t["result"] == "TP")
            losses = sum(1 for t in pair_trades if t["result"] == "SL")
            expired = sum(1 for t in pair_trades if t["result"] == "EXPIRED")
            wr = wins / max(1, wins + losses) * 100

            emoji = "✅" if wr > 50 else "❌"
            print(
                f"  {emoji} {pair}: {n_signals} trades | {wins}W {losses}L {expired}E | WR: {wr:.1f}%"
            )

            all_trades.extend(pair_trades)

        except Exception as e:
            print(f"  [!] Error en {pair}: {e}")
            continue

    # Resumen
    print(f"\n{'='*70}")
    print(f"  RESUMEN BACKTEST HUB")
    print(f"{'='*70}")

    total = len(all_trades)
    wins = sum(1 for t in all_trades if t["result"] == "TP")
    losses = sum(1 for t in all_trades if t["result"] == "SL")
    expired = sum(1 for t in all_trades if t["result"] == "EXPIRED")
    wr = wins / max(1, wins + losses) * 100

    print(f"  Total trades:  {total}")
    print(f"  Ganados (TP):  {wins}")
    print(f"  Perdidos (SL): {losses}")
    print(f"  Expirados:     {expired}")
    print(f"  Win Rate:      {wr:.1f}%")

    # Subir a Firebase
    if HAS_FIREBASE:
        print(f"\n  Subiendo {len(all_trades)} trades a Firebase...")

        # Limpiar historial viejo
        old_docs = db.collection("backtest_history").stream()
        for doc in old_docs:
            doc.reference.delete()

        # Subir nuevos (últimos 50 como máximo)
        recent = sorted(all_trades, key=lambda x: x["candle_index"], reverse=True)[:50]
        for i, trade in enumerate(recent):
            trade["uploaded_at"] = datetime.utcnow().isoformat()
            trade["order"] = i
            db.collection("backtest_history").document(f"trade_{i:03d}").set(trade)

        print(f"  ✅ {len(recent)} trades subidos a Firebase (backtest_history)")
    else:
        print(f"\n  ⚠️ Firebase no disponible. Guardando localmente...")
        pd.DataFrame(all_trades).to_csv("backtest_history.csv", index=False)
        print(f"  Guardado en: backtest_history.csv")

    return all_trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Hub")
    parser.add_argument("--months", type=int, default=6, help="Meses de historial")
    args = parser.parse_args()
    run_backtest_hub(months=args.months)
