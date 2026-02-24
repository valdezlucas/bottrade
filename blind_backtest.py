"""
Blind Backtest — Validación Técnica de Ingeniería
==================================================
Evalúa el modelo en un periodo 100% "ciego" (OOS): 2024-2026.
Calcula métricas clave: Profit Factor, Max Drawdown, Sharpe Ratio, y Esperanza Matemática.

Uso:
    python blind_backtest.py
"""

import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from costs import TradingCosts

# Importar lógica interna
from features import create_features
from fractals import detect_fractals
from ml_dataset import label_data

# Configuración de pares y pips (similar a multi_pair_oos.py)
from multi_pair_oos import PAIRS


def calculate_metrics(trades):
    """Calcula métricas de ingeniería sobre una lista de resultados de trades."""
    if not trades:
        return None

    df = pd.DataFrame(trades)

    # Básicos
    total = len(df)
    wins = df[df["result"] == "TP"]
    losses = df[df["result"] == "SL"]

    win_rate = (len(wins) / total) * 100 if total > 0 else 0

    # Profit Factor
    gross_profit = wins["pips"].sum() if not wins.empty else 0
    gross_loss = abs(losses["pips"].sum()) if not losses.empty else 0
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (gross_profit if gross_profit > 0 else 0)
    )

    # Esperanza Matemática (en pips)
    avg_win = wins["pips"].mean() if not wins.empty else 0
    avg_loss = abs(losses["pips"].mean()) if not losses.empty else 0
    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

    # Drawdown (sobre la curva de equity en pips acumulados)
    df["cum_pips"] = df["pips"].cumsum()
    rolling_max = df["cum_pips"].cummax()
    drawdown = rolling_max - df["cum_pips"]
    max_drawdown = drawdown.max()

    # Sharpe Ratio Simplificado (anualizado asumiendo ~250 trades/año o escala diaria)
    # Usamos retornos de pips por simplicidad técnica
    if total > 5:
        returns = df["pips"]
        std = returns.std()
        sharpe = (returns.mean() / std) * np.sqrt(252) if std > 0 else 0
    else:
        sharpe = 0

    return {
        "total_trades": int(total),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "max_drawdown_pips": float(max_drawdown),
        "sharpe_ratio": float(sharpe),
        "net_pips": float(df["pips"].sum()),
    }


def run_blind_test(split_year=2024):
    """Ejecuta el test ciego en todos los pares."""
    print("=" * 70)
    print(f"  BLIND BACKTEST (OOS {split_year}-2026)")
    print("=" * 70)

    try:
        model_data = joblib.load("model_multi.joblib")
        sell_model_data = joblib.load("model_multi_sell.joblib")
        model = model_data["model"]
        sell_model = sell_model_data["model"]
        feature_cols = model_data["feature_columns"]
        threshold = model_data.get("threshold", 0.6)
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return

    all_results = []

    for pair, config in PAIRS.items():
        if config.get("skip_backtest") and pair != "XAUUSD":
            continue

        path = config["file"]
        if not os.path.exists(path):
            continue

        # Cargar y filtrar OOS (2024+)
        df_raw = pd.read_csv(path)
        # Re-estimar fechas para el split (mismo sistema que multi_pair_oos)
        total_rows = len(df_raw)
        start_yr = 2010 if "2010" in path else (2016 if "2016" in path else 2010)
        dates = pd.bdate_range(start=f"{start_yr}-01-04", periods=total_rows, freq="B")
        df_raw["_date"] = dates[:total_rows]

        df_oos = (
            df_raw[df_raw["_date"] >= pd.Timestamp(f"{split_year}-01-01")]
            .copy()
            .reset_index(drop=True)
        )

        if len(df_oos) < 50:
            continue

        # Preparar features
        df = create_features(df_oos)
        df = detect_fractals(df)
        df = label_data(df, lookahead=20, rr=1.5)
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        if df.empty:
            continue

        # Predicciones
        X = df[feature_cols].values
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        y_sell_proba = sell_model.predict_proba(X)[:, 1]

        costs = TradingCosts(spread_pips=config["spread"], pip_value=config["pip"])
        pair_trades = []

        for i in range(len(df) - 20):
            pred = y_pred[i]
            prob = y_proba[i].max()

            if pred == 0 or prob < threshold:
                continue

            signal = "BUY" if pred == 1 else "SELL"
            if signal == "SELL" and y_sell_proba[i] < 0.5:
                continue

            # Simular trade
            entry = df.iloc[i]["Close"]
            sl_dist = df.iloc[i].get("potential_loss", entry * 0.01)
            tp_dist = df.iloc[i].get("potential_win", entry * 0.015)

            sl = entry - sl_dist if signal == "BUY" else entry + sl_dist
            tp = entry + tp_dist if signal == "BUY" else entry - tp_dist

            future = df.iloc[i + 1 : i + 21]
            result = "EXPIRED"
            exit_price = future.iloc[-1]["Close"]

            for _, f_row in future.iterrows():
                if signal == "BUY":
                    if f_row["Low"] <= sl:
                        result = "SL"
                        exit_price = sl
                        break
                    if f_row["High"] >= tp:
                        result = "TP"
                        exit_price = tp
                        break
                else:
                    if f_row["High"] >= sl:
                        result = "SL"
                        exit_price = sl
                        break
                    if f_row["Low"] <= tp:
                        result = "TP"
                        exit_price = tp
                        break

            # Calcular pips netos con costos reales (Spread + 1 pip slippage)
            slippage = 1.0  # Margen de seguridad sugerido por realidad
            raw_pips = (
                (exit_price - entry) / config["pip"]
                if signal == "BUY"
                else (entry - exit_price) / config["pip"]
            )
            net_pips = raw_pips - config["spread"] - slippage

            pair_trades.append({"result": result, "pips": net_pips})

        if pair_trades:
            stats = calculate_metrics(pair_trades)
            stats["pair"] = pair
            all_results.append(stats)
            print(
                f"  · {pair:6}: {stats['total_trades']:3} trades | WR: {stats['win_rate']:4.1f}% | PF: {stats['profit_factor']:.2f} | Exp: {stats['expectancy']:.1f} pips"
            )

    # Reporte Global
    if not all_results:
        print("\n  [!] No se generaron trades en el periodo OOS.")
        return

    df_res = pd.DataFrame(all_results)
    global_stats = {
        "total_trades": int(df_res["total_trades"].sum()),
        "avg_win_rate": float(df_res["win_rate"].mean()),
        "avg_profit_factor": float(df_res["profit_factor"].mean()),
        "total_net_pips": float(
            df_res["total_trades"].sum() * df_res["expectancy"].mean()
        ),
        "max_drawdown_avg": float(df_res["max_drawdown_pips"].max()),
        "avg_sharpe": float(df_res["sharpe_ratio"].mean()),
        "mathematical_expectancy": float(df_res["expectancy"].mean()),
        "slippage_added": 1.0,
    }

    print("\n" + "=" * 70)
    print("  REPORTE FINAL DE INGENIERÍA (CON SLIPPAGE +1 PIP)")
    print("=" * 70)
    print(f"  Total Trades:         {global_stats['total_trades']}")
    print(f"  Win Rate Promedio:    {global_stats['avg_win_rate']:.1f}%")
    print(f"  Profit Factor:        {global_stats['avg_profit_factor']:.2f}")
    print(
        f"  Esperanza Matemática: {global_stats['mathematical_expectancy']:.2f} pips/trade"
    )
    print(f"  Sharpe Ratio:         {global_stats['avg_sharpe']:.2f}")
    print(f"  Max Drawdown (Pips):  {global_stats['max_drawdown_avg']:.0f}")
    print("=" * 70)

    # Guardar en Firebase si es posible
    try:
        from firebase_manager import db

        if db:
            global_stats["timestamp"] = datetime.utcnow().isoformat()
            db.collection("reports").document("blind_test_v8").set(global_stats)
            print("  ✅ Reporte guardado en Firebase (reports/blind_test_v8)")
    except:
        pass


if __name__ == "__main__":
    run_blind_test()
