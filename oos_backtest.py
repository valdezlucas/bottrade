"""
Out-of-Sample Backtest
======================
Entrena con datos hasta 2022, backtestea en 2023-2026 (datos que el modelo NUNCA vió).

Uso:
    python oos_backtest.py --data NZDUSD_daily_2016_2026.csv --split-year 2023
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from backtest import run_backtest
from costs import TradingCosts
from features import create_features
from fractals import detect_fractals
from ml_dataset import label_data
from train import EXCLUDE_COLS, get_feature_columns, walk_forward_train


def run_oos_backtest(
    data_path,
    split_year=2023,
    spread_pips=1.5,
    rr=1.5,
    lookahead=20,
    risk_per_trade=0.005,
    initial_balance=10000,
    max_drawdown_pct=20.0,
):
    """
    Pipeline completo de Out-of-Sample backtest.

    1. Carga datos con fechas
    2. Split temporal: train (< split_year) / test (>= split_year)
    3. Walk-forward en datos de train
    4. Entrena modelo final con datos de train
    5. Backtestea en datos de test (out-of-sample)
    """
    print("=" * 60)
    print(f"  OUT-OF-SAMPLE BACKTEST")
    print(f"  Train: < {split_year} | Test: >= {split_year}")
    print("=" * 60)

    # --- 1. Cargar datos con fechas ---
    print(f"\n[1/5] Cargando datos...")
    df_raw = pd.read_csv(data_path)

    # Detectar columna de fecha si existe
    date_col = None
    for col in df_raw.columns:
        if "date" in col.lower() or "time" in col.lower() or "datetime" in col.lower():
            date_col = col
            break

    # Si no hay columna de fecha, reconstruimos basándonos en días de trading
    # (~252 días/año para mercados financieros)
    if date_col:
        df_raw["_date"] = pd.to_datetime(df_raw[date_col])
    else:
        # Para datos de Yahoo sin fecha, estimamos por posición
        # Sabemos que empieza en 2016 y son datos diarios
        total_rows = len(df_raw)
        # ~252 días de trading por año
        start_date = pd.Timestamp("2016-01-04")  # Primer día de trading 2016
        dates = pd.bdate_range(start=start_date, periods=total_rows, freq="B")
        df_raw["_date"] = dates[:total_rows]

    print(f"   Datos totales: {len(df_raw)} velas")
    print(
        f"   Rango: {df_raw['_date'].iloc[0].date()} → {df_raw['_date'].iloc[-1].date()}"
    )

    # --- 2. Split temporal ---
    split_date = pd.Timestamp(f"{split_year}-01-01")
    mask_train = df_raw["_date"] < split_date
    mask_test = df_raw["_date"] >= split_date

    df_train_raw = df_raw[mask_train].copy().reset_index(drop=True)
    df_test_raw = df_raw[mask_test].copy().reset_index(drop=True)

    print(f"\n[2/5] Split temporal:")
    print(
        f"   Train: {len(df_train_raw)} velas ({df_train_raw['_date'].iloc[0].date()} → {df_train_raw['_date'].iloc[-1].date()})"
    )
    print(
        f"   Test:  {len(df_test_raw)} velas ({df_test_raw['_date'].iloc[0].date()} → {df_test_raw['_date'].iloc[-1].date()})"
    )

    # --- 3. Preparar datasets ---
    print(f"\n[3/5] Preparando features y labels...")

    # Train
    df_train = df_train_raw.drop(columns=["_date"])
    df_train = create_features(df_train)
    df_train = detect_fractals(df_train)
    df_train = label_data(df_train, lookahead=lookahead, rr=rr)

    feature_cols = get_feature_columns(df_train)
    df_train = df_train.dropna(subset=feature_cols).reset_index(drop=True)

    counts = df_train["target"].value_counts().sort_index()
    labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
    print(f"   Train distribucion:")
    for val, count in counts.items():
        print(f"     {labels.get(val, val)}: {count} ({count/len(df_train)*100:.1f}%)")

    # Test
    df_test = df_test_raw.drop(columns=["_date"])
    df_test = create_features(df_test)
    df_test = detect_fractals(df_test)
    df_test = label_data(df_test, lookahead=lookahead, rr=rr)
    df_test = df_test.dropna(subset=feature_cols).reset_index(drop=True)

    counts_test = df_test["target"].value_counts().sort_index()
    print(f"   Test distribucion:")
    for val, count in counts_test.items():
        print(f"     {labels.get(val, val)}: {count} ({count/len(df_test)*100:.1f}%)")

    # --- 4. Walk-forward en train ---
    print(f"\n[4/5] Walk-forward validation en datos de TRAIN...")
    costs = TradingCosts(spread_pips=spread_pips)
    fold_results = walk_forward_train(df_train, n_folds=4, costs=costs)

    valid_folds = [r for r in fold_results if r["best_metrics"] is not None]
    if valid_folds:
        best_thresholds = [r["best_threshold"] for r in valid_folds]
        optimal_threshold = max(set(best_thresholds), key=best_thresholds.count)
    else:
        optimal_threshold = 0.5

    # Entrenar modelo final con TODOS los datos de train
    print(f"\n   Entrenando modelo final con {len(df_train)} filas de train...")
    X_train = df_train[feature_cols].values
    y_train = df_train["target"].values

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Guardar modelo
    model_path = "model_oos.joblib"
    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "threshold": optimal_threshold,
    }
    joblib.dump(artifact, model_path)
    print(f"   Modelo guardado: {model_path} (threshold: {optimal_threshold})")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print(f"\n   Top 10 features:")
    for feat, imp in importances.head(10).items():
        bar = "#" * int(imp * 100)
        print(f"     {feat:25s} {imp:.4f} {bar}")

    # --- 4b. Entrenar modelo SELL separado ---
    print(f"\n   Entrenando modelo SELL separado...")
    df_train["sell_target"] = (df_train["target"] == 2).astype(int)
    X_sell = df_train[feature_cols].values
    y_sell = df_train["sell_target"].values

    sell_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    sell_model.fit(X_sell, y_sell)

    sell_model_path = "model_oos_sell.joblib"
    sell_artifact = {
        "model": sell_model,
        "feature_columns": feature_cols,
        "threshold": optimal_threshold,
    }
    joblib.dump(sell_artifact, sell_model_path)
    n_sell = y_sell.sum()
    print(f"   Modelo SELL guardado: {sell_model_path} ({n_sell} ejemplos SELL)")

    # --- 5. Backtest OUT-OF-SAMPLE ---
    print(f"\n{'='*60}")
    print(f"  [5/5] BACKTEST OUT-OF-SAMPLE ({split_year}–2026)")
    print(f"  Datos que el modelo NUNCA vio durante entrenamiento")
    print(f"{'='*60}")

    # Guardar test data temporalmente para el backtester
    test_csv = "_oos_test_data.csv"
    df_test.to_csv(test_csv, index=False)

    # Correr backtest con ambos modelos y drawdown breaker
    results = run_backtest(
        data_path=test_csv,
        model_path=model_path,
        sell_model_path=sell_model_path,
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        rr_ratio=rr,
        max_drawdown_pct=max_drawdown_pct,
        spread_pips=spread_pips,
    )

    # Limpiar temp
    import os

    if os.path.exists(test_csv):
        os.remove(test_csv)

    # --- Comparación walk-forward vs OOS ---
    if results["metrics"] and valid_folds:
        oos = results["metrics"]
        avg_wf_exp = np.mean([r["best_metrics"]["expectancy"] for r in valid_folds])
        avg_wf_pf = np.mean([r["best_metrics"]["profit_factor"] for r in valid_folds])

        print(f"\n{'='*60}")
        print(f"  COMPARACION: Walk-Forward vs Out-of-Sample")
        print(f"{'='*60}")
        print(f"{'Metrica':<25} {'Walk-Forward':>15} {'OOS ({split_year}+)':>15}")
        print(f"{'-'*55}")
        print(f"{'Expectancy':<25} {avg_wf_exp:>15.6f} {oos['expectancy']:>15.6f}")
        print(f"{'Profit Factor':<25} {avg_wf_pf:>15.4f} {oos['profit_factor']:>15.4f}")
        print(f"{'Win Rate':<25} {'':>15} {oos['win_rate']*100:>14.1f}%")
        print(f"{'Total Trades':<25} {'':>15} {oos['n_trades']:>15}")
        print(f"{'Max Drawdown':<25} {'':>15} {oos['max_drawdown_pct']:>14.1f}%")
        print(f"{'Sharpe Ratio':<25} {'':>15} {oos['sharpe_ratio']:>15.4f}")

        if oos["expectancy"] > 0:
            print(
                f"\n  [OK] El modelo mantiene expectancy positiva en datos NUNCA vistos"
            )
        else:
            print(
                f"\n  [X]  El modelo pierde expectancy en datos nuevos — posible overfitting"
            )

        if oos["profit_factor"] > 1.0:
            print(f"  [OK] Profit Factor > 1.0 out-of-sample")
        else:
            print(f"  [X]  Profit Factor < 1.0 out-of-sample")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Out-of-Sample Backtest")
    parser.add_argument("--data", required=True, help="CSV con datos OHLC")
    parser.add_argument(
        "--split-year", type=int, default=2023, help="Anio de corte (default: 2023)"
    )
    parser.add_argument("--spread", type=float, default=1.5, help="Spread en pips")
    parser.add_argument("--rr", type=float, default=1.5, help="R:R ratio")
    parser.add_argument("--balance", type=float, default=10000, help="Capital inicial")
    parser.add_argument(
        "--risk", type=float, default=0.005, help="Riesgo por trade (default: 0.5%)"
    )
    parser.add_argument(
        "--max-dd", type=float, default=20.0, help="Max drawdown %% (default: 20)"
    )

    args = parser.parse_args()

    run_oos_backtest(
        data_path=args.data,
        split_year=args.split_year,
        spread_pips=args.spread,
        rr=args.rr,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        max_drawdown_pct=args.max_dd,
    )
