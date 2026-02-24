"""
Test end-to-end del pipeline con datos sintéticos.
Genera un CSV con datos OHLCV simulados y corre el entrenamiento completo.
"""

import os
import sys

import numpy as np
import pandas as pd

# Fijar encoding para Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def generate_synthetic_ohlcv(n_rows=2000, seed=42):
    """Genera datos OHLCV sintéticos con tendencia y ruido."""
    np.random.seed(seed)

    # Simular precio con random walk + tendencia leve
    returns = np.random.normal(0.0001, 0.005, n_rows)
    close = 1.1000 + np.cumsum(returns)

    # Generar OHLC a partir de close
    noise = np.random.uniform(0.001, 0.005, n_rows)
    high = close + noise
    low = close - noise
    open_price = close + np.random.normal(0, 0.002, n_rows)
    volume = np.random.randint(100, 10000, n_rows)

    df = pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  TEST END-TO-END — Pipeline ML Trading")
    print("=" * 60)

    # 1. Generar datos sinteticos
    print("\n[1/4] Generando datos sinteticos (2000 filas)...")
    df = generate_synthetic_ohlcv(2000)
    csv_path = "test_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"      Guardado en: {csv_path}")

    # 2. Preparar dataset
    print("\n[2/4] Preparando dataset (features + fractales + labels)...")
    from train import prepare_dataset

    df_prepared = prepare_dataset(csv_path)
    print(f"      Dataset listo: {len(df_prepared)} filas")

    # 3. Walk-forward validation
    print("\n[3/4] Ejecutando walk-forward validation...")
    from costs import TradingCosts
    from train import walk_forward_train

    costs = TradingCosts(spread_pips=1.5, max_slippage_pips=1.0)
    fold_results = walk_forward_train(df_prepared, n_folds=4, costs=costs)

    # 4. Entrenar modelo final
    print("\n[4/4] Entrenando modelo final...")
    from train import train_final_model

    valid_folds = [r for r in fold_results if r["best_metrics"] is not None]
    if valid_folds:
        threshold = valid_folds[0]["best_threshold"]
    else:
        threshold = 0.5

    train_final_model(df_prepared, model_path="test_model.joblib", threshold=threshold)

    # 5. Verificar prediccion
    print("\n[5/5] Verificando prediccion...")
    from predict import predict as run_predict

    signals, _ = run_predict(csv_path, model_path="test_model.joblib")

    print("\n" + "=" * 60)
    print("  TEST COMPLETADO")
    print("=" * 60)
    print(f"  Folds validos: {len(valid_folds)}/{len(fold_results)}")
    print(f"  Senales generadas: {len(signals)}")
    print(f"  Modelo guardado: test_model.joblib")

    # Limpiar archivos de test
    for f in ["test_data.csv", "test_model.joblib"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Limpiado: {f}")

    print("\n  OK - Pipeline funciona correctamente")
