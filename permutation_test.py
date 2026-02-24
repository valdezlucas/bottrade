"""
Permutation Test (Shuffle Labels)
Para asegurar que el modelo está aprendiendo patrones reales de mercado
y no memorizando ruido o aprovechando un data leakage generalizado.

Si el modelo OOS entrenado con labels mezclados mantiene un rendimiento
positivo (Expectancy/PF > 1.0), hay un problema grave de leakage.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from main import TradingCosts
from train import walk_forward_train

warnings.filterwarnings("ignore")


def run_permutation_test(csv_path="df_1d_joined.csv"):
    df = pd.read_csv(csv_path)

    # 1. Aplicar Shuffle a la columna 'target' para destruir cualquier
    # correlación entre el pasado (features) y el futuro (target)
    print("🔀 Shuffling targets (Permutation Test)...")
    np.random.seed(42)  # Fijo para reproducibilidad
    shuffled_targets = np.random.permutation(df["target"].values)
    df["target"] = shuffled_targets

    # Limpiamos infs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Costos estandar
    costs = TradingCosts(
        spread_pips=1.5,
        max_slippage_pips=1.0,
        swap_per_night=0.0,
    )

    thresholds = [0.55, 0.60, 0.65]  # Umbrales realistas

    print("\n🔬 Ejecutando Walk-Forward con Labels Mezclados (Shuffled)...")
    print(
        "   Se espera que TODAS las metricas colapsen a aleatorio (PF <= 1.0, Expectancy < 0)"
    )

    # Ejecutamos el Walk-Forward completo con el dataset destruido
    fold_results = walk_forward_train(df, n_folds=4, costs=costs, thresholds=thresholds)

    print(f"\n{'='*60}")
    print("  RESULTADOS PERMUTATION TEST (SHUFFLED LABELS)")
    print(f"{'='*60}")

    valid_folds = [r for r in fold_results if r["best_metrics"] is not None]

    if not valid_folds:
        print(
            "  ✅ ÉXITO: El modelo colapsó rápidamente. Ningún fold pudo encontrar un threshold válido."
        )
        print(
            "  ✅ Esto confirma que el modelo original SI aprendía de la serie temporal estructurada."
        )
    else:
        avg_expectancy = sum(
            r["best_metrics"]["expectancy"] for r in valid_folds
        ) / len(valid_folds)
        avg_pf = sum(r["best_metrics"]["profit_factor"] for r in valid_folds) / len(
            valid_folds
        )
        print(f"  Folds validos encontrados: {len(valid_folds)}/4")
        print(f"  Expectancy Medio del modelo falso: {avg_expectancy:.4f}")
        print(f"  Profit Factor Medio del modelo falso: {avg_pf:.4f}")

        if avg_expectancy > 0 and avg_pf > 1.0:
            print("\n  ❌ PELIGRO (FALSIFICACION PENDIENTE):")
            print("  El modelo entrenado con datos falsos SIGUE ganando dinero.")
            print(
                "  ¡Esto sugiere un DATA LEAKAGE fuerte en la calculacion del target o pnl/features!"
            )
        else:
            print(
                "\n  ✅ EXPECTATIVA CONFIRMADA: Las métricas del modelo falso son negativas."
            )
            print(
                "  ✅ El edge del modelo original proviene de estructura real de mercado."
            )


if __name__ == "__main__":
    run_permutation_test()
