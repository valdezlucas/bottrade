import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from data import load_data
from fractals import detect_fractals
from ml_dataset import label_data
from costs import TradingCosts
from evaluation import evaluate_signals, optimize_threshold, print_fold_report


# Features que usa el modelo (excluye target, potential_win/loss, y columnas OHLCV)
EXCLUDE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "target", "potential_win", "potential_loss",
    "fractal_high", "fractal_low",
]


def get_feature_columns(df):
    """Obtiene las columnas de features válidas para el modelo."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def prepare_dataset(path, lookahead=20, rr=1.5):
    """
    Carga datos, aplica features, fractales y labeling.
    Devuelve DataFrame limpio listo para entrenar.
    """
    print(f"📂 Cargando datos de: {path}")
    df = load_data(path)

    print("🔎 Detectando fractales...")
    df = detect_fractals(df)

    print(f"🏷️  Etiquetando datos (lookahead={lookahead}, R:R={rr})...")
    df = label_data(df, lookahead=lookahead, rr=rr)

    # Eliminar filas con NaN en features
    feature_cols = get_feature_columns(df)
    initial_len = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = initial_len - len(df)
    print(f"🧹 Eliminadas {dropped} filas con NaN ({len(df)} filas restantes)")

    # Estadísticas de labels
    counts = df["target"].value_counts().sort_index()
    labels_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    print("\n📊 Distribución de clases:")
    for val, count in counts.items():
        pct = count / len(df) * 100
        print(f"   {labels_map.get(val, val)}: {count} ({pct:.1f}%)")

    return df


def walk_forward_train(df, n_folds=4, costs=None, thresholds=None):
    """
    Entrenamiento Walk-Forward con evaluación financiera completa.

    Train siempre crece, test es el siguiente bloque temporal.
    """
    if costs is None:
        costs = TradingCosts()

    feature_cols = get_feature_columns(df)
    total = len(df)
    # Primer fold empieza en 60%, cada fold avanza 10%
    fold_size = total // 10

    print(f"\n{'#'*60}")
    print(f"  WALK-FORWARD VALIDATION — {n_folds} Folds")
    print(f"  Total de datos: {total} filas")
    print(f"{'#'*60}")

    all_fold_results = []

    for fold in range(n_folds):
        train_end = int(total * (0.6 + fold * 0.1))
        test_end = int(total * (0.7 + fold * 0.1))
        test_end = min(test_end, total)

        df_train = df.iloc[:train_end]
        df_test = df.iloc[train_end:test_end]

        print(f"\n--- Fold {fold+1}: Train [0:{train_end}] → Test [{train_end}:{test_end}] ---")
        print(f"    Train: {len(df_train)} filas | Test: {len(df_test)} filas")

        X_train = df_train[feature_cols].values
        y_train = df_train["target"].values
        X_test = df_test[feature_cols].values

        # Entrenar modelo con class_weight balanceado
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Predicciones y probabilidades
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Classification report (informativo)
        print(f"\n    Classification Report:")
        target_names = ["HOLD", "BUY", "SELL"]
        present_classes = sorted(set(y_train) | set(y_pred))
        names = [target_names[i] for i in present_classes if i < len(target_names)]
        print(classification_report(
            df_test["target"].values, y_pred,
            labels=present_classes,
            target_names=names,
            zero_division=0,
        ))

        # Optimizar threshold
        best_th, best_metrics, th_results = optimize_threshold(
            df_test, y_proba, model, thresholds=thresholds, costs=costs
        )

        # Imprimir resultados por threshold
        print(f"    Threshold optimization:")
        for r in th_results:
            th = r["threshold"]
            m = r["metrics"]
            if m is None:
                print(f"      th={th:.2f} → < 30 trades (inválido)")
            else:
                print(f"      th={th:.2f} → {m['n_trades']} trades | "
                      f"Exp: {m['expectancy']:.6f} | PF: {m['profit_factor']:.4f}")

        # Reporte del mejor threshold
        print_fold_report(fold + 1, best_metrics, best_th)

        all_fold_results.append({
            "fold": fold + 1,
            "train_size": len(df_train),
            "test_size": len(df_test),
            "best_threshold": best_th,
            "best_metrics": best_metrics,
        })

    return all_fold_results


def train_final_model(df, model_path="model.joblib", threshold=0.6):
    """
    Entrena el modelo final con todos los datos y lo guarda.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["target"].values

    print(f"\n🏋️ Entrenando modelo final con {len(df)} filas...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Guardar modelo + metadata
    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "threshold": threshold,
    }

    joblib.dump(artifact, model_path)
    print(f"💾 Modelo guardado en: {model_path}")
    print(f"   Threshold óptimo: {threshold}")
    print(f"   Features: {len(feature_cols)}")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print(f"\n📊 Top 10 features más importantes:")
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 100)
        print(f"   {feat:25s} {imp:.4f} {bar}")

    return model
