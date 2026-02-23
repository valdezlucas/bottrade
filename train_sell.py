"""
Entrena un modelo dedicado para señales SELL.

El modelo principal tiende a sesgar BUY. Este modelo binario se enfoca 
exclusivamente en detectar oportunidades de venta (short).

Target binario: 0 = NO_SELL, 1 = SELL
"""
import sys
import numpy as np
import pandas as pd
import joblib
import argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

from features import create_features
from fractals import detect_fractals
from ml_dataset import label_data
from train import get_feature_columns
from costs import TradingCosts
from evaluation import evaluate_signals, print_fold_report


def prepare_sell_dataset(path, lookahead=20, rr=1.5):
    """Prepara dataset con target binario para SELL."""
    print(f"Cargando datos: {path}")
    df = pd.read_csv(path)
    df = create_features(df)
    df = detect_fractals(df)
    df = label_data(df, lookahead=lookahead, rr=rr)

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Convertir a target binario: SELL (2) -> 1, todo lo demás -> 0
    df["sell_target"] = (df["target"] == 2).astype(int)

    n_sell = df["sell_target"].sum()
    n_total = len(df)
    print(f"Datos: {n_total} filas | SELL: {n_sell} ({n_sell/n_total*100:.1f}%) | NO_SELL: {n_total-n_sell} ({(n_total-n_sell)/n_total*100:.1f}%)")

    return df, feature_cols


def walk_forward_sell(df, feature_cols, n_folds=4, costs=None):
    """Walk-forward para el modelo SELL binario."""
    if costs is None:
        costs = TradingCosts()

    total = len(df)

    print(f"\n{'#'*60}")
    print(f"  WALK-FORWARD — Modelo SELL")
    print(f"{'#'*60}")

    all_results = []

    for fold in range(n_folds):
        train_end = int(total * (0.6 + fold * 0.1))
        test_end = int(total * (0.7 + fold * 0.1))
        test_end = min(test_end, total)

        df_train = df.iloc[:train_end]
        df_test = df.iloc[train_end:test_end]

        print(f"\n--- Fold {fold+1}: Train [0:{train_end}] -> Test [{train_end}:{test_end}] ---")

        X_train = df_train[feature_cols].values
        y_train = df_train["sell_target"].values
        X_test = df_test[feature_cols].values
        y_test = df_test["sell_target"].values

        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        print(classification_report(y_test, y_pred, target_names=["NO_SELL", "SELL"], zero_division=0))

        # Evaluar señales SELL: cuando predice SELL (1), el trade es short
        # Usar potential_win/loss del target original (SELL = target 2)
        n_sell_preds = (y_pred == 1).sum()
        print(f"    Predicciones SELL: {n_sell_preds}")

        # Evaluar por threshold
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        best_th = 0.5
        best_exp = -999

        for th in thresholds:
            max_probs = y_proba.max(axis=1)
            mask = (y_pred == 1) & (max_probs >= th)
            n_trades = mask.sum()

            if n_trades < 30:
                print(f"    th={th:.2f} -> < 30 trades (invalido)")
                continue

            # Simular PnL de SELL trades
            wins = 0
            losses = 0
            total_win = 0
            total_loss = 0

            for idx in np.where(mask)[0]:
                actual_target = df_test.iloc[idx]["target"]
                pot_win = df_test.iloc[idx]["potential_win"]
                pot_loss = df_test.iloc[idx]["potential_loss"]

                if actual_target == 2:  # Correct SELL
                    pnl = costs.apply_to_pnl(pot_win)
                    wins += 1
                    total_win += pnl
                else:  # Wrong
                    pnl = costs.apply_to_pnl(-pot_loss)
                    losses += 1
                    total_loss += abs(pnl)

            wr = wins / n_trades if n_trades > 0 else 0
            avg_w = total_win / wins if wins > 0 else 0
            avg_l = total_loss / losses if losses > 0 else 0
            exp = (wr * avg_w) - ((1 - wr) * avg_l)
            pf = total_win / total_loss if total_loss > 0 else float("inf")

            print(f"    th={th:.2f} -> {n_trades} trades | Exp: {exp:.6f} | PF: {pf:.4f}")

            if exp > best_exp:
                best_exp = exp
                best_th = th

        all_results.append({"fold": fold + 1, "best_threshold": best_th, "best_exp": best_exp})

    return all_results


def train_sell_model(data_path, model_path="model_sell.joblib", lookahead=20, rr=1.5):
    """Entrena y guarda el modelo SELL."""
    df, feature_cols = prepare_sell_dataset(data_path, lookahead, rr)

    # Walk-forward
    costs = TradingCosts()
    fold_results = walk_forward_sell(df, feature_cols, costs=costs)

    # Threshold optimo
    valid = [r for r in fold_results if r["best_exp"] > -999]
    if valid:
        ths = [r["best_threshold"] for r in valid]
        optimal_th = max(set(ths), key=ths.count)
    else:
        optimal_th = 0.5

    # Entrenar modelo final
    print(f"\nEntrenando modelo SELL final con {len(df)} filas...")
    X = df[feature_cols].values
    y = df["sell_target"].values

    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "threshold": optimal_th,
    }
    joblib.dump(artifact, model_path)
    print(f"Modelo SELL guardado: {model_path} (threshold: {optimal_th})")

    # Feature importance a partir del ensemble calibrado
    importances_array = np.mean([clf.estimator.feature_importances_ for clf in model.calibrated_classifiers_], axis=0)
    importances = pd.Series(importances_array, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print(f"\nTop 10 features SELL:")
    for feat, imp in importances.head(10).items():
        bar = "#" * int(imp * 100)
        print(f"  {feat:25s} {imp:.4f} {bar}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo SELL separado")
    parser.add_argument("--data", required=True, help="CSV con datos OHLC")
    parser.add_argument("--model", default="model_sell.joblib", help="Path de salida")
    parser.add_argument("--rr", type=float, default=1.5, help="R:R ratio")
    parser.add_argument("--lookahead", type=int, default=20, help="Lookahead")

    args = parser.parse_args()
    train_sell_model(args.data, args.model, args.lookahead, args.rr)
