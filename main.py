"""
Bot Trade — ML Trading Pipeline
================================
Uso:
    python main.py train --data datos.csv [--spread 1.5] [--rr 1.5] [--lookahead 20]
    python main.py predict --data datos_nuevos.csv [--model model.joblib]
    python main.py backtest --data datos.csv [--model model.joblib] [--balance 10000]
"""

import argparse
import sys

from train import prepare_dataset, walk_forward_train, train_final_model
from predict import predict as run_predict
from costs import TradingCosts
from backtest import run_backtest


def cmd_train(args):
    """Modo entrenamiento con walk-forward validation."""
    costs = TradingCosts(
        spread_pips=args.spread,
        max_slippage_pips=args.slippage,
        swap_per_night=args.swap,
    )

    # Preparar dataset
    df = prepare_dataset(args.data, lookahead=args.lookahead, rr=args.rr)

    if len(df) < 500:
        print(f"\n⚠️  ADVERTENCIA: Solo {len(df)} filas. Se recomiendan mínimo 500 "
              "para walk-forward con 4 folds.")

    # Walk-forward validation
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    fold_results = walk_forward_train(df, n_folds=4, costs=costs, thresholds=thresholds)

    # Resumen de folds
    print(f"\n{'#'*60}")
    print("  RESUMEN WALK-FORWARD")
    print(f"{'#'*60}")

    valid_folds = [r for r in fold_results if r["best_metrics"] is not None]
    invalid_folds = len(fold_results) - len(valid_folds)

    if invalid_folds > 0:
        print(f"  ⚠️  {invalid_folds} folds inválidos (< 30 trades)")

    if valid_folds:
        avg_expectancy = sum(r["best_metrics"]["expectancy"] for r in valid_folds) / len(valid_folds)
        avg_pf = sum(r["best_metrics"]["profit_factor"] for r in valid_folds) / len(valid_folds)
        avg_sharpe = sum(r["best_metrics"]["sharpe_ratio"] for r in valid_folds) / len(valid_folds)

        # Threshold más frecuente o promedio
        best_thresholds = [r["best_threshold"] for r in valid_folds]
        optimal_threshold = max(set(best_thresholds), key=best_thresholds.count)

        print(f"  Folds válidos:       {len(valid_folds)}/{len(fold_results)}")
        print(f"  Avg Expectancy:      {avg_expectancy:.6f}")
        print(f"  Avg Profit Factor:   {avg_pf:.4f}")
        print(f"  Avg Sharpe Ratio:    {avg_sharpe:.4f}")
        print(f"  Threshold óptimo:    {optimal_threshold}")

        if avg_expectancy > 0:
            print("\n  ✅ Expectancy promedio positiva — El modelo tiene potencial")
        else:
            print("\n  ❌ Expectancy promedio negativa — El modelo necesita mejoras")

        # Entrenar modelo final
        train_final_model(df, model_path=args.model, threshold=optimal_threshold)
    else:
        print("  ❌ Ningún fold fue válido. Necesitas más datos o ajustar parámetros.")
        sys.exit(1)


def cmd_predict(args):
    """Modo predicción sobre datos nuevos."""
    signals, df = run_predict(args.data, model_path=args.model)


def cmd_backtest(args):
    """Modo backtest — simula trades con el modelo entrenado."""
    results = run_backtest(
        data_path=args.data,
        model_path=args.model,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        rr_ratio=args.rr,
        atr_sl_multiplier=args.sl_atr,
        spread_pips=args.spread,
        max_slippage_pips=args.slippage,
        swap_per_night=args.swap,
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Bot Trade — ML Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="Entrenar modelo con walk-forward")
    train_parser.add_argument("--data", required=True, help="Path al CSV con datos OHLC(V)")
    train_parser.add_argument("--model", default="model.joblib", help="Path para guardar el modelo")
    train_parser.add_argument("--spread", type=float, default=1.5, help="Spread en pips (default: 1.5)")
    train_parser.add_argument("--slippage", type=float, default=1.0, help="Max slippage en pips (default: 1.0)")
    train_parser.add_argument("--swap", type=float, default=0.0, help="Swap por noche (default: 0)")
    train_parser.add_argument("--rr", type=float, default=1.5, help="Ratio reward:risk (default: 1.5)")
    train_parser.add_argument("--lookahead", type=int, default=20, help="Ventana de lookahead para labels (default: 20)")

    # --- Predict ---
    predict_parser = subparsers.add_parser("predict", help="Predecir sobre datos nuevos")
    predict_parser.add_argument("--data", required=True, help="Path al CSV con datos OHLC(V)")
    predict_parser.add_argument("--model", default="model.joblib", help="Path al modelo guardado")

    # --- Backtest ---
    bt_parser = subparsers.add_parser("backtest", help="Simular trades con el modelo entrenado")
    bt_parser.add_argument("--data", required=True, help="Path al CSV con datos OHLC(V)")
    bt_parser.add_argument("--model", default="model.joblib", help="Path al modelo guardado")
    bt_parser.add_argument("--balance", type=float, default=10000, help="Capital inicial USD (default: 10000)")
    bt_parser.add_argument("--risk", type=float, default=0.01, help="Riesgo por trade como decimal (default: 0.01 = 1%%)")
    bt_parser.add_argument("--rr", type=float, default=1.5, help="Ratio reward:risk (default: 1.5)")
    bt_parser.add_argument("--sl-atr", type=float, default=1.0, help="Multiplicador ATR para SL (default: 1.0)")
    bt_parser.add_argument("--spread", type=float, default=1.5, help="Spread en pips (default: 1.5)")
    bt_parser.add_argument("--slippage", type=float, default=1.0, help="Max slippage en pips (default: 1.0)")
    bt_parser.add_argument("--swap", type=float, default=0.0, help="Swap por noche (default: 0)")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
