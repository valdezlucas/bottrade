import numpy as np
import pandas as pd
from costs import TradingCosts


def evaluate_signals(df_test, predictions, probabilities, threshold, costs=None):
    """
    Evalúa señales de trading con métricas financieras completas.

    Args:
        df_test: DataFrame con columnas potential_win, potential_loss, target
        predictions: Array de predicciones del modelo (0=HOLD, 1=BUY, 2=SELL)
        probabilities: Array de probabilidades max por predicción
        threshold: Umbral mínimo de probabilidad para operar
        costs: Instancia de TradingCosts (None = sin costos)

    Returns:
        dict con todas las métricas, o None si < 30 trades
    """
    if costs is None:
        costs = TradingCosts(spread_pips=0, max_slippage_pips=0)

    # Solo tomar señales BUY/SELL que superen el threshold
    mask = (predictions != 0) & (probabilities >= threshold)
    trade_indices = np.where(mask)[0]

    n_trades = len(trade_indices)
    if n_trades < 30:
        return None  # Fold inválido

    # Calcular PnL por trade
    pnls = []
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for idx in trade_indices:
        pred = predictions[idx]
        actual = df_test.iloc[idx]["target"]
        pot_win = df_test.iloc[idx]["potential_win"]
        pot_loss = df_test.iloc[idx]["potential_loss"]

        # Si la predicción coincide con el target real → win
        if pred == actual and actual != 0:
            pnl = costs.apply_to_pnl(pot_win)
            wins += 1
            gross_profit += pnl
        else:
            pnl = costs.apply_to_pnl(-pot_loss)
            losses += 1
            gross_loss += abs(pnl)

        pnls.append(pnl)

    pnls = np.array(pnls)

    # --- Métricas financieras ---
    win_rate = wins / n_trades if n_trades > 0 else 0
    loss_rate = losses / n_trades if n_trades > 0 else 0

    avg_win = gross_profit / wins if wins > 0 else 0
    avg_loss = gross_loss / losses if losses > 0 else 0

    # Expectancy
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    # Profit Factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Return acumulado
    cumulative_return = pnls.sum()

    # Max Drawdown
    equity_curve = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = peak - equity_curve
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

    # Sharpe Ratio (asumiendo trades como retornos)
    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252) if pnls.std() > 0 else 0

    return {
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "expectancy": round(expectancy, 6),
        "profit_factor": round(profit_factor, 4),
        "cumulative_return": round(cumulative_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "sharpe_ratio": round(sharpe, 4),
        "threshold": threshold,
    }


def optimize_threshold(df_test, predictions_proba, model, thresholds=None, costs=None):
    """
    Evalúa múltiples thresholds de probabilidad y devuelve el óptimo.

    Args:
        df_test: DataFrame de test
        predictions_proba: Probabilidades por clase del modelo (n_samples, n_classes)
        model: Modelo entrenado (para obtener clases)
        thresholds: Lista de thresholds a evaluar
        costs: Instancia de TradingCosts

    Returns:
        (best_threshold, best_metrics, all_results)
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    all_results = []
    best_metrics = None
    best_threshold = thresholds[0]

    for th in thresholds:
        # Obtener predicciones con el threshold
        max_probs = predictions_proba.max(axis=1)
        predictions = predictions_proba.argmax(axis=1)

        # Evaluar
        metrics = evaluate_signals(df_test, predictions, max_probs, th, costs)

        result = {"threshold": th, "metrics": metrics}
        all_results.append(result)

        if metrics is not None:
            if best_metrics is None or metrics["expectancy"] > best_metrics["expectancy"]:
                best_metrics = metrics
                best_threshold = th

    return best_threshold, best_metrics, all_results


def print_fold_report(fold_num, metrics, threshold):
    """Imprime un reporte formateado de un fold."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_num} — Threshold: {threshold}")
    print(f"{'='*60}")

    if metrics is None:
        print("  ⚠️  INVÁLIDO — Menos de 30 trades")
        return

    print(f"  Trades:         {metrics['n_trades']} ({metrics['wins']}W / {metrics['losses']}L)")
    print(f"  Win Rate:       {metrics['win_rate']*100:.1f}%")
    print(f"  Avg Win:        {metrics['avg_win']:.6f}")
    print(f"  Avg Loss:       {metrics['avg_loss']:.6f}")
    print(f"  Expectancy:     {metrics['expectancy']:.6f}")
    print(f"  Profit Factor:  {metrics['profit_factor']:.4f}")
    print(f"  Cum. Return:    {metrics['cumulative_return']:.6f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.6f}")
    print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.4f}")

    # Indicadores de calidad
    if metrics["expectancy"] > 0:
        print("  ✅ Expectancy positiva")
    else:
        print("  ❌ Expectancy negativa")

    if metrics["profit_factor"] > 1.0:
        print("  ✅ Profit Factor > 1.0")
    else:
        print("  ❌ Profit Factor < 1.0")
