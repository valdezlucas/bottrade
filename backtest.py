"""
Backtester completo que conecta el modelo ML con simulación de trades.

Simula operaciones reales con:
- Señales del modelo ML (BUY/SELL con threshold)
- Stop Loss basado en ATR
- Take Profit basado en R:R ratio
- Costos reales (spread, slippage, swap)
- Equity curve y métricas de rendimiento
"""
import numpy as np
import pandas as pd
import joblib

from data import load_data
from fractals import detect_fractals
from costs import TradingCosts


class Trade:
    """Representa una operación individual."""

    def __init__(self, entry_idx, direction, entry_price, sl, tp, costs_entry):
        self.entry_idx = entry_idx
        self.direction = direction  # "BUY" o "SELL"
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.costs_entry = costs_entry
        self.exit_idx = None
        self.exit_price = None
        self.exit_reason = None  # "TP", "SL", "END"
        self.pnl = None
        self.nights = 0

    def is_open(self):
        return self.exit_idx is None

    def close(self, exit_idx, exit_price, reason, costs_exit, nights=0):
        self.exit_idx = exit_idx
        self.exit_price = exit_price
        self.exit_reason = reason
        self.nights = nights

        if self.direction == "BUY":
            gross_pnl = exit_price - self.entry_price
        else:
            gross_pnl = self.entry_price - exit_price

        total_costs = self.costs_entry + costs_exit
        self.pnl = gross_pnl - total_costs

    def __repr__(self):
        status = "OPEN" if self.is_open() else f"CLOSED ({self.exit_reason})"
        return (f"Trade({self.direction} @ {self.entry_price:.5f} | "
                f"SL: {self.sl:.5f} | TP: {self.tp:.5f} | {status} | "
                f"PnL: {self.pnl if self.pnl else 'N/A'})")


def run_backtest(data_path, model_path="model.joblib", sell_model_path=None,
                 initial_balance=10000,
                 risk_per_trade=0.005, rr_ratio=1.5, atr_sl_multiplier=1.0,
                 max_drawdown_pct=20.0,
                 spread_pips=1.5, max_slippage_pips=1.0, swap_per_night=0.0,
                 pip_value=0.0001, lot_size=100000):
    """
    Ejecuta el backtest completo.

    Args:
        data_path: Path al CSV con datos OHLCV
        model_path: Path al modelo principal (o modelo BUY si sell_model_path se especifica)
        sell_model_path: Path al modelo SELL separado (opcional)
        initial_balance: Capital inicial en USD
        risk_per_trade: % del capital a arriesgar por trade (0.005 = 0.5%)
        rr_ratio: Ratio reward:risk para el TP
        atr_sl_multiplier: Multiplicador de ATR para el SL
        max_drawdown_pct: Max drawdown % permitido antes de parar (default: 20%)
        spread_pips: Spread del broker
        max_slippage_pips: Slippage máximo
        swap_per_night: Costo swap por noche
        pip_value: Valor de 1 pip
        lot_size: Tamaño de 1 lote
    """
    # --- Cargar modelo(s) ---
    artifact = joblib.load(model_path)
    model_buy = artifact["model"]
    feature_cols = artifact["feature_columns"]
    threshold = artifact["threshold"]

    model_sell = None
    sell_threshold = threshold
    if sell_model_path:
        try:
            sell_artifact = joblib.load(sell_model_path)
            model_sell = sell_artifact["model"]
            sell_threshold = sell_artifact["threshold"]
            print(f"  Modelo SELL separado: {sell_model_path}")
        except Exception:
            print(f"  [!] No se pudo cargar modelo SELL: {sell_model_path}")

    print(f"{'='*60}")
    print(f"  BACKTEST — ML Trading Bot")
    print(f"{'='*60}")
    print(f"  Modelo: {model_path} (threshold: {threshold})")
    print(f"  Capital: ${initial_balance:,.2f}")
    print(f"  Riesgo/trade: {risk_per_trade*100:.1f}%")
    print(f"  R:R Ratio: {rr_ratio}")
    print(f"  SL: {atr_sl_multiplier}x ATR")
    print(f"  Spread: {spread_pips} pips")
    print(f"  Max Drawdown Breaker: {max_drawdown_pct}%")

    # --- Preparar datos ---
    print(f"\nCargando datos...")
    df = load_data(data_path)
    df = detect_fractals(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"Datos: {len(df)} velas")

    # --- Generar predicciones ---
    X = df[feature_cols].values
    probas = model_buy.predict_proba(X)
    predictions = probas.argmax(axis=1)
    max_probs = probas.max(axis=1)

    labels_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    costs_sim = TradingCosts(spread_pips, max_slippage_pips, swap_per_night, pip_value)

    # --- Generar predicciones SELL separadas si hay modelo ---
    if model_sell is not None:
        sell_probas = model_sell.predict_proba(X)
        sell_preds = sell_probas.argmax(axis=1)
        sell_max_probs = sell_probas.max(axis=1)
    else:
        sell_probas = None

    # --- Simulación de trades ---
    balance = initial_balance
    peak_balance = initial_balance
    trades = []
    equity_curve = [initial_balance]
    current_trade = None
    dd_breaker_triggered = False
    dd_breaker_bar = None

    for i in range(len(df)):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        close = df["Close"].iloc[i]
        atr = df["ATR"].iloc[i]

        if np.isnan(atr) or atr <= 0:
            equity_curve.append(balance)
            continue

        # --- Verificar si hay trade abierto ---
        if current_trade is not None and current_trade.is_open():
            current_trade.nights += 1

            hit_tp = False
            hit_sl = False

            if current_trade.direction == "BUY":
                if high >= current_trade.tp:
                    hit_tp = True
                if low <= current_trade.sl:
                    hit_sl = True
            else:
                if low <= current_trade.tp:
                    hit_tp = True
                if high >= current_trade.sl:
                    hit_sl = True

            # SL tiene prioridad (peor caso)
            if hit_sl:
                exit_price = current_trade.sl
                exit_cost = costs_sim.exit_cost() + costs_sim.holding_cost(current_trade.nights)
                current_trade.close(i, exit_price, "SL", exit_cost, current_trade.nights)

                # Calcular P&L en USD
                pips_pnl = current_trade.pnl / pip_value
                position_size = (balance * risk_per_trade) / (atr_sl_multiplier * atr / pip_value)
                usd_pnl = pips_pnl * position_size * pip_value * lot_size
                balance += usd_pnl
                trades.append(current_trade)
                current_trade = None

            elif hit_tp:
                exit_price = current_trade.tp
                exit_cost = costs_sim.exit_cost() + costs_sim.holding_cost(current_trade.nights)
                current_trade.close(i, exit_price, "TP", exit_cost, current_trade.nights)

                pips_pnl = current_trade.pnl / pip_value
                position_size = (balance * risk_per_trade) / (atr_sl_multiplier * atr / pip_value)
                usd_pnl = pips_pnl * position_size * pip_value * lot_size
                balance += usd_pnl
                trades.append(current_trade)
                current_trade = None

        # --- Drawdown breaker check ---
        current_dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        if current_dd >= max_drawdown_pct and not dd_breaker_triggered:
            dd_breaker_triggered = True
            dd_breaker_bar = i

        # --- Abrir nuevo trade si no hay posición ---
        if current_trade is None and not dd_breaker_triggered:
            # Decidir señal: usar modelo principal o separado para SELL
            signal = None
            if model_sell is not None:
                # Modelo BUY
                if predictions[i] == 1 and max_probs[i] >= threshold:
                    signal = "BUY"
                # Modelo SELL separado
                elif sell_preds[i] == 1 and sell_max_probs[i] >= sell_threshold:
                    signal = "SELL"
            else:
                pred = predictions[i]
                prob = max_probs[i]
                if pred != 0 and prob >= threshold:
                    signal = labels_map[pred]

            if signal:
                entry_cost = costs_sim.entry_cost()
                sl_distance = atr * atr_sl_multiplier
                tp_distance = sl_distance * rr_ratio

                if signal == "BUY":
                    entry_price = close + entry_cost
                    sl_price = entry_price - sl_distance
                    tp_price = entry_price + tp_distance
                else:
                    entry_price = close - entry_cost
                    sl_price = entry_price + sl_distance
                    tp_price = entry_price - tp_distance

                current_trade = Trade(i, signal, entry_price, sl_price, tp_price, entry_cost)

        # Actualizar equity curve
        equity_curve.append(balance)
        peak_balance = max(peak_balance, balance)

    # Cerrar trade abierto al final
    if current_trade is not None and current_trade.is_open():
        exit_cost = costs_sim.exit_cost()
        current_trade.close(len(df)-1, df["Close"].iloc[-1], "END", exit_cost)
        pips_pnl = current_trade.pnl / pip_value
        atr_last = df["ATR"].iloc[-1]
        position_size = (balance * risk_per_trade) / (atr_sl_multiplier * atr_last / pip_value)
        usd_pnl = pips_pnl * position_size * pip_value * lot_size
        balance += usd_pnl
        trades.append(current_trade)

    # --- Calcular métricas ---
    return _compute_results(trades, equity_curve, initial_balance, balance)


def _compute_results(trades, equity_curve, initial_balance, final_balance):
    """Calcula y muestra todas las métricas del backtest."""

    if len(trades) == 0:
        print("\n  No se ejecutaron trades.")
        return {"trades": [], "equity_curve": equity_curve, "metrics": None}

    # Clasificar trades
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    tp_exits = [t for t in trades if t.exit_reason == "TP"]
    sl_exits = [t for t in trades if t.exit_reason == "SL"]

    win_pnls = [t.pnl for t in wins]
    loss_pnls = [abs(t.pnl) for t in losses]

    n_trades = len(trades)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades if n_trades > 0 else 0

    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0

    # Expectancy en pips
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Profit Factor
    gross_profit = sum(win_pnls)
    gross_loss = sum(loss_pnls)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve metrics
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_drawdown = drawdown.max()
    max_dd_idx = drawdown.argmax()

    # Sharpe Ratio
    returns = np.diff(equity) / equity[:-1]
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Return total
    total_return = (final_balance - initial_balance) / initial_balance * 100

    # Avg holding time
    holding_times = [t.nights for t in trades if t.exit_idx is not None]
    avg_holding = np.mean(holding_times) if holding_times else 0

    # Trades por dirección
    buys = [t for t in trades if t.direction == "BUY"]
    sells = [t for t in trades if t.direction == "SELL"]

    # --- Reporte ---
    print(f"\n{'='*60}")
    print(f"  RESULTADOS DEL BACKTEST")
    print(f"{'='*60}")

    print(f"\n  --- Capital ---")
    print(f"  Balance inicial:  ${initial_balance:,.2f}")
    print(f"  Balance final:    ${final_balance:,.2f}")
    print(f"  Return total:     {total_return:+.2f}%")
    print(f"  Max Drawdown:     {max_drawdown:.2f}%")
    print(f"  Sharpe Ratio:     {sharpe:.4f}")

    print(f"\n  --- Trades ---")
    print(f"  Total trades:     {n_trades}")
    print(f"  Wins:             {n_wins} ({win_rate*100:.1f}%)")
    print(f"  Losses:           {n_losses} ({(1-win_rate)*100:.1f}%)")
    print(f"  BUY trades:       {len(buys)}")
    print(f"  SELL trades:      {len(sells)}")
    print(f"  TP exits:         {len(tp_exits)}")
    print(f"  SL exits:         {len(sl_exits)}")
    print(f"  Avg holding:      {avg_holding:.1f} barras")

    print(f"\n  --- Rendimiento ---")
    print(f"  Avg Win:          {avg_win:.6f}")
    print(f"  Avg Loss:         {avg_loss:.6f}")
    print(f"  Expectancy:       {expectancy:.6f}")
    print(f"  Profit Factor:    {profit_factor:.4f}")

    # Indicadores
    print(f"\n  --- Evaluacion ---")
    if expectancy > 0:
        print(f"  [OK] Expectancy positiva")
    else:
        print(f"  [X]  Expectancy negativa")

    if profit_factor > 1.0:
        print(f"  [OK] Profit Factor > 1.0")
    else:
        print(f"  [X]  Profit Factor < 1.0")

    if max_drawdown < 20:
        print(f"  [OK] Drawdown controlado (<20%)")
    else:
        print(f"  [!]  Drawdown alto (>{max_drawdown:.1f}%)")

    if n_trades >= 30:
        print(f"  [OK] Muestra estadistica valida (>= 30 trades)")
    else:
        print(f"  [!]  Muestra pequena ({n_trades} trades)")

    # Últimos 10 trades
    print(f"\n  --- Ultimos 10 trades ---")
    for t in trades[-10:]:
        emoji = "BUY " if t.direction == "BUY" else "SELL"
        result = "WIN " if t.pnl and t.pnl > 0 else "LOSS"
        pnl_str = f"{t.pnl:+.6f}" if t.pnl else "N/A"
        print(f"  [{t.entry_idx:5d}] {emoji} @ {t.entry_price:.5f} -> "
              f"{t.exit_reason} @ {t.exit_price:.5f} | {result} {pnl_str}")

    metrics = {
        "n_trades": n_trades,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "expectancy": round(expectancy, 6),
        "profit_factor": round(profit_factor, 4),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 4),
        "initial_balance": initial_balance,
        "final_balance": round(final_balance, 2),
        "n_buys": len(buys),
        "n_sells": len(sells),
    }

    return {"trades": trades, "equity_curve": equity_curve, "metrics": metrics}