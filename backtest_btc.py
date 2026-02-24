"""
Multi-Timeframe Backtest
========================
Downloads data from yfinance for Daily, 4H, and 1H,
then runs each trained model against it and prints results.

Usage:
    python backtest_btc.py                           # BTC-USD default
    python backtest_btc.py --pair GBPJPY --ticker GBPJPY=X --pip 0.01 --spread 2.0 --lot-size 100000
    python backtest_btc.py --pair EURUSD --ticker EURUSD=X --pip 0.0001 --spread 1.0 --lot-size 100000
"""

import argparse
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from costs import TradingCosts
from features import create_features
from fractals import detect_fractals

# Models per timeframe
MODELS = {
    "1H": {"buy": "model_1h.joblib", "sell": "model_1h_sell.joblib"},
    "4H": {"buy": "model_4h.joblib", "sell": "model_4h_sell.joblib"},
    "Daily": {"buy": "model_multi.joblib", "sell": "model_multi_sell.joblib"},
}


class Trade:
    def __init__(self, entry_idx, direction, entry_price, sl, tp, costs_entry):
        self.entry_idx = entry_idx
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.costs_entry = costs_entry
        self.exit_idx = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = None
        self.nights = 0

    def is_open(self):
        return self.exit_price is None

    def close(self, exit_idx, exit_price, reason, costs_exit, nights=0):
        self.exit_idx = exit_idx
        self.exit_price = exit_price
        self.exit_reason = reason
        self.nights = nights
        if self.direction == "BUY":
            self.pnl = (exit_price - self.entry_price) - self.costs_entry - costs_exit
        else:
            self.pnl = (self.entry_price - exit_price) - self.costs_entry - costs_exit


# ─── Data download ──────────────────────────────────────────────────────
def download_pair(ticker, pair_name, interval="1d", days=730):
    """Download data from yfinance."""
    print(f"  Descargando {pair_name} ({interval}, {days} dias)...")
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
    )
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    for col in ["Datetime", "Date"]:
        if col in df.columns:
            df = df.rename(columns={col: "Datetime"})
            break

    print(f"  {len(df)} velas descargadas")
    return df


def resample_to_4h(df_1h):
    """Resample 1H data to 4H."""
    if df_1h is None or df_1h.empty:
        return None
    df = df_1h.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")

    ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        ohlc["Volume"] = "sum"
    df_4h = df.resample("4h").agg(ohlc).dropna()
    print(f"  Resampled a {len(df_4h)} velas de 4H")
    return df_4h.reset_index(drop=True)


# ─── Backtest engine ────────────────────────────────────────────────────
def run_pair_backtest(
    df,
    model_path,
    sell_model_path,
    tf_name,
    pip_value=0.0001,
    spread_pips=1.5,
    lot_size=100000,
    risk_per_trade=0.005,
    rr_ratio=1.5,
    max_dd_pct=25.0,
    initial_balance=10000,
):
    """Run backtest on any pair data with the specified model."""

    # Load models
    try:
        artifact = joblib.load(model_path)
        model_buy = artifact["model"]
        feature_cols = artifact["feature_columns"]
        threshold = artifact["threshold"]
    except FileNotFoundError:
        print(f"  [!] Modelo no encontrado: {model_path}")
        return None

    try:
        sell_artifact = joblib.load(sell_model_path)
        model_sell = sell_artifact["model"]
        sell_threshold = sell_artifact["threshold"]
    except FileNotFoundError:
        model_sell = None
        sell_threshold = threshold

    # Prepare data
    df_work = df[["Open", "High", "Low", "Close"]].copy()
    if "Volume" in df.columns:
        df_work["Volume"] = df["Volume"].values

    df_work = create_features(df_work)
    df_work = detect_fractals(df_work)

    missing = [c for c in feature_cols if c not in df_work.columns]
    if missing:
        print(f"  [!] Features faltantes: {missing}")
        return None

    df_work = df_work.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"  Datos preparados: {len(df_work)} velas")

    # Generate predictions
    X = df_work[feature_cols].values
    probas = model_buy.predict_proba(X)
    predictions = probas.argmax(axis=1)
    max_probs = probas.max(axis=1)

    if model_sell is not None:
        sell_probas = model_sell.predict_proba(X)
        sell_preds = sell_probas.argmax(axis=1)
        sell_max_probs = sell_probas.max(axis=1)
    else:
        sell_probas = None

    # Costs
    costs_sim = TradingCosts(
        spread_pips=spread_pips,
        max_slippage_pips=max(1.0, spread_pips * 0.3),
        swap_per_night=0.0,
        pip_value=pip_value,
    )

    # Simulation
    balance = initial_balance
    peak_balance = initial_balance
    trades = []
    equity_curve = [initial_balance]
    current_trade = None
    dd_breaker_triggered = False

    for i in range(len(df_work)):
        high = df_work["High"].iloc[i]
        low = df_work["Low"].iloc[i]
        close = df_work["Close"].iloc[i]
        atr = df_work["ATR"].iloc[i]

        if np.isnan(atr) or atr <= 0:
            equity_curve.append(balance)
            continue

        # Check open trade
        if current_trade is not None and current_trade.is_open():
            current_trade.nights += 1
            hit_tp = hit_sl = False

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

            if hit_sl:
                exit_price = current_trade.sl
                exit_cost = costs_sim.exit_cost() + costs_sim.holding_cost(
                    current_trade.nights
                )
                current_trade.close(
                    i, exit_price, "SL", exit_cost, current_trade.nights
                )

                risk_usd = balance * risk_per_trade
                usd_pnl = current_trade.pnl * (risk_usd / atr)
                balance += usd_pnl
                trades.append(current_trade)
                current_trade = None

            elif hit_tp:
                exit_price = current_trade.tp
                exit_cost = costs_sim.exit_cost() + costs_sim.holding_cost(
                    current_trade.nights
                )
                current_trade.close(
                    i, exit_price, "TP", exit_cost, current_trade.nights
                )

                risk_usd = balance * risk_per_trade
                usd_pnl = current_trade.pnl * (risk_usd / atr)
                balance += usd_pnl
                trades.append(current_trade)
                current_trade = None

        # Drawdown check
        current_dd = (
            (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        )
        if current_dd >= max_dd_pct and not dd_breaker_triggered:
            dd_breaker_triggered = True

        # Open new trade
        if current_trade is None and not dd_breaker_triggered:
            signal = None
            if model_sell is not None:
                if predictions[i] == 1 and max_probs[i] >= threshold:
                    signal = "BUY"
                elif sell_preds[i] == 1 and sell_max_probs[i] >= sell_threshold:
                    signal = "SELL"
            else:
                pred = predictions[i]
                prob = max_probs[i]
                if pred != 0 and prob >= threshold:
                    signal = {0: "HOLD", 1: "BUY", 2: "SELL"}[pred]

            if signal:
                entry_cost = costs_sim.entry_cost()
                sl_distance = atr * 1.0
                tp_distance = sl_distance * rr_ratio

                if signal == "BUY":
                    entry_price = close + entry_cost
                    sl_price = entry_price - sl_distance
                    tp_price = entry_price + tp_distance
                else:
                    entry_price = close - entry_cost
                    sl_price = entry_price + sl_distance
                    tp_price = entry_price - tp_distance

                current_trade = Trade(
                    i, signal, entry_price, sl_price, tp_price, entry_cost
                )

        equity_curve.append(balance)
        peak_balance = max(peak_balance, balance)

    # Close open trade at end
    if current_trade is not None and current_trade.is_open():
        exit_cost = costs_sim.exit_cost()
        current_trade.close(
            len(df_work) - 1, df_work["Close"].iloc[-1], "END", exit_cost
        )
        atr_last = df_work["ATR"].iloc[-1]
        risk_usd = balance * risk_per_trade
        usd_pnl = current_trade.pnl * (risk_usd / atr_last) if atr_last > 0 else 0
        balance += usd_pnl
        trades.append(current_trade)

    return compute_results(tf_name, trades, equity_curve, initial_balance, balance)


def compute_results(tf_name, trades, equity_curve, initial_balance, final_balance):
    """Compute and print backtest metrics."""
    n_trades = len(trades)
    if n_trades == 0:
        print(f"  Sin trades generados")
        return {"tf": tf_name, "trades": 0}

    wins = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades * 100

    total_return = (final_balance - initial_balance) / initial_balance * 100

    # Max drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    max_dd = dd.max()

    # Trades by direction
    buys = [t for t in trades if t.direction == "BUY"]
    sells = [t for t in trades if t.direction == "SELL"]
    buy_wins = len([t for t in buys if t.pnl and t.pnl > 0])
    sell_wins = len([t for t in sells if t.pnl and t.pnl > 0])

    # Avg hold time
    avg_bars = np.mean([t.nights for t in trades]) if trades else 0

    # TP / SL counts
    tp_count = len([t for t in trades if t.exit_reason == "TP"])
    sl_count = len([t for t in trades if t.exit_reason == "SL"])
    end_count = len([t for t in trades if t.exit_reason == "END"])

    # Profit factor
    total_profit = sum(t.pnl for t in wins if t.pnl) if wins else 0
    total_loss = abs(sum(t.pnl for t in losses if t.pnl)) if losses else 0
    pf = total_profit / total_loss if total_loss > 0 else float("inf")

    # Consecutive stats
    max_cons_wins = max_cons_losses = 0
    current_streak = 0
    last_was_win = None
    for t in trades:
        if t.pnl and t.pnl > 0:
            if last_was_win:
                current_streak += 1
            else:
                current_streak = 1
            last_was_win = True
            max_cons_wins = max(max_cons_wins, current_streak)
        else:
            if not last_was_win and last_was_win is not None:
                current_streak += 1
            else:
                current_streak = 1
            last_was_win = False
            max_cons_losses = max(max_cons_losses, current_streak)

    # Print
    emoji = {"1H": "---", "4H": "----", "Daily": "-----"}
    print(f"\n{'='*60}")
    print(f"  BTC-USD BACKTEST — {tf_name}")
    print(f"{'='*60}")
    print(f"  Capital inicial: ${initial_balance:,.2f}")
    print(f"  Capital final:   ${final_balance:,.2f}")
    print(f"  Retorno:         {total_return:+.2f}%")
    print(f"  Max Drawdown:    {max_dd:.2f}%")
    print(f"{'-'*60}")
    print(f"  Total trades:    {n_trades}")
    print(f"  Wins:            {n_wins} ({win_rate:.1f}%)")
    print(f"  Losses:          {n_losses}")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"{'-'*60}")
    print(f"  BUY trades:      {len(buys)} (wins: {buy_wins})")
    print(f"  SELL trades:     {len(sells)} (wins: {sell_wins})")
    print(f"  TP exits:        {tp_count}")
    print(f"  SL exits:        {sl_count}")
    print(f"  END (open):      {end_count}")
    print(f"{'-'*60}")
    print(f"  Avg hold (bars): {avg_bars:.1f}")
    print(f"  Max cons. wins:  {max_cons_wins}")
    print(f"  Max cons. losses:{max_cons_losses}")
    print(f"{'='*60}")

    return {
        "tf": tf_name,
        "trades": n_trades,
        "win_rate": win_rate,
        "return_pct": total_return,
        "max_dd": max_dd,
        "profit_factor": pf,
        "final_balance": final_balance,
    }


# ─── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Timeframe Backtest")
    parser.add_argument(
        "--pair",
        type=str,
        default="BTC-USD",
        help="Pair name (e.g. GBPJPY, EURUSD, BTC-USD)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="yfinance ticker (e.g. GBPJPY=X, BTC-USD)",
    )
    parser.add_argument("--pip", type=float, default=None, help="Pip value")
    parser.add_argument("--spread", type=float, default=None, help="Spread in pips")
    parser.add_argument("--lot-size", type=int, default=None, help="Lot size")
    parser.add_argument(
        "--risk", type=float, default=0.005, help="Risk per trade (default: 0.5%%)"
    )
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    args = parser.parse_args()

    # Presets
    PRESETS = {
        "BTC-USD": {"ticker": "BTC-USD", "pip": 1.0, "spread": 30.0, "lot_size": 1},
        "GBPJPY": {
            "ticker": "GBPJPY=X",
            "pip": 0.01,
            "spread": 2.0,
            "lot_size": 100000,
        },
        "EURJPY": {
            "ticker": "EURJPY=X",
            "pip": 0.01,
            "spread": 1.5,
            "lot_size": 100000,
        },
        "USDJPY": {
            "ticker": "USDJPY=X",
            "pip": 0.01,
            "spread": 1.0,
            "lot_size": 100000,
        },
        "EURUSD": {
            "ticker": "EURUSD=X",
            "pip": 0.0001,
            "spread": 1.0,
            "lot_size": 100000,
        },
        "GBPUSD": {
            "ticker": "GBPUSD=X",
            "pip": 0.0001,
            "spread": 1.2,
            "lot_size": 100000,
        },
        "AUDUSD": {
            "ticker": "AUDUSD=X",
            "pip": 0.0001,
            "spread": 1.2,
            "lot_size": 100000,
        },
        "NZDUSD": {
            "ticker": "NZDUSD=X",
            "pip": 0.0001,
            "spread": 1.5,
            "lot_size": 100000,
        },
        "USDCAD": {
            "ticker": "USDCAD=X",
            "pip": 0.0001,
            "spread": 1.5,
            "lot_size": 100000,
        },
        "USDCHF": {
            "ticker": "USDCHF=X",
            "pip": 0.0001,
            "spread": 1.5,
            "lot_size": 100000,
        },
        "EURGBP": {
            "ticker": "EURGBP=X",
            "pip": 0.0001,
            "spread": 1.5,
            "lot_size": 100000,
        },
    }

    preset = PRESETS.get(args.pair, {})
    ticker = args.ticker or preset.get("ticker", f"{args.pair}=X")
    pip_value = args.pip or preset.get("pip", 0.0001)
    spread = args.spread or preset.get("spread", 1.5)
    lot_size = args.lot_size or preset.get("lot_size", 100000)
    initial_balance = args.balance
    risk = args.risk

    bt_kwargs = dict(
        pip_value=pip_value,
        spread_pips=spread,
        lot_size=lot_size,
        risk_per_trade=risk,
        rr_ratio=1.5,
        max_dd_pct=25.0,
        initial_balance=initial_balance,
    )

    print("=" * 60)
    print(f"  {args.pair} MULTI-TIMEFRAME BACKTEST")
    print(f"  Pip: {pip_value} | Spread: {spread} pips | Lot: {lot_size}")
    print("=" * 60)

    results = []

    # ─── Daily ──────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  [1/3] DAILY -- {args.pair}")
    print(f"{'#'*60}")

    df_daily = download_pair(ticker, args.pair, interval="1d", days=1500)
    if df_daily is not None:
        df_daily_clean = df_daily.drop(columns=["Datetime"], errors="ignore")
        r = run_pair_backtest(
            df_daily_clean,
            MODELS["Daily"]["buy"],
            MODELS["Daily"]["sell"],
            "Daily",
            **bt_kwargs,
        )
        if r:
            results.append(r)

    # ─── 4H ─────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  [2/3] 4H -- {args.pair}")
    print(f"{'#'*60}")

    df_1h_raw = download_pair(ticker, args.pair, interval="1h", days=720)
    if df_1h_raw is not None:
        df_4h = resample_to_4h(df_1h_raw)
        if df_4h is not None:
            r = run_pair_backtest(
                df_4h, MODELS["4H"]["buy"], MODELS["4H"]["sell"], "4H", **bt_kwargs
            )
            if r:
                results.append(r)

    # ─── 1H ─────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  [3/3] 1H -- {args.pair}")
    print(f"{'#'*60}")

    if df_1h_raw is not None:
        df_1h_clean = df_1h_raw.drop(columns=["Datetime"], errors="ignore")
        r = run_pair_backtest(
            df_1h_clean, MODELS["1H"]["buy"], MODELS["1H"]["sell"], "1H", **bt_kwargs
        )
        if r:
            results.append(r)

    # ─── Resumen comparativo ────────────────────────────────────────
    if results:
        print(f"\n{'='*60}")
        print(f"  RESUMEN COMPARATIVO -- {args.pair}")
        print(f"{'='*60}")
        print(
            f"  {'TF':<8} {'Trades':>7} {'WinRate':>8} {'Return':>9} {'MaxDD':>7} {'PF':>6} {'Final$':>10}"
        )
        print(f"  {'-'*58}")
        for r in results:
            if r.get("trades", 0) > 0:
                print(
                    f"  {r['tf']:<8} {r['trades']:>7} {r['win_rate']:>7.1f}% {r['return_pct']:>+8.2f}% {r['max_dd']:>6.2f}% {r['profit_factor']:>5.2f} ${r['final_balance']:>9,.2f}"
                )
            else:
                print(f"  {r['tf']:<8} {'No trades':>50}")
        print(f"{'='*60}")
