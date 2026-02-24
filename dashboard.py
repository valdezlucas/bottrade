"""
Paper Trading Dashboard
========================
Lee el trade journal y muestra el estado actual del paper trading.
Permite marcar trades como cerrados (TP/SL hit) y calcular P&L.

Uso:
    python dashboard.py                 # Ver estado
    python dashboard.py --close 3 TP    # Cerrar trade #3 por TP
    python dashboard.py --close 3 SL    # Cerrar trade #3 por SL
    python dashboard.py --stats         # Ver estadísticas
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

JOURNAL_FILE = "trade_journal.csv"


def load_journal():
    """Carga el journal de trades."""
    if not os.path.exists(JOURNAL_FILE):
        print(f"  ❌ No existe {JOURNAL_FILE}")
        print(f"     Ejecutá: python live_scanner.py")
        return None

    df = pd.read_csv(JOURNAL_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def show_dashboard():
    """Muestra el dashboard con trades abiertos y cerrados."""
    df = load_journal()
    if df is None:
        return

    # Filtrar señales (no HOLD)
    trades = df[df["signal"].isin(["BUY", "SELL"])].copy()

    if trades.empty:
        print("\n  Sin trades registrados.")
        return

    open_trades = trades[trades["status"] == "OPEN"]
    closed_trades = trades[trades["status"].isin(["WIN", "LOSS", "BE"])]

    print("═" * 70)
    print(f"  📊  PAPER TRADING DASHBOARD")
    print(f"  Journal: {JOURNAL_FILE} ({len(trades)} trades)")
    print("═" * 70)

    # Trades abiertos
    if not open_trades.empty:
        print(f"\n  🔓 TRADES ABIERTOS ({len(open_trades)})")
        print(
            f"  {'#':<4} {'Fecha':<12} {'Par':<8} {'Señal':<6} {'Entry':>10} {'SL':>10} {'TP':>10} {'Risk$':>8}"
        )
        print(f"  {'-'*68}")

        for idx, row in open_trades.iterrows():
            emoji = "🟢" if row["signal"] == "BUY" else "🔴"
            date_str = row["datetime"].strftime("%m-%d %H:%M")
            entry = f"{row['entry']:.5f}" if pd.notna(row["entry"]) else "—"
            sl = f"{row['sl']:.5f}" if pd.notna(row["sl"]) else "—"
            tp = f"{row['tp']:.5f}" if pd.notna(row["tp"]) else "—"
            risk = f"${row['risk_usd']:.0f}" if pd.notna(row["risk_usd"]) else "—"
            print(
                f"  {idx:<4} {date_str:<12} {row['pair']:<8} {emoji}{row['signal']:<5} {entry:>10} {sl:>10} {tp:>10} {risk:>8}"
            )

        print(f"\n  Para cerrar: python dashboard.py --close <#> TP|SL")
    else:
        print(f"\n  🔒 Sin trades abiertos")

    # Trades cerrados
    if not closed_trades.empty:
        print(f"\n  📜 TRADES CERRADOS ({len(closed_trades)})")
        print(
            f"  {'#':<4} {'Fecha':<12} {'Par':<8} {'Señal':<6} {'Result':<6} {'PnL':>10}"
        )
        print(f"  {'-'*50}")

        total_pnl = 0
        for idx, row in closed_trades.iterrows():
            emoji = "✅" if row["status"] == "WIN" else "❌"
            date_str = row["datetime"].strftime("%m-%d %H:%M")
            pnl = row["pnl_usd"] if pd.notna(row["pnl_usd"]) else 0
            total_pnl += pnl
            pnl_str = f"${pnl:+.2f}" if pnl != 0 else "—"
            print(
                f"  {idx:<4} {date_str:<12} {row['pair']:<8} {row['signal']:<6} {emoji}{row['status']:<5} {pnl_str:>10}"
            )

        print(f"\n  ─────────────────────────")
        color = "📈" if total_pnl >= 0 else "📉"
        print(f"  {color} Total P&L: ${total_pnl:+.2f}")


def show_stats():
    """Muestra estadísticas del paper trading."""
    df = load_journal()
    if df is None:
        return

    trades = df[df["signal"].isin(["BUY", "SELL"])]
    closed = trades[trades["status"].isin(["WIN", "LOSS", "BE"])]

    if closed.empty:
        print("\n  Sin trades cerrados para calcular stats.")
        return

    wins = closed[closed["status"] == "WIN"]
    losses = closed[closed["status"] == "LOSS"]

    n = len(closed)
    n_wins = len(wins)
    wr = n_wins / n if n > 0 else 0

    pnl_values = closed["pnl_usd"].fillna(0)
    total_pnl = pnl_values.sum()

    win_pnls = wins["pnl_usd"].fillna(0)
    loss_pnls = losses["pnl_usd"].fillna(0).abs()

    avg_win = win_pnls.mean() if len(win_pnls) > 0 else 0
    avg_loss = loss_pnls.mean() if len(loss_pnls) > 0 else 0

    gross_win = win_pnls.sum()
    gross_loss = loss_pnls.sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # By pair
    pair_stats = {}
    for pair in closed["pair"].unique():
        p_trades = closed[closed["pair"] == pair]
        p_wins = p_trades[p_trades["status"] == "WIN"]
        p_pnl = p_trades["pnl_usd"].fillna(0).sum()
        pair_stats[pair] = {
            "n": len(p_trades),
            "wins": len(p_wins),
            "pnl": p_pnl,
        }

    print("═" * 60)
    print(f"  📊 ESTADÍSTICAS PAPER TRADING")
    print("═" * 60)
    print(f"\n  Trades cerrados:  {n}")
    print(f"  Wins:             {n_wins} ({wr:.1%})")
    print(f"  Losses:           {len(losses)} ({1-wr:.1%})")
    print(f"  Win Rate:         {wr:.1%}")
    print(f"  Avg Win:          ${avg_win:.2f}")
    print(f"  Avg Loss:         ${avg_loss:.2f}")
    print(f"  Profit Factor:    {pf:.2f}")
    print(f"  Total P&L:        ${total_pnl:+.2f}")

    if pair_stats:
        print(f"\n  Por par:")
        for pair, ps in sorted(
            pair_stats.items(), key=lambda x: x[1]["pnl"], reverse=True
        ):
            wr_p = ps["wins"] / ps["n"] if ps["n"] > 0 else 0
            emoji = "📈" if ps["pnl"] >= 0 else "📉"
            print(
                f"    {emoji} {pair}: {ps['n']} trades, {wr_p:.0%} WR, ${ps['pnl']:+.2f}"
            )


def close_trade(trade_idx, result):
    """Marca un trade como cerrado en el journal."""
    df = load_journal()
    if df is None:
        return

    if trade_idx >= len(df):
        print(f"  ❌ Trade #{trade_idx} no existe")
        return

    row = df.iloc[trade_idx]
    if row["status"] != "OPEN":
        print(f"  ❌ Trade #{trade_idx} no está abierto (status: {row['status']})")
        return

    result = result.upper()
    if result not in ["TP", "SL", "BE", "MANUAL"]:
        print(f"  ❌ Resultado inválido: {result} (usar: TP, SL, BE, MANUAL)")
        return

    # Calcular PnL
    if result == "TP":
        exit_price = row["tp"]
        pnl = row["risk_usd"] * 1.5  # R:R 1.5
        status = "WIN"
    elif result == "SL":
        exit_price = row["sl"]
        pnl = -row["risk_usd"]
        status = "LOSS"
    elif result == "BE":
        exit_price = row["entry"]
        pnl = 0
        status = "BE"
    else:
        exit_price = row["entry"]
        pnl = 0
        status = "MANUAL"

    # Actualizar journal
    df.at[trade_idx, "status"] = status
    df.at[trade_idx, "exit_price"] = exit_price
    df.at[trade_idx, "exit_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.at[trade_idx, "pnl_usd"] = round(pnl, 2)

    df.to_csv(JOURNAL_FILE, index=False)

    emoji = "✅" if status == "WIN" else "❌" if status == "LOSS" else "➖"
    print(
        f"\n  {emoji} Trade #{trade_idx} cerrado: {row['pair']} {row['signal']} → {status} (${pnl:+.2f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper Trading Dashboard")
    parser.add_argument(
        "--close",
        nargs=2,
        metavar=("TRADE_ID", "RESULT"),
        help="Cerrar trade (ej: --close 3 TP)",
    )
    parser.add_argument("--stats", action="store_true", help="Ver estadísticas")

    args = parser.parse_args()

    if args.close:
        close_trade(int(args.close[0]), args.close[1])
    elif args.stats:
        show_stats()
    else:
        show_dashboard()
