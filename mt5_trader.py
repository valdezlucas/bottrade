"""
MetaTrader 5 Auto-Trader
=========================
Conecta el bot ML con MetaTrader 5 para ejecutar trades automáticamente.
Puede funcionar en cuenta demo (paper trading) o real.

Flujo:
  1. Conecta a MT5
  2. Descarga datos OHLC directamente de MT5 (más preciso que Yahoo)
  3. Ejecuta el modelo ML
  4. Abre/cierra trades automáticamente con SL/TP

Uso:
    python mt5_trader.py                     # Escanear + ejecutar (demo)
    python mt5_trader.py --scan-only         # Solo escanear, no ejecutar
    python mt5_trader.py --pair EURUSD       # Solo un par
    python mt5_trader.py --balance 5000      # Capital personalizado
    python mt5_trader.py --close-all         # Cerrar todos los trades
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 no instalado. Ejecutá: pip install MetaTrader5")
    sys.exit(1)

from features import create_features
from fractals import detect_fractals
from train import get_feature_columns

try:
    from telegram_alerts import send_signal_alert

    TELEGRAM_ENABLED = True
except Exception:
    TELEGRAM_ENABLED = False

# Configuración de pares MT5 (Activos Robustos OOS 2022-2026)
MT5_PAIRS = {
    # Forex Robusto
    "GBPUSD": {"symbol": "GBPUSD", "spread": 1.2, "pip": 0.0001, "decimals": 5},
    "NZDUSD": {"symbol": "NZDUSD", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "AUDUSD": {"symbol": "AUDUSD", "spread": 1.2, "pip": 0.0001, "decimals": 5},
    "USDCAD": {"symbol": "USDCAD", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "USDCHF": {"symbol": "USDCHF", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "USDJPY": {"symbol": "USDJPY", "spread": 1.2, "pip": 0.01, "decimals": 3},
    "EURJPY": {"symbol": "EURJPY", "spread": 1.5, "pip": 0.01, "decimals": 3},
    "GBPJPY": {"symbol": "GBPJPY", "spread": 2.0, "pip": 0.01, "decimals": 3},
    # Acciones Robustas
    "MSFT": {"symbol": "MSFT", "spread": 5.0, "pip": 0.01, "decimals": 2},
    "TSLA": {"symbol": "TSLA", "spread": 5.0, "pip": 0.01, "decimals": 2},
    "PG": {"symbol": "PG", "spread": 5.0, "pip": 0.01, "decimals": 2},
    "XOM": {"symbol": "XOM", "spread": 5.0, "pip": 0.01, "decimals": 2},
}


def connect_mt5():
    """Inicializa y conecta a MetaTrader 5."""
    if not mt5.initialize():
        print(f"  ❌ No se pudo conectar a MT5: {mt5.last_error()}")
        print(f"     Asegurate de tener MetaTrader 5 abierto y logueado.")
        return False

    info = mt5.account_info()
    if info is None:
        print(f"  ❌ No se pudo obtener info de cuenta: {mt5.last_error()}")
        return False

    print(f"  ✅ Conectado a MT5")
    print(f"     Servidor:  {info.server}")
    print(f"     Cuenta:    {info.login}")
    print(f"     Nombre:    {info.name}")
    print(f"     Balance:   ${info.balance:,.2f}")
    print(f"     Equity:    ${info.equity:,.2f}")
    print(f"     Profit:    ${info.profit:+,.2f}")

    trade_mode = (
        "DEMO"
        if info.trade_mode == 0
        else "REAL" if info.trade_mode == 2 else "CONTEST"
    )
    print(f"     Modo:      {trade_mode}")

    if info.trade_mode == 2:
        print(f"\n  ⚠️  ¡CUENTA REAL! Usá --scan-only para solo ver señales")

    return True


def get_mt5_data(symbol, timeframe=mt5.TIMEFRAME_D1, bars=120):
    """Obtiene datos OHLC directamente de MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        }
    )

    return df[["Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)


def generate_mt5_signal(df, model_artifact):
    """Genera señal ML para la última vela."""
    df = create_features(df)
    df = detect_fractals(df)

    feature_cols = model_artifact["feature_columns"]
    threshold = model_artifact["threshold"]

    last = df.iloc[-1]
    if pd.isna(last[feature_cols]).any():
        return None

    X = df.iloc[[-1]][feature_cols]

    # Modelo principal unificado (Multiclase: 0=HOLD, 1=BUY, 2=SELL)
    main_model = model_artifact["model"]
    probas = main_model.predict_proba(X)[0]

    signal = "HOLD"
    confidence = 0

    if len(probas) >= 3:
        prob_buy = probas[1]
        prob_sell = probas[2]

        if prob_buy >= threshold:
            signal = "BUY"
            confidence = prob_buy
        elif prob_sell >= threshold:
            signal = "SELL"
            confidence = prob_sell
    else:
        # Fallback temporal si detecta un modelo viejo binario
        pred = main_model.predict(X)[0]
        conf = probas.max()
        if pred == 1 and conf >= threshold:
            signal = "BUY"
            confidence = conf
        elif pred == 2 and conf >= threshold:
            signal = "SELL"
            confidence = conf

    return {
        "signal": signal,
        "confidence": confidence,
        "close": float(last["Close"]),
        "atr": float(last["ATR"]),
    }


def get_current_positions(symbol=None):
    """Obtiene posiciones abiertas en MT5."""
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()

    if positions is None:
        return []

    return list(positions)


def open_trade_mt5(symbol, signal, atr, config, risk_pct=0.005, rr_ratio=1.5, tf="1d"):
    """Abre un trade en MT5 con SL y TP."""
    info = mt5.account_info()
    if info is None:
        print(f"    ❌ No se pudo obtener balance")
        return False

    balance = info.balance
    pip = config["pip"]

    # Precios actuales
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"    ❌ No se pudo obtener precio de {symbol}")
        return False

    # SL y TP distances
    sl_distance = atr
    tp_distance = sl_distance * rr_ratio
    sl_pips = sl_distance / pip

    # Position size
    risk_usd = balance * risk_pct
    # Valor por pip para 1 lote (aprox $10 para pares USD)
    pip_value_per_lot = 10.0
    volume = round(risk_usd / (sl_pips * pip_value_per_lot), 2)
    volume = max(0.01, min(volume, 10.0))  # Límites: 0.01 - 10 lotes

    # Verificar símbolo
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        print(f"    ❌ Símbolo {symbol} no encontrado")
        return False

    if not sym_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"    ❌ No se pudo seleccionar {symbol}")
            return False

    # Calcular niveles
    if signal == "BUY":
        price = tick.ask
        sl = round(price - sl_distance, config["decimals"])
        tp = round(price + tp_distance, config["decimals"])
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = round(price + sl_distance, config["decimals"])
        tp = round(price - tp_distance, config["decimals"])
        order_type = mt5.ORDER_TYPE_SELL

    # Preparar request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,  # Max slippage en puntos
        "magic": 202604 if tf == "4h" else 202602,  # ID del bot segun TF
        "comment": f"MLBOT {tf.upper()} {signal}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Enviar orden
    result = mt5.order_send(request)

    if result is None:
        print(f"    ❌ Error al enviar orden: {mt5.last_error()}")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"    ❌ Orden rechazada: {result.comment} (code: {result.retcode})")
        return False

    print(f"    ✅ Orden ejecutada!")
    print(f"       Ticket:  #{result.order}")
    print(f"       Precio:  {result.price:.{config['decimals']}f}")
    print(f"       Volume:  {volume} lotes")
    print(f"       SL:      {sl:.{config['decimals']}f} ({sl_pips:.0f} pips)")
    print(f"       TP:      {tp:.{config['decimals']}f} ({tp_distance/pip:.0f} pips)")
    print(f"       Riesgo:  ${risk_usd:.2f}")

    return True


def close_all_positions():
    """Cierra todas las posiciones abiertas del bot."""
    positions = get_current_positions()
    bot_positions = [p for p in positions if p.magic == 202602]

    if not bot_positions:
        print("  Sin posiciones abiertas del bot.")
        return

    print(f"  Cerrando {len(bot_positions)} posiciones...")

    for pos in bot_positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            continue

        if pos.type == mt5.ORDER_TYPE_BUY:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": 202602,
            "comment": "MLBOT CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"    ✅ Cerrado #{pos.ticket} {pos.symbol} {pos.profit:+.2f}")
        else:
            error = result.comment if result else mt5.last_error()
            print(f"    ❌ Error cerrando #{pos.ticket}: {error}")


def show_positions():
    """Muestra posiciones abiertas del bot."""
    positions = get_current_positions()
    bot_positions = [p for p in positions if p.magic == 202602]

    if not bot_positions:
        print(f"\n  🔒 Sin posiciones abiertas del bot")
        return

    print(f"\n  🔓 POSICIONES ABIERTAS ({len(bot_positions)})")
    print(
        f"  {'Ticket':<12} {'Par':<8} {'Tipo':<6} {'Vol':>6} {'Precio':>10} {'SL':>10} {'TP':>10} {'P&L':>10}"
    )
    print(f"  {'-'*75}")

    total_pnl = 0
    for p in bot_positions:
        tipo = "BUY" if p.type == 0 else "SELL"
        emoji = "🟢" if p.type == 0 else "🔴"
        pnl_emoji = "📈" if p.profit >= 0 else "📉"
        print(
            f"  #{p.ticket:<11} {p.symbol:<8} {emoji}{tipo:<5} {p.volume:>6.2f} "
            f"{p.price_open:>10.5f} {p.sl:>10.5f} {p.tp:>10.5f} "
            f"{pnl_emoji}${p.profit:>+8.2f}"
        )
        total_pnl += p.profit

    print(f"\n  Total P&L: ${total_pnl:+.2f}")


def run_mt5_scanner(
    model_path=None,
    balance_override=None,
    risk_pct=0.005,
    rr_ratio=1.5,
    scan_only=False,
    only_pair=None,
    close_all=False,
    tf="1d",
):
    """Pipeline principal: conectar → escanear → ejecutar."""

    print("═" * 70)
    print(
        f"  🤖  MT5 AUTO-TRADER [{tf.upper()}] — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    print("═" * 70)

    # 1. Conectar
    print(f"\n  [1/3] Conectando a MetaTrader 5...")
    if not connect_mt5():
        mt5.shutdown()
        return

    # Cerrar todo si se pide
    if close_all:
        close_all_positions()
        mt5.shutdown()
        return

    # Mostrar posiciones actuales
    show_positions()

    # Balance
    info = mt5.account_info()
    balance = balance_override or info.balance
    risk_usd = balance * risk_pct
    mode = "SCAN-ONLY" if scan_only else "AUTO-TRADE"

    print(f"\n  [2/3] Configuración")
    print(f"     Modo:      {mode}")
    print(f"     Capital:   ${balance:,.2f}")
    print(f"     Riesgo:    {risk_pct*100:.1f}% (${risk_usd:.2f}/trade)")
    print(f"     R:R:       1:{rr_ratio}")

    # 2. Cargar modelos basados en TF
    if model_path is None:
        model_path = "model_4h.joblib" if tf == "4h" else "model_multi.joblib"

    if not os.path.exists(model_path):
        print(f"\n  ❌ Modelo no encontrado: {model_path}")
        mt5.shutdown()
        return

    model_artifact = joblib.load(model_path)
    threshold = model_artifact["threshold"]
    print(f"     Modelo:    {model_path} (th: {threshold})")

    # Determinar MT5 timeframe
    mt5_tf = mt5.TIMEFRAME_H4 if tf == "4h" else mt5.TIMEFRAME_D1
    magic_num = 202604 if tf == "4h" else 202602  # Magic frames separados

    # 3. Escanear pares
    print(f"\n  [3/3] Escaneando pares...")
    pairs_to_scan = {only_pair: MT5_PAIRS[only_pair]} if only_pair else MT5_PAIRS
    signals_found = []

    cluster_counts = {}
    cluster_risk = {}
    MAX_PER_CLUSTER = 2
    MAX_CLUSTER_RISK_PCT = 0.015

    for pair, config in pairs_to_scan.items():
        symbol = config["symbol"]
        print(f"\n  ── {pair} ", end="", flush=True)

        # Verificar si ya hay posición abierta del bot para este par
        existing = [p for p in get_current_positions(symbol) if p.magic == magic_num]
        if existing:
            print(f"⏩ Ya hay posición abierta (#{existing[0].ticket})")
            continue

        # Obtener datos de MT5
        df = get_mt5_data(symbol, mt5_tf, 120)
        if df is None or len(df) < 60:
            print(f"❌ Sin datos suficientes")
            continue

        # Generar señal
        result = generate_mt5_signal(df, model_artifact)
        if result is None:
            print(f"❌ Error generando señal")
            continue

        signal = result["signal"]

        if signal == "HOLD":
            print(f"⏸️  HOLD (conf: {result['confidence']:.1%})")
            continue

        base_curr = pair[:3]
        quote_curr = pair[3:]

        # 1. Cluster Count Checks
        if (
            cluster_counts.get(base_curr, 0) >= MAX_PER_CLUSTER
            or cluster_counts.get(quote_curr, 0) >= MAX_PER_CLUSTER
        ):
            print(
                f"❌ Rechazada por límite de clúster (Máx {MAX_PER_CLUSTER} para {base_curr}/{quote_curr})"
            )
            continue

        # Calcular niveles y sizing
        atr = result["atr"]
        pip = config["pip"]
        sl_distance = atr
        tp_distance = sl_distance * rr_ratio

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue

        if signal == "BUY":
            entry = tick.ask
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            entry = tick.bid
            sl = entry + sl_distance
            tp = entry - tp_distance

        sl_pips = sl_distance / pip

        base_risk_pct = risk_pct
        max_risk_pct = 0.01
        alpha = 0.5

        confidence = result["confidence"]
        if confidence > threshold:
            scaled_risk = (
                base_risk_pct * ((confidence - threshold) / (1.0 - threshold)) * alpha
                + base_risk_pct
            )
        else:
            scaled_risk = base_risk_pct

        adjusted_risk_pct = min(scaled_risk, max_risk_pct)

        # 2. Cluster Risk Checks
        if (
            cluster_risk.get(base_curr, 0) + adjusted_risk_pct > MAX_CLUSTER_RISK_PCT
            or cluster_risk.get(quote_curr, 0) + adjusted_risk_pct
            > MAX_CLUSTER_RISK_PCT
        ):
            print(
                f"❌ Rechazada por riesgo correlacionado (>{MAX_CLUSTER_RISK_PCT*100}% {base_curr}/{quote_curr})"
            )
            continue

        # Update clusters
        cluster_counts[base_curr] = cluster_counts.get(base_curr, 0) + 1
        cluster_counts[quote_curr] = cluster_counts.get(quote_curr, 0) + 1
        cluster_risk[base_curr] = cluster_risk.get(base_curr, 0) + adjusted_risk_pct
        cluster_risk[quote_curr] = cluster_risk.get(quote_curr, 0) + adjusted_risk_pct

        vol = round((balance * adjusted_risk_pct) / (sl_pips * 10), 2)
        vol = max(0.01, min(vol, 10.0))

        emoji = "🟢 BUY " if signal == "BUY" else "🔴 SELL"
        print(f"")
        print(f"  ┌───────────────────────────────────────────────┐")
        print(f"  │  {emoji} {pair}  —  Confianza: {result['confidence']:.1%}")
        print(f"  ├───────────────────────────────────────────────┤")
        print(f"  │  Entry:      {entry:.{config['decimals']}f}")
        print(f"  │  SL:         {sl:.{config['decimals']}f}  ({sl_pips:.0f} pips)")
        print(
            f"  │  TP:         {tp:.{config['decimals']}f}  ({tp_distance/pip:.0f} pips)"
        )
        print(f"  │  Volumen:    {vol:.2f} lotes")
        print(f"  │  Riesgo:     ${balance * risk_pct:.2f}")
        print(f"  └───────────────────────────────────────────────┘")

        signals_found.append(
            {
                "pair": pair,
                "signal": signal,
                "atr": atr,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "sl_pips": sl_pips,
                "tp_pips": tp_distance / pip,
                "vol": vol,
                "risk_usd": balance * risk_pct,
                "atr_pips": atr / pip,
            }
        )

        # Enviar alerta Telegram
        if TELEGRAM_ENABLED:
            send_signal_alert(
                pair=pair,
                signal=signal,
                entry=round(entry, config["decimals"]),
                sl=round(sl, config["decimals"]),
                tp=round(tp, config["decimals"]),
                sl_pips=sl_pips,
                tp_pips=tp_distance / pip,
                confidence=result["confidence"],
                risk_usd=balance * risk_pct,
                volume=vol,
                atr_pips=atr / pip,
            )

        # Ejecutar si no es scan-only
        if not scan_only:
            print(f"  → Ejecutando orden...")
            open_trade_mt5(symbol, signal, atr, config, risk_pct, rr_ratio, tf)

    # Resumen
    print(f"\n{'═'*70}")
    if signals_found:
        print(f"  📊 {len(signals_found)} señal(es) encontrada(s)")
        for s in signals_found:
            emoji = "🟢" if s["signal"] == "BUY" else "🔴"
            executed = "✅ Ejecutada" if not scan_only else "👁️ Solo escaneada"
            print(f"     {emoji} {s['pair']} {s['signal']} — {executed}")
    else:
        print(f"  ⏸️ Sin señales nuevas")

    # Mostrar posiciones finales
    show_positions()

    print(f"{'═'*70}")

    mt5.shutdown()
    print(f"  MT5 desconectado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT5 Auto-Trader")
    parser.add_argument(
        "--tf", type=str, default="1d", choices=["1d", "4h"], help="Timeframe (1d o 4h)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo principal (opcional, se autoasigna por tf)",
    )
    parser.add_argument("--balance", type=float, default=None, help="Override balance")
    parser.add_argument(
        "--risk", type=float, default=0.005, help="Riesgo %% (default 0.5%%)"
    )
    parser.add_argument("--rr", type=float, default=1.5, help="R:R ratio")
    parser.add_argument("--pair", type=str, default=None, help="Solo un par")
    parser.add_argument(
        "--scan-only", action="store_true", help="Solo escanear, no ejecutar"
    )
    parser.add_argument(
        "--close-all", action="store_true", help="Cerrar todas las posiciones"
    )

    args = parser.parse_args()

    run_mt5_scanner(
        model_path=args.model,
        balance_override=args.balance,
        risk_pct=args.risk,
        rr_ratio=args.rr,
        scan_only=args.scan_only,
        only_pair=args.pair,
        close_all=args.close_all,
        tf=args.tf,
    )
