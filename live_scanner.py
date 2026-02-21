"""
Live Signal Scanner — Paper Trading
====================================
Descarga datos en tiempo real de Yahoo Finance, ejecuta el modelo ML
en todos los pares configurados, y genera señales con niveles exactos
de SL/TP. Registra todo en un journal CSV para tracking.

Uso:
    python live_scanner.py                # Escanear todos los pares
    python live_scanner.py --pair EURUSD  # Solo un par
    python live_scanner.py --balance 5000 # Capital personalizado

El scanner muestra:
  - Señal actual (BUY/SELL/HOLD) con confianza
  - Precio de entrada, SL y TP exactos
  - Tamaño de posición en lotes
  - Risk/reward y riesgo en USD
"""
import argparse
import sys
import os
import csv
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import yfinance as yf

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from features import create_features
from fractals import detect_fractals
from train import get_feature_columns

try:
    from telegram_alerts import send_signal_alert, send_no_signals_alert
    TELEGRAM_ENABLED = True
except Exception:
    TELEGRAM_ENABLED = False


# Configuración de pares
SCAN_PAIRS = {
    "EURUSD": {"ticker": "EURUSD=X", "spread": 1.0, "pip": 0.0001, "decimals": 5},
    "GBPUSD": {"ticker": "GBPUSD=X", "spread": 1.2, "pip": 0.0001, "decimals": 5},
    "NZDUSD": {"ticker": "NZDUSD=X", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "AUDUSD": {"ticker": "AUDUSD=X", "spread": 1.2, "pip": 0.0001, "decimals": 5},
    "USDCAD": {"ticker": "USDCAD=X", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "USDCHF": {"ticker": "USDCHF=X", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "EURGBP": {"ticker": "EURGBP=X", "spread": 1.5, "pip": 0.0001, "decimals": 5},
}

JOURNAL_FILE = "trade_journal.csv"
JOURNAL_HEADERS = [
    "datetime", "pair", "signal", "confidence", "close", "atr",
    "entry", "sl", "tp", "sl_pips", "tp_pips",
    "risk_usd", "position_lots", "rr_ratio", "status", "exit_price",
    "exit_datetime", "pnl_usd", "notes"
]


def download_live_data(ticker, lookback_days=120):
    """Descarga datos recientes para generar features."""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
    )

    if df.empty:
        return None

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index(drop=True)
    df = df.rename(columns={"Adj Close": "Close"}) if "Adj Close" in df.columns else df

    # Ensure standard column names
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return None

    return df


def generate_signal(df, model_artifact, sell_artifact, pair_config):
    """Genera señal para la última vela."""
    # Aplicar features
    df = create_features(df)
    df = detect_fractals(df)

    feature_cols = model_artifact["feature_columns"]
    threshold = model_artifact["threshold"]

    # Verificar que tenemos todas las features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return None, f"Features faltantes: {missing}"

    # Tomar última fila
    last = df.iloc[-1]

    if pd.isna(last[feature_cols]).any():
        return None, "Features con NaN en última vela"

    X = last[feature_cols].values.reshape(1, -1)

    # --- Modelo principal (BUY) ---
    main_model = model_artifact["model"]
    main_proba = main_model.predict_proba(X)[0]
    main_pred = main_model.predict(X)[0]
    main_conf = main_proba.max()

    # --- Modelo SELL ---
    sell_model = sell_artifact["model"]
    sell_proba = sell_model.predict_proba(X)[0]
    sell_pred = sell_model.predict(X)[0]  # 0=NO_SELL, 1=SELL
    sell_conf = sell_proba[1] if len(sell_proba) > 1 else 0

    # Determinar señal
    signal = "HOLD"
    confidence = 0

    # BUY: modelo principal predice BUY (1) con confianza >= threshold
    if main_pred == 1 and main_conf >= threshold:
        signal = "BUY"
        confidence = main_conf

    # SELL: modelo SELL predice SELL (1) con confianza >= threshold
    elif sell_pred == 1 and sell_conf >= threshold:
        signal = "SELL"
        confidence = sell_conf

    # También chequear si el modelo principal predice SELL (2)
    elif main_pred == 2 and main_conf >= threshold:
        signal = "SELL"
        confidence = main_conf

    # Info de la vela actual
    close = float(last["Close"])
    atr = float(last["ATR"])
    pip = pair_config["pip"]
    spread_cost = pair_config["spread"] * pip

    result = {
        "signal": signal,
        "confidence": round(confidence, 4),
        "close": close,
        "atr": atr,
        "atr_pips": round(atr / pip, 1),
        "spread_pips": pair_config["spread"],
        "all_probs": {
            "main": [round(p, 4) for p in main_proba],
            "sell": round(sell_conf, 4),
        },
    }

    if signal != "HOLD":
        sl_distance = atr  # 1x ATR
        tp_distance = sl_distance * 1.5  # 1.5x R:R

        if signal == "BUY":
            entry = close + spread_cost
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            entry = close - spread_cost
            sl = entry + sl_distance
            tp = entry - tp_distance

        result.update({
            "entry": round(entry, pair_config["decimals"]),
            "sl": round(sl, pair_config["decimals"]),
            "tp": round(tp, pair_config["decimals"]),
            "sl_pips": round(sl_distance / pip, 1),
            "tp_pips": round(tp_distance / pip, 1),
        })

    return result, None


def calculate_position(signal_result, balance, risk_pct=0.005, pip_value_usd=10):
    """Calcula tamaño de posición y riesgo en USD."""
    if signal_result["signal"] == "HOLD":
        return {}

    risk_usd = balance * risk_pct
    sl_pips = signal_result["sl_pips"]

    # Valor por pip para 1 lote estándar (100,000 unidades) ≈ $10
    position_lots = risk_usd / (sl_pips * pip_value_usd)
    position_lots = round(position_lots, 2)

    # Mini lotes (0.01 = 1000 unidades)
    mini_lots = round(position_lots, 2)

    return {
        "risk_usd": round(risk_usd, 2),
        "position_lots": mini_lots,
        "position_units": int(mini_lots * 100000),
        "potential_win": round(risk_usd * 1.5, 2),  # R:R 1.5
        "potential_loss": round(risk_usd, 2),
    }


def log_to_journal(pair, signal_result, position):
    """Registra la señal en el journal CSV."""
    file_exists = os.path.exists(JOURNAL_FILE)

    with open(JOURNAL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(JOURNAL_HEADERS)

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pair,
            signal_result.get("signal", "HOLD"),
            signal_result.get("confidence", 0),
            signal_result.get("close", 0),
            signal_result.get("atr", 0),
            signal_result.get("entry", ""),
            signal_result.get("sl", ""),
            signal_result.get("tp", ""),
            signal_result.get("sl_pips", ""),
            signal_result.get("tp_pips", ""),
            position.get("risk_usd", ""),
            position.get("position_lots", ""),
            1.5,  # rr_ratio
            "OPEN" if signal_result["signal"] != "HOLD" else "",
            "",  # exit_price
            "",  # exit_datetime
            "",  # pnl_usd
            "",  # notes
        ]
        writer.writerow(row)


def scan_all_pairs(model_path="model_multi.joblib",
                   sell_model_path="model_multi_sell.joblib",
                   balance=10000, risk_pct=0.005, only_pair=None):
    """Escanea todos los pares y muestra señales."""

    print("═" * 70)
    print(f"  🔍  LIVE SIGNAL SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Capital: ${balance:,.2f} | Riesgo: {risk_pct*100:.1f}% (${balance*risk_pct:.2f}/trade)")
    print("═" * 70)

    # Cargar modelos
    if not os.path.exists(model_path):
        print(f"\n  ❌ Modelo no encontrado: {model_path}")
        print(f"     Ejecutá primero: python multi_pair_oos.py --split-year 2021")
        return

    model_artifact = joblib.load(model_path)
    sell_artifact = joblib.load(sell_model_path)
    print(f"  Modelo: {model_path} (threshold: {model_artifact['threshold']})")

    pairs_to_scan = {only_pair: SCAN_PAIRS[only_pair]} if only_pair else SCAN_PAIRS
    signals_found = []

    for pair, config in pairs_to_scan.items():
        print(f"\n  Descargando {pair}...", end=" ", flush=True)

        df = download_live_data(config["ticker"])
        if df is None or len(df) < 60:
            print(f"❌ Sin datos")
            continue

        result, error = generate_signal(df, model_artifact, sell_artifact, config)
        if error:
            print(f"❌ {error}")
            continue

        signal = result["signal"]

        if signal == "HOLD":
            print(f"⏸️  HOLD (conf: {result['confidence']:.1%})")
            continue

        # Calcular posición
        position = calculate_position(result, balance, risk_pct)

        # Mostrar señal
        emoji = "🟢 BUY " if signal == "BUY" else "🔴 SELL"
        print(f"")
        print(f"  ┌──────────────────────────────────────────────────┐")
        print(f"  │  {emoji} {pair}  —  Confianza: {result['confidence']:.1%}")
        print(f"  ├──────────────────────────────────────────────────┤")
        print(f"  │  Precio actual:  {result['close']:.{config['decimals']}f}")
        print(f"  │  Entry:          {result['entry']:.{config['decimals']}f}")
        print(f"  │  Stop Loss:      {result['sl']:.{config['decimals']}f}  ({result['sl_pips']:.0f} pips)")
        print(f"  │  Take Profit:    {result['tp']:.{config['decimals']}f}  ({result['tp_pips']:.0f} pips)")
        print(f"  │  ATR:            {result['atr_pips']:.0f} pips")
        print(f"  ├──────────────────────────────────────────────────┤")
        print(f"  │  Posición:       {position['position_lots']:.2f} lotes ({position['position_units']:,} uds)")
        print(f"  │  Riesgo:         ${position['risk_usd']:.2f}")
        print(f"  │  Si TP:          +${position['potential_win']:.2f}")
        print(f"  │  Si SL:          -${position['potential_loss']:.2f}")
        print(f"  │  R:R:            1:1.5")
        print(f"  └──────────────────────────────────────────────────┘")

        # Registrar en journal
        log_to_journal(pair, result, position)
        signals_found.append({"pair": pair, **result, **position})

        # Enviar alerta Telegram
        if TELEGRAM_ENABLED:
            send_signal_alert(
                pair=pair,
                signal=signal,
                entry=result["entry"],
                sl=result["sl"],
                tp=result["tp"],
                sl_pips=result["sl_pips"],
                tp_pips=result["tp_pips"],
                confidence=result["confidence"],
                risk_usd=position["risk_usd"],
                volume=position["position_lots"],
                atr_pips=result["atr_pips"],
            )

    # Resumen
    print(f"\n{'═'*70}")
    if signals_found:
        print(f"  📊  {len(signals_found)} SEÑAL(ES) ENCONTRADA(S)")
        for s in signals_found:
            emoji = "🟢" if s["signal"] == "BUY" else "🔴"
            print(f"      {emoji} {s['pair']} {s['signal']} @ {s['entry']} → SL:{s['sl']} TP:{s['tp']}")
        print(f"\n  📁 Registrado en: {JOURNAL_FILE}")
    else:
        print(f"  ⏸️  Sin señales hoy — modelo en HOLD para todos los pares")
        # (Opcional) notificar a Telegram que no hubo señales
        # if TELEGRAM_ENABLED: send_no_signals_alert()

    print(f"{'═'*70}")

    return signals_found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Signal Scanner")
    parser.add_argument("--model", default="model_multi.joblib", help="Modelo principal")
    parser.add_argument("--sell-model", default="model_multi_sell.joblib", help="Modelo SELL")
    parser.add_argument("--balance", type=float, default=10000, help="Capital")
    parser.add_argument("--risk", type=float, default=0.005, help="Riesgo por trade (0.005 = 0.5%%)")
    parser.add_argument("--pair", type=str, default=None, help="Solo escanear un par (ej: EURUSD)")

    args = parser.parse_args()

    scan_all_pairs(
        model_path=args.model,
        sell_model_path=args.sell_model,
        balance=args.balance,
        risk_pct=args.risk,
        only_pair=args.pair,
    )
