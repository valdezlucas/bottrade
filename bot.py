"""
Trading Signal Bot — Telegram Multi-Subscriber
================================================
Bot en tiempo real que:
  - Cualquier persona que mande /start se suscribe automáticamente
  - Escanea los pares Forex 1x por día (horario configurable)
  - Transmite señales a TODOS los suscriptores simultáneamente
  - Guarda suscriptores en subscribers.json (persiste entre reinicios)

Comandos del bot:
  /start    → Suscribirse a señales
  /stop     → Desuscribirse
  /status   → Ver estado del bot y último scan
  /signal   → Forzar un scan ahora mismo

Uso:
  python bot.py                  # Correr el bot (local o cloud)
  python bot.py --scan-now       # Solo hacer un scan y broadcast
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import requests
import schedule

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from features import create_features
from firebase_manager import (
    SIGNAL_COST,
    db,
    db_add_subscriber,
    db_can_receive_signal,
    db_close_signal,
    db_deposit,
    db_get_active_signals,
    db_get_subscribers,
    db_get_user_account,
    db_init_user_account,
    db_is_trial_active,
    db_remove_subscriber,
    db_save_signal,
    db_toggle_alerts,
)
from fractals import detect_fractals

# ─── CONFIG ────────────────────────────────────────────────────────────
BOT_TOKEN = "5967657374:AAHX9XuJBmRxIYWn9AgcsCBtTK5mr3O2yTY"
# Modelos por timeframe
MODELS = {
    "4H": "model_4h.joblib",
    "Daily": "model_multi.joblib",
    "BTC_Daily": "model_btc_daily.joblib",
}
TF_EMOJIS = {"4H": "⏳", "Daily": "📅", "BTC_Daily": "₿"}
MODEL_PATH = "model_multi.joblib"  # legacy fallback
SUBSCRIBERS_FILE = "subscribers.json"
SCAN_HOUR = 22  # Hora en que escanea (22:00 UTC-3 = cierre vela diaria NY)
RISK_PCT = 0.005  # 0.5% riesgo por trade (para calcular lotes)
DEFAULT_BALANCE = 10000  # Balance de referencia para calcular lotes
# ──────────────────────────────────────────────────────────────────────

# Pares a escanear (Solo los que pasaron validación OOS 2022-2026)
PAIRS = {
    # Forex Robusto (PF > 1.3)
    "GBPUSD": {"ticker": "GBPUSD", "spread": 1.2, "pip": 0.0001, "decimals": 5},
    "NZDUSD": {"ticker": "NZDUSD", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "AUDUSD": {"ticker": "AUDUSD", "spread": 1.2, "pip": 0.0001, "decimals": 5},
    "USDCAD": {"ticker": "USDCAD", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "USDCHF": {"ticker": "USDCHF", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "USDJPY": {"ticker": "USDJPY", "spread": 1.2, "pip": 0.01, "decimals": 3},
    "EURJPY": {"ticker": "EURJPY", "spread": 1.5, "pip": 0.01, "decimals": 3},
    "GBPJPY": {"ticker": "GBPJPY", "spread": 2.0, "pip": 0.01, "decimals": 3},
    # Pares Cruzados Nuevos (PF > 1.2 OOS)
    "EURGBP": {"ticker": "EURGBP", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "EURAUD": {"ticker": "EURAUD", "spread": 2.0, "pip": 0.0001, "decimals": 5},
    "GBPAUD": {"ticker": "GBPAUD", "spread": 2.5, "pip": 0.0001, "decimals": 5},
    # Acciones Robustas (PF > 1.3)
    "MSFT": {"ticker": "MSFT", "spread": 5.0, "pip": 0.01, "decimals": 2},
    "TSLA": {"ticker": "TSLA", "spread": 5.0, "pip": 0.01, "decimals": 2},
    "PG": {"ticker": "PG", "spread": 5.0, "pip": 0.01, "decimals": 2},
    "XOM": {"ticker": "XOM", "spread": 5.0, "pip": 0.01, "decimals": 2},
}

# Bitcoin — par separado con modelo propio
BTC_PAIRS = {
    "BTCUSD": {"ticker": "BTCUSD", "spread": 30.0, "pip": 1.0, "decimals": 2},
}

PAIR_FLAGS = {
    "GBPUSD": "🇬🇧🇺🇸",
    "AUDUSD": "🇦🇺🇺🇸",
    "NZDUSD": "🇳🇿🇺🇸",
    "USDCAD": "🇺🇸🇨🇦",
    "USDCHF": "🇺🇸🇨🇭",
    "USDJPY": "🇺🇸🇯🇵",
    "EURJPY": "🇪🇺🇯🇵",
    "GBPJPY": "🇬🇧🇯🇵",
    "EURGBP": "🇪🇺🇬🇧",
    "EURAUD": "🇪🇺🇦🇺",
    "GBPAUD": "🇬🇧🇦🇺",
    "BTCUSD": "₿",
    "MSFT": "💻",
    "TSLA": "🚗",
    "PG": "🧴",
    "XOM": "🛢️",
}

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─── Estado global ─────────────────────────────────────────────────────
# Estado de depósitos pendientes { chat_id: True }
awaiting_deposit = {}

bot_state = {
    "last_scan": None,
    "last_signals": [],
    "total_scans": 0,
    "subscribers": 0,
}


# ─── Suscriptores ──────────────────────────────────────────────────────
# Eliminamos funciones locales de JSON y usamos las de Firebase
# ──────────────────────────────────────────────────────────────────────


# ─── Telegram API ──────────────────────────────────────────────────────
def tg_send(chat_id, text, parse_mode="MarkdownV2"):
    """Envía un mensaje a un chat."""
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        return r.json().get("ok", False)
    except Exception as e:
        log.error(f"Error enviando a {chat_id}: {e}")
        return False


def tg_send_inline(chat_id, text, buttons, parse_mode="MarkdownV2"):
    """Envía un mensaje con botones inline (dentro del mensaje)."""
    keyboard = {"inline_keyboard": buttons}
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": keyboard,
            },
            timeout=10,
        )
        return r.json().get("ok", False)
    except Exception as e:
        log.error(f"Error enviando inline a {chat_id}: {e}")
        return False


def tg_send_keyboard(chat_id, text, parse_mode="MarkdownV2"):
    """Envía un mensaje y muestra el teclado principal en la parte inferior."""
    keyboard = {
        "keyboard": get_main_keyboard(),
        "resize_keyboard": True,
        "is_persistent": True,
    }
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": keyboard,
            },
            timeout=10,
        )
        return r.json().get("ok", False)
    except Exception as e:
        log.error(f"Error enviando teclado a {chat_id}: {e}")
        return False


def tg_answer_callback(callback_id):
    """Responde un callback para sacar el 'loading' del botón."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": callback_id},
            timeout=5,
        )
    except:
        pass


def get_main_keyboard():
    """Retorna el teclado persistente que aparece abajo."""
    return [
        [{"text": "💼 Cuenta"}, {"text": "💰 Depositar"}, {"text": "📊 Alertas"}],
        [{"text": "💰 Precios"}, {"text": "📈 Activas"}, {"text": "📜 Historial"}],
        [{"text": "🔍 Forex AI"}, {"text": "🔍 Cripto AI"}],
        [{"text": "ℹ️ Info"}, {"text": "📊 Rendimiento"}],
    ]


def tg_broadcast_with_billing(text, parse_mode="MarkdownV2"):
    """
    Envía un mensaje a TODOS los suscriptores que pueden recibirlo.
    Aplica lógica de billing: trial gratuito o cobra $0.50.
    """
    subs = db_get_subscribers()
    ok_count = 0
    blocked_count = 0

    for chat_id in subs:
        reason, can_receive = db_can_receive_signal(int(chat_id))

        if can_receive:
            if tg_send(int(chat_id), text, parse_mode):
                ok_count += 1
        else:
            blocked_count += 1
            if reason == "no_balance":
                tg_send_keyboard(
                    int(chat_id),
                    "⚠️ *Señal detectada pero no enviada*\n\n"
                    "Tu prueba gratuita finalizó y no tenés saldo suficiente\\.\n"
                    f"Cada señal cuesta *$0\\.50 USD*\\.\n\n"
                    "_Depositá saldo para seguir recibiendo señales usando el botón 💰 Depositar\\._",
                )
            # Si es "no_alerts" no enviamos nada (el usuario las desactivó)
        time.sleep(0.05)

    log.info(f"Broadcast: {ok_count} enviados, {blocked_count} bloqueados")
    return ok_count


def tg_get_updates(offset=None):
    """Obtiene updates pendientes de Telegram."""
    params = {"timeout": 30, "offset": offset}
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params=params,
            timeout=35,
        )
        data = r.json()
        if data.get("ok"):
            return data["result"]
    except requests.exceptions.ReadTimeout:
        log.warning("Telegram API timeout (ReadTimeout). Retrying...")
    except requests.exceptions.ConnectionError:
        log.warning("Telegram API ConnectionError. Retrying...")
    except Exception as e:
        log.error(f"Error getUpdates: {e}")
    return []


# ─── ML Scanner ────────────────────────────────────────────────────────
def download_data(ticker, days=120, interval="1d"):
    """Descarga datos históricos vía MetaTrader 5 o YFinance como fallback."""
    import pandas as pd
    from datetime import timedelta, datetime
    
    mt5_success = False
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            mt5_success = True
    except ImportError:
        pass
        
    if mt5_success:
        try:
            if interval == "1d":
                mt5_tf = mt5.TIMEFRAME_D1
                bars = days
            elif interval == "1h":
                mt5_tf = mt5.TIMEFRAME_H1
                bars = days * 24
            elif interval == "4h":
                mt5_tf = mt5.TIMEFRAME_H4
                bars = days * 6
            elif interval == "15m":
                mt5_tf = mt5.TIMEFRAME_M15
                bars = days * 96
            elif interval == "1m":
                mt5_tf = mt5.TIMEFRAME_M1
                bars = days * 1440
            else:
                mt5_tf = mt5.TIMEFRAME_D1
                bars = days
                
            rates = mt5.copy_rates_from_pos(ticker, mt5_tf, 0, bars)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df = df.rename(
                    columns={
                        "time": "Datetime",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "tick_volume": "Volume",
                    }
                )
                return df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            log.warning(f"Error MT5 download for {ticker}: {e}. Intentando fallback...")

    # Fallback YFinance
    import yfinance as yf
    yf_ticker = ticker
    if ticker == "BTCUSD":
        yf_ticker = "BTC-USD"
    elif len(ticker) == 6 and ticker not in ["MSFT", "TSLA", "PG", "XOM"]:
        yf_ticker = f"{ticker}=X"

    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(
        yf_ticker,
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
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "Datetime"})
    elif "index" in df.columns:
        df = df.rename(columns={"index": "Datetime"})
    return (
        df if all(c in df.columns for c in ["Open", "High", "Low", "Close"]) else None
    )


def resample_to_4h(df_1h):
    """Resamplea datos 1H a 4H."""
    if df_1h is None or df_1h.empty:
        return None
    df = df_1h.copy()
    # Si tiene columna datetime, usarla como index
    for col in ["Datetime", "Date", "index"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        ohlc["Volume"] = "sum"
    df_4h = df.resample("4h").agg(ohlc).dropna()
    return df_4h.reset_index(drop=True)


def run_scan(timeframe="Daily"):
    """Escanea todos los pares para un timeframe específico."""
    model_path = MODELS.get(timeframe, MODELS["Daily"])

    log.info(f"Iniciando scan ML [{timeframe}]...")

    if not os.path.exists(model_path):
        log.warning(f"Modelo {timeframe} no encontrado: {model_path} — skip")
        return []

    model_artifact = joblib.load(model_path)
    feature_cols = model_artifact["feature_columns"]
    threshold = model_artifact["threshold"]
    main_model = model_artifact["model"]

    # Configuración de descarga por timeframe
    if timeframe == "4H":
        yf_interval, yf_days = "1h", 90  # descarga 1h y resamplea
    elif timeframe == "BTC_Daily":
        yf_interval, yf_days = "1d", 200
    else:
        yf_interval, yf_days = "1d", 200

    signals = []

    # Portfolio cluster tracking for this scan
    cluster_counts = {}
    cluster_risk = {}
    MAX_PER_CLUSTER = 2
    MAX_CLUSTER_RISK_PCT = 0.015  # 1.5% max risk per currency

    # Elegir pares según timeframe
    scan_pairs = BTC_PAIRS if timeframe == "BTC_Daily" else PAIRS

    for pair, config in scan_pairs.items():
        log.info(f"Escaneando {pair} [{timeframe}]...")
        df = download_data(config["ticker"], days=yf_days, interval=yf_interval)
        if df is None or len(df) < 60:
            continue

        # Para 4H, resamplear de 1H
        if timeframe == "4H":
            df = resample_to_4h(df)
        if df is None or len(df) < 60:
            continue

        try:
            df = create_features(df)
            df = detect_fractals(df)
        except Exception as e:
            log.error(f"Error features {pair}: {e}")
            continue

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            continue

        last = df.iloc[-1]
        if pd.isna(last[feature_cols]).any():
            continue

        X = df.iloc[[-1]][feature_cols]

        probas = main_model.predict_proba(X)[0]

        signal = "HOLD"
        confidence = 0

        if len(probas) >= 3:
            prob_buy = probas[1]
            prob_sell = probas[2]

            if prob_buy >= threshold and prob_buy > prob_sell:
                signal = "BUY"
                confidence = prob_buy
            elif prob_sell >= threshold and prob_sell > prob_buy:
                signal = "SELL"
                confidence = prob_sell
        else:
            pred = main_model.predict(X)[0]
            conf = probas.max()
            if pred == 1 and conf >= threshold:
                signal = "BUY"
                confidence = conf
            elif pred == 2 and conf >= threshold:
                signal = "SELL"
                confidence = conf

        if signal == "HOLD":
            log.info(f"  {pair}: HOLD")
            continue

        base_curr = pair[:3]
        quote_curr = pair[3:]

        # 1. Check Pair Count Constraints (Max 2 per currency)
        if (
            cluster_counts.get(base_curr, 0) >= MAX_PER_CLUSTER
            or cluster_counts.get(quote_curr, 0) >= MAX_PER_CLUSTER
        ):
            log.info(
                f"  {pair}: {signal} rechazada por límite de clúster (Máx {MAX_PER_CLUSTER} pares para {base_curr}/{quote_curr})"
            )
            continue

        # Sizing dinámico conservador basado en probabilidad calibrada
        base_risk_pct = RISK_PCT  # 0.5%
        max_risk_pct = 0.01  # 1.0% cap para canary por trade
        alpha = 0.5

        # Risk = BaseRisk * ( (Prob - Threshold) / (1 - Threshold) ) * alpha + BaseRisk
        if confidence > threshold:
            scaled_risk = (
                base_risk_pct * ((confidence - threshold) / (1.0 - threshold)) * alpha
                + base_risk_pct
            )
        else:
            scaled_risk = base_risk_pct

        adjusted_risk_pct = min(scaled_risk, max_risk_pct)

        # 2. Check Currency Risk Constraints (Max 1.5% exposure per currency)
        if (
            cluster_risk.get(base_curr, 0) + adjusted_risk_pct > MAX_CLUSTER_RISK_PCT
            or cluster_risk.get(quote_curr, 0) + adjusted_risk_pct
            > MAX_CLUSTER_RISK_PCT
        ):
            log.info(
                f"  {pair}: {signal} rechazada por límite de riesgo correlacionado (Expondría {base_curr}/{quote_curr} a > 1.5%)"
            )
            continue

        # Validated. Update cluster tracking.
        cluster_counts[base_curr] = cluster_counts.get(base_curr, 0) + 1
        cluster_counts[quote_curr] = cluster_counts.get(quote_curr, 0) + 1
        cluster_risk[base_curr] = cluster_risk.get(base_curr, 0) + adjusted_risk_pct
        cluster_risk[quote_curr] = cluster_risk.get(quote_curr, 0) + adjusted_risk_pct

        close = float(last["Close"])
        atr = float(last["ATR"])
        pip = config["pip"]
        spread_cost = config["spread"] * pip
        sl_dist = atr
        tp_dist = atr * 1.5

        if signal == "BUY":
            entry = close + spread_cost
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            entry = close - spread_cost
            sl = entry + sl_dist
            tp = entry - tp_dist

        sl_pips = sl_dist / pip
        tp_pips = tp_dist / pip

        risk_usd = DEFAULT_BALANCE * adjusted_risk_pct
        volume = round(risk_usd / (sl_pips * 10), 2)

        signals.append(
            {
                "pair": pair,
                "signal": signal,
                "confidence": confidence,
                "entry": round(entry, config["decimals"]),
                "sl": round(sl, config["decimals"]),
                "tp": round(tp, config["decimals"]),
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "atr_pips": atr / pip,
                "risk_usd": risk_usd,
                "volume": volume,
                "timeframe": timeframe,
            }
        )
        log.info(f"  {pair}: {signal} ({confidence:.1%})")

    bot_state["last_scan"] = datetime.now().isoformat()
    bot_state["last_signals"] = signals
    bot_state["total_scans"] += 1
    log.info(
        f"Scan completo: {len(signals)} señal(es). Riesgo USD total: {cluster_risk.get('USD', 0):.2%}"
    )
    return signals


def build_signal_message(s):
    """Construye mensaje de señal para Telegram."""
    flag = PAIR_FLAGS.get(s["pair"], "💱")
    emoji = "🟢 BUY" if s["signal"] == "BUY" else "🔴 SELL"
    tf = s.get("timeframe", "Daily")
    tf_emoji = TF_EMOJIS.get(tf, "📅")

    # Escapar caracteres especiales para MarkdownV2
    def esc(v):
        return str(v).replace(".", "\\.").replace("-", "\\-").replace("+", "\\+")

    text = (
        f"⚡ *SEÑAL ML* {tf_emoji} *{tf}* — {esc(datetime.now().strftime('%H:%M %d/%m/%Y'))}\n\n"
        f"{flag} *{s['pair']}* — {emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 *Entry:*    `{esc(s['entry'])}`\n"
        f"🛑 *SL:*       `{esc(s['sl'])}`  \\({s['sl_pips']:.0f} pips\\)\n"
        f"🎯 *TP:*       `{esc(s['tp'])}`  \\({s['tp_pips']:.0f} pips\\)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *ATR:*      {s['atr_pips']:.0f} pips  \\|  *R:R* 1:1\\.5\n"
        f"🤖 *Confianza:* {esc(str(round(s['confidence'] * 100)) + '%')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Volumen ref:*   {esc(round(s['volume'], 2))} lotes\n"
        f"⚠️  *Riesgo ref:*   ${s['risk_usd']:.0f}  \\({esc(round((s['risk_usd']/10000)*100, 1))}% de $10k\\)\n"
        f"✅ *Si TP:*    \\+${s['risk_usd'] * 1.5:.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Ajustá lotes según tu capital\\._"
    )
    return text


def scan_and_broadcast(timeframe="Daily"):
    """Escanea y transmite señales con sistema de billing."""
    subs = db_get_subscribers()
    n_subs = len(subs)
    log.info(f"Iniciando scan + broadcast [{timeframe}] ({n_subs} suscriptores)")

    signals = run_scan(timeframe)

    if not signals:
        log.info(f"Sin señales [{timeframe}] — no se envía broadcast")
        return

    for s in signals:
        db_save_signal(s["pair"], s)
        msg = build_signal_message(s)
        sent = tg_broadcast_with_billing(msg)
        log.info(f"[{timeframe}] {s['pair']} {s['signal']} enviada a {sent} subs")


# ─── Command handler ───────────────────────────────────────────────────
def handle_command(chat_id, text, first_name, username):
    """Procesa comandos de Telegram."""
    raw = text.strip() if text else ""
    cmd = raw.lower().split()[0] if raw else ""

    # ─── Botones del teclado (Reply Keyboard) ──────────────────────
    if "Cuenta" in raw:
        handle_action(chat_id, "cmd_cuenta")
        return
    elif "Depositar" in raw:
        handle_action(chat_id, "cmd_depositar")
        return
    elif "Alertas" in raw:
        handle_action(chat_id, "cmd_toggle_alerts")
        return
    elif "Precios" in raw or cmd == "/price":
        handle_action(chat_id, "cmd_price")
        return
    elif "Activas" in raw:
        handle_action(chat_id, "cmd_active")
        return
    elif "Historial" in raw:
        handle_action(chat_id, "cmd_history")
        return
    elif "Info" in raw:
        handle_action(chat_id, "cmd_info")
        return
    elif "Rendimiento" in raw:
        handle_action(chat_id, "cmd_performance")
        return
    elif "Forex AI" in raw:
        handle_action(chat_id, "cmd_scan_forex")
        return
    elif "Cripto AI" in raw:
        handle_action(chat_id, "cmd_scan_crypto")
        return

    # ─── Comandos clásicos ─────────────────────────────────────────
    if cmd in ["/start", "start"]:
        db_init_user_account(chat_id, first_name, username)
        n = len(db_get_subscribers())
        is_trial = db_is_trial_active(chat_id)
        trial_msg = "🆓 *Prueba gratuita:* 15 días activada" if is_trial else ""
        msg = (
            f"✅ *¡Bienvenido\\, {first_name}\\!*\n\n"
            f"Ahora recibirás señales de trading con IA\\.\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *Modelo:* GradientBoosting v9\n"
            f"📊 *Pares:* 27 instrumentos\n"
            f"⏰ *Scan:* Diario al cierre NY\n"
            f"{trial_msg}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Suscriptores activos: *{n}*\n\n"
            f"_Usá los botones de abajo para navegar ⬇️_"
        )
        tg_send_keyboard(chat_id, msg)
        log.info(f"/start de {first_name} ({chat_id})")

    elif cmd in ["/stop", "stop"]:
        db_remove_subscriber(chat_id)
        tg_send(
            chat_id,
            "👋 *Desuscripto\\.*\n\nYa no recibirás más señales\\.\nUsá /start para volver\\.",
        )

    elif cmd in ["/signal", "signal"]:
        tg_send(chat_id, "🔍 *Escaneando mercados\\.\\.\\.*")
        signals = run_scan()
        if signals:
            for s in signals:
                tg_send(chat_id, build_signal_message(s))
        else:
            tg_send(
                chat_id,
                "⏸ *Sin señales ahora*\\.\n\nEl modelo está en HOLD para todos los pares\\.",
            )

    elif cmd in ["/menu", "menu"]:
        tg_send_keyboard(
            chat_id, "🤖 *Menú Principal*\n\n_Usá los botones de abajo ⬇️_"
        )

    else:
        tg_send_keyboard(chat_id, "❓ *Usá los botones del teclado o /start*")


def handle_action(chat_id, action):
    """Procesa las acciones de los botones inline."""

    if action == "cmd_cuenta":
        user = db_get_user_account(chat_id)
        if not user:
            tg_send(chat_id, "⚠️ Usá /start primero para crear tu cuenta\\.")
            return

        joined = user.get("joined", "Desconocido")
        balance = user.get("balance", 0.0)
        alerts = user.get("alerts_enabled", True)
        is_trial = db_is_trial_active(chat_id)
        total_signals = user.get("total_signals_received", 0)
        total_spent = user.get("total_spent", 0.0)

        if is_trial:
            trial_end = user.get("trial_end", "")[:10]
            status = f"🆓 Prueba Gratuita \\(hasta {trial_end}\\)"
        elif balance > 0:
            status = "💳 Usuario Pago"
        else:
            status = "⚠️ Sin saldo"

        alerts_emoji = "🟢 Activadas" if alerts else "🔴 Desactivadas"
        toggle_text = "🔴 Desactivar Alertas" if alerts else "🟢 Activar Alertas"

        msg = (
            f"💼 *Tu Cuenta*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 *Nombre:* {user.get('first_name', 'N/A')}\n"
            f"💵 *Saldo:* ${balance:.2f} USD\n"
            f"📊 *Estado:* {status}\n"
            f"🔔 *Alertas:* {alerts_emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 *Señales recibidas:* {total_signals}\n"
            f"💸 *Total gastado:* ${total_spent:.2f}\n"
            f"📅 *Miembro desde:* {joined[:10] if len(joined) > 10 else joined}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Costo por señal: $0\\.50 USD_\n"
            f"_(Usa los botones del teclado para recargar o desactivar alertas)_"
        )
        tg_send_keyboard(chat_id, msg.replace(".", "\\."))

    elif action == "cmd_toggle_alerts":
        new_state = db_toggle_alerts(chat_id)
        if new_state is None:
            tg_send(chat_id, "⚠️ Error\\. Usá /start primero\\.")
            return
        if new_state:
            tg_send_keyboard(
                chat_id,
                "🟢 *Alertas ACTIVADAS*\n\nVolverás a recibir señales de trading\\.",
            )
        else:
            tg_send_keyboard(
                chat_id,
                "🔴 *Alertas DESACTIVADAS*\n\nNo recibirás señales hasta que las reactives\\.",
            )

    elif action == "cmd_depositar":
        tg_send_keyboard(
            chat_id,
            "💰 *Depositar Saldo*\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "Elegí tu método de pago y avisa al @admin:\n\n"
            "🟡 *PayPal* \u2014 Instantáneo (pagos@tradingbot.com)\n"
            "🟢 *Cripto USDT \\(TRC20\\)* \u2014 TXyz1234567890abcdef\n"
            "🔵 *Transferencia Bancaria* \u2014 CBU: 0000003100012345678901\n\n"
            "_Cada señal cuesta $0\\.50 USD_",
        )

    elif action == "dep_paypal":
        awaiting_deposit[chat_id] = "paypal"
        awaiting_deposit[chat_id] = "paypal"
        tg_send_keyboard(
            chat_id,
            "🟡 *Depósito vía PayPal*\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "Enviá tu pago a:\n"
            "📧 `pagos@tradingbot\\.com`\n\n"
            "Luego envía un mensaje con el *monto* enviado\n"
            "Ejemplo: `10`\n\n"
            "_Tu saldo se actualiza al instante\\._",
        )

    elif action == "dep_crypto":
        awaiting_deposit[chat_id] = "crypto"
        awaiting_deposit[chat_id] = "crypto"
        tg_send_keyboard(
            chat_id,
            "🟢 *Depósito vía Cripto*\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "Enviá USDT \\(Red TRC20\\) a:\n"
            "📎 `TXyz1234567890abcdef`\n\n"
            "Luego envía un mensaje con el *monto* enviado\n"
            "Ejemplo: `25`\n\n"
            "_Sin comisión \\| Confirmación en 2 min\\._",
        )

    elif action == "dep_bank":
        awaiting_deposit[chat_id] = "bank"
        awaiting_deposit[chat_id] = "bank"
        tg_send_keyboard(
            chat_id,
            "🔵 *Depósito vía Transferencia*\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "🏦 *Banco:* Ejemplo Bank\n"
            "🔢 *CBU:* `0000003100012345678901`\n"
            "👤 *Titular:* Trading Bot SRL\n\n"
            "Luego envía un mensaje con el *monto* transferido\n"
            "Ejemplo: `50`\n\n"
            "_Acreditación en 1\\-2 días hábiles\\._",
        )

    elif action == "cmd_cancel_deposit":
        awaiting_deposit.pop(chat_id, None)
        awaiting_deposit.pop(chat_id, None)
        tg_send_keyboard(chat_id, "❌ *Depósito cancelado\\.*")

    elif action == "cmd_info":
        msg = (
            "ℹ️ *Info del Bot*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🤖 *Motor:* GradientBoosting v8\n"
            "🧠 *Entrenado con:* 27 pares\n"
            "📊 *Datos:* 2010\\-2026 \\(16 años\\)\n"
            "🎯 *Estrategia:* Fractal Breakout\n"
            "📈 *Señales:* BUY / SELL / HOLD\n"
            "⏰ *Frecuencia:* Diaria \\(cierre NY\\)\n"
            "🔒 *Riesgo:* 0\\.5% por operación\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🏗 *Stack:* Python \\+ Firebase \\+ Railway\n"
            "📡 *Uptime:* 24/7 en la nube\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "_Desarrollado con Machine Learning_"
        )
        tg_send_keyboard(chat_id, msg)

    elif action == "cmd_performance":
        report = {}
        if db:
            doc = db.collection("reports").document("blind_test_v8").get()
            if doc.exists:
                report = doc.to_dict()

        if not report:
            tg_send_keyboard(
                chat_id,
                "📊 *Rendimiento Out-of-Sample*\n━━━━━━━━━━━━━━━━━━━━\n\n_Reporte no generado aún\\._\n\nEjecutá `blind_backtest.py` para generar métricas de ingeniería\\.",
            )
            return

        msg = (
            f"📊 *Reporte de Ingeniería (OOS)*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 *Periodo:* 2024 \\- 2026 \\(Ciego\\)\n"
            f"🔢 *Total Trades:* {report.get('total_trades', 0)}\n"
            f"✅ *Win Rate:* {report.get('avg_win_rate', 0):.1f}%\n"
            f"📈 *Profit Factor:* {report.get('avg_profit_factor', 0):.2f}\n"
            f"🎯 *Esperanza:* {report.get('mathematical_expectancy', 0):.2f} pips/trade\n"
            f"📉 *Max Drawdown:* {report.get('max_drawdown_avg', 0):.0f} pips\n"
            f"⚖️ *Ratio Sharpe:* {report.get('avg_sharpe', 0):.2f}\n"
            f"🛡 *Slippage:* \\+1\\.0 pip \\(Incluido\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Validación ultra\\-robusta sin fugas de datos_"
        )
        tg_send_keyboard(chat_id, msg.replace(".", "\\."))

    elif action == "cmd_history":
        trades = []
        if db:
            docs = (
                db.collection("backtest_history").order_by("order").limit(10).stream()
            )
            trades = [doc.to_dict() for doc in docs]

        if not trades:
            tg_send_keyboard(
                chat_id,
                "📜 *Historial de Operaciones*\n━━━━━━━━━━━━━━━━━━━━\n\n_No hay operaciones registradas aún\\._\n\n_El historial se actualiza con cada backtest\\._",
            )
            return

        wins = sum(1 for t in trades if t.get("result") == "TP")
        losses = sum(1 for t in trades if t.get("result") == "SL")
        total = wins + losses
        wr = wins / max(1, total) * 100

        msg = f"📜 *Últimas Operaciones \\(Backtest\\)*\n━━━━━━━━━━━━━━━━━━━━\n"

        for t in trades[:10]:
            result = t.get("result", "?")
            if result == "TP":
                emoji = "✅"
            elif result == "SL":
                emoji = "❌"
            else:
                emoji = "⏳"

            sig = "🟢" if t.get("signal") == "BUY" else "🔴"
            pair = t.get("pair", "???")
            entry = str(t.get("entry", "?"))[:8]
            conf = t.get("confidence", 0)

            msg += f"{emoji} {sig} *{pair}* → `{entry}` \\({conf*100:.0f}%\\)\n"

        msg += (
            f"\n━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *Win Rate:* {wr:.0f}% \\({wins}W / {losses}L\\)\n"
            f"_Basado en backtest de últimos 6 meses_"
        )

        tg_send_keyboard(chat_id, msg.replace(".", "\\."))

    elif action == "cmd_vip":
        msg = (
            "💎 *Planes Premium*\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "🆓 *FREE*\n"
            "• 1 señal diaria\n"
            "• 7 pares principales\n"
            "• Alertas básicas\n\n"
            "⭐ *PRO \\— $9\\.99/mes*\n"
            "• Señales ilimitadas\n"
            "• 27 pares \\+ Oro\n"
            "• Alertas SL/TP en vivo\n"
            "• Historial completo\n\n"
            "👑 *ELITE \\— $24\\.99/mes*\n"
            "• Todo de PRO\n"
            "• Señales intradía \\(1H\\)\n"
            "• Soporte prioritario\n"
            "• Acceso a la API\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "_Contactar @admin para upgrade_"
        )
        tg_send_keyboard(chat_id, msg)

    elif action == "cmd_price":
        tg_send(chat_id, "💰 *Consultando precios actuales\\.\\.\\.*")
        msg = "📊 *Precios en Vivo*\n━━━━━━━━━━━━━━━━━━━━\n"
        
        mt5_success = False
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                mt5_success = True
        except ImportError:
            pass

        import yfinance as yf
        import pandas as pd
        
        for pair, config in PAIRS.items():
            try:
                flag = PAIR_FLAGS.get(pair, "💱")
                last_price = None
                
                if mt5_success:
                    tick = mt5.symbol_info_tick(config["ticker"])
                    if tick is not None:
                        last_price = tick.bid
                
                if last_price is None:
                    yf_ticker = config["ticker"]
                    if len(yf_ticker) == 6 and yf_ticker not in ["MSFT", "TSLA", "PG", "XOM"]:
                        yf_ticker = f"{yf_ticker}=X"
                        
                    df = yf.download(yf_ticker, period="1d", interval="1m", progress=False)
                    if df is not None and not df.empty:
                        last_price = df["Close"].iloc[-1]
                        if isinstance(last_price, pd.Series):
                            last_price = last_price.iloc[0]
                
                if last_price is not None:
                    msg += f"{flag} *{pair}:* `{float(last_price):.{config['decimals']}f}`\n"
            except Exception as e:
                log.error(f"Error price {pair}: {e}")
                
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        if mt5_success:
            msg += "_Datos vía MetaTrader 5_"
        else:
            msg += "_Datos vía Yahoo Finance_"
        tg_send_keyboard(chat_id, msg.replace(".", "\\."))

    elif action == "cmd_active":
        active = db_get_active_signals()
        if not active:
            tg_send_keyboard(
                chat_id, "📭 *No hay operaciones abiertas en este momento\\.*"
            )
            return
        msg = "📈 *Operaciones en Seguimiento*\n━━━━━━━━━━━━━━━━━━━━\n"
        for pair, s in active.items():
            emoji = "🟢 BUY" if s["signal"] == "BUY" else "🔴 SELL"
            msg += f"*{pair}* — {emoji}\n📍 En: `{s['entry']}`\n🛑 SL: `{s['sl']}` | 🎯 TP: `{s['tp']}`\n\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n_Monitoreando cada 5 min_"
        tg_send_keyboard(chat_id, msg.replace(".", "\\."))

    elif action == "cmd_scan_forex":
        tg_send_keyboard(chat_id, "🔍 *Escaneando mercado Forex con IA\\.\\.\\.*")
        signals = run_scan(timeframe="Manual")
        if signals:
            for s in signals:
                tg_send(chat_id, build_signal_message(s))
            tg_send_keyboard(
                chat_id,
                f"✅ *Escaneo finalizado\\.* {len(signals)} señales encontradas\\.",
            )
        else:
            tg_send_keyboard(
                chat_id,
                "⏸ *Sin señales en este momento*\\.\n\nEl modelo está en HOLD para todos los pares\\.",
            )

    elif action == "cmd_scan_crypto":
        tg_send_keyboard(
            chat_id, "🔍 *Escaneando mercado Cripto \\(Pump & Dump\\) con IA\\.\\.\\.*"
        )
        try:
            import bot_shitcoins

            bot_shitcoins.run_scan_job()
            tg_send_keyboard(
                chat_id,
                "✅ *Escaneo Cripto finalizado\\.* Si hubo señales, fueron enviadas\\.",
            )
        except Exception as e:
            tg_send_keyboard(
                chat_id, f"❌ *Error en scanner:* `{str(e)}`".replace(".", "\\.")
            )

    elif action == "cmd_menu":
        tg_send_keyboard(
            chat_id, "🤖 *Menú Principal*\\n\\n_Usá los botones de abajo ⬇️_"
        )


# ─── Main loop (long polling) ──────────────────────────────────────────
# ─── Monitor Loop ──────────────────────────────────────────────────────
def run_monitor_loop():
    """
    Rastrea señales activas contra precios en vivo.
    Si toca SL o TP, avisa a todos y cierra la señal en DB.
    """
    log.info("Iniciado hilo de monitoreo de señales...")
    while True:
        try:
            active = db_get_active_signals()
            if not active:
                time.sleep(300)
                continue

            for pair, s in active.items():
                config = PAIRS.get(pair)
                if not config:
                    continue

                df = download_data(config["ticker"], days=3)
                if df is None or df.empty:
                    continue

                curr = df.iloc[-1]["Close"]
                high = df.iloc[-1]["High"]
                low = df.iloc[-1]["Low"]

                hit = None
                if s["signal"] == "BUY":
                    if high >= s["tp"]:
                        hit = "TP"
                    elif low <= s["sl"]:
                        hit = "SL"
                else:  # SELL
                    if low <= s["tp"]:
                        hit = "TP"
                    elif high >= s["sl"]:
                        hit = "SL"

                if hit:
                    # Notificar cierre
                    emoji = "🎯 TP" if hit == "TP" else "🛑 SL"
                    flag = PAIR_FLAGS.get(pair, "💱")
                    entry_str = str(s["entry"]).replace(".", "\\.")
                    curr_str = str(curr).replace(".", "\\.")
                    msg = (
                        f"🏁 *OPERACIÓN CERRADA* — {pair}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"{flag} *Cierre:* {emoji}\n"
                        f"📍 *Entrada:* `{entry_str}`\n"
                        f"🏁 *Salida:*  `{curr_str}`\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"{'✅ PROFIT' if hit == 'TP' else '❌ LOSS'}\n"
                    )
                    tg_broadcast_with_billing(msg)
                    db_close_signal(pair, hit, curr)

            time.sleep(300)  # Revisa cada 5 min
        except Exception as e:
            log.error(f"Error en monitor loop: {e}")
            time.sleep(60)


def run_polling():
    """Loop principal: escucha mensajes de Telegram con long polling."""
    log.info("Bot iniciado — escuchando mensajes...")
    offset = None

    while True:
        try:
            updates = tg_get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1

                # Manejar callback queries (botones inline)
                cb = update.get("callback_query")
                if cb:
                    cb_chat_id = cb["message"]["chat"]["id"]
                    cb_data = cb.get("data", "")
                    tg_answer_callback(cb["id"])
                    handle_action(cb_chat_id, cb_data)
                    continue

                msg = update.get("message")
                if not msg:
                    continue
                chat_id = msg["chat"]["id"]
                text = msg.get("text", "").strip()
                first_name = msg["chat"].get("first_name", "")
                username = msg["chat"].get("username", "")

                # Manejar depósitos pendientes
                if chat_id in awaiting_deposit and text:
                    try:
                        amount = float(text.replace(",", ".").replace("$", ""))
                        if amount > 0:
                            new_balance = db_deposit(chat_id, amount)
                            if new_balance is not None:
                                tg_send_keyboard(
                                    chat_id,
                                    f"✅ *Depósito exitoso*\n\n"
                                    f"💵 Monto: *${amount:.2f} USD*\n"
                                    f"💰 Nuevo saldo: *${new_balance:.2f} USD*\n\n"
                                    f"_Tenés para {int(new_balance / SIGNAL_COST)} señales\\._".replace(
                                        ".", "\\."
                                    ),
                                )
                            else:
                                tg_send(
                                    chat_id,
                                    "❌ Error al depositar\\. Intentá de nuevo\\.",
                                )
                            awaiting_deposit.pop(chat_id, None)
                            continue
                        else:
                            tg_send(chat_id, "⚠️ El monto debe ser mayor a 0\\.")
                            continue
                    except ValueError:
                        # No es un número, procesar como comando normal
                        awaiting_deposit.pop(chat_id, None)

                handle_command(chat_id, text, first_name, username)
        except KeyboardInterrupt:
            log.info("Bot detenido.")
            break
        except Exception as e:
            log.error(f"Error en polling: {e}")
            time.sleep(5)


def run_scheduler():
    """Programa scans multi-timeframe."""
    scan_time = f"{SCAN_HOUR:02d}:00"

    # Daily: 1x al día al cierre NY
    schedule.every().day.at(scan_time).do(scan_and_broadcast, "Daily")
    log.info(f"📅 Daily scan programado a las {scan_time}")

    # 4H: cada 4 horas (si existe el modelo)
    if os.path.exists(MODELS["4H"]):
        schedule.every(4).hours.do(scan_and_broadcast, "4H")
        log.info("⏳ 4H scan programado cada 4 horas")

    # 1H (Crypto / Shitcoins Parabolic Reversal)
    # Analiza futuros de Binance cada hora en el minuto :01
    try:
        from bot_shitcoins import run_scan_job as run_shitcoin_scan

        schedule.every().hour.at(":01").do(run_shitcoin_scan)
        log.info("🚀 Shitcoins 1H scan programado cada hora en el minuto :01")
    except Exception as e:
        log.warning(f"No se pudo cargar el módulo de shitcoins: {e}")

    # BTC Daily: 1x al día junto con forex daily
    if os.path.exists(MODELS["BTC_Daily"]):
        schedule.every().day.at(scan_time).do(scan_and_broadcast, "BTC_Daily")
        log.info("₿ BTC Daily scan programado a las " + scan_time)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-now", action="store_true", help="Hacer scan y broadcast ahora"
    )
    args = parser.parse_args()

    if args.scan_now:
        scan_and_broadcast()
    else:
        # 1. Scheduler (Signals)
        t_scheduler = threading.Thread(target=run_scheduler, daemon=True)
        t_scheduler.start()

        # 2. Monitor (SL/TP)
        t_monitor = threading.Thread(target=run_monitor_loop, daemon=True)
        t_monitor.start()

        # 3. Polling (Commands)
        run_polling()
