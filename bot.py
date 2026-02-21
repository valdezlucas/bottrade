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
import sys
import os
import json
import logging
import asyncio
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
from fractals import detect_fractals
from firebase_manager import (
    db_add_subscriber, db_remove_subscriber, db_get_subscribers,
    db_save_signal, db_get_active_signals, db_close_signal,
    db
)

# ─── CONFIG ────────────────────────────────────────────────────────────
BOT_TOKEN = "5967657374:AAHX9XuJBmRxIYWn9AgcsCBtTK5mr3O2yTY"
MODEL_PATH = "model_multi.joblib"
SELL_MODEL_PATH = "model_multi_sell.joblib"
SUBSCRIBERS_FILE = "subscribers.json"
SCAN_HOUR = 22        # Hora en que escanea (22:00 UTC-3 = cierre vela diaria NY)
RISK_PCT = 0.005      # 0.5% riesgo por trade (para calcular lotes)
DEFAULT_BALANCE = 10000  # Balance de referencia para calcular lotes
# ──────────────────────────────────────────────────────────────────────

# Pares a escanear
PAIRS = {
    "EURUSD": {"ticker": "EURUSD=X", "spread": 1.0, "pip": 0.0001, "decimals": 5},
    "GBPUSD": {"ticker": "GBPUSD=X", "spread": 1.2, "pip": 0.0001, "decimals": 5},
    "NZDUSD": {"ticker": "NZDUSD=X", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "AUDUSD": {"ticker": "AUDUSD=X", "spread": 1.2, "pip": 0.0001, "decimals": 5},
    "USDCAD": {"ticker": "USDCAD=X", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "USDCHF": {"ticker": "USDCHF=X", "spread": 1.5, "pip": 0.0001, "decimals": 5},
    "EURGBP": {"ticker": "EURGBP=X", "spread": 1.5, "pip": 0.0001, "decimals": 5},
}

PAIR_FLAGS = {
    "EURUSD": "🇪🇺🇺🇸", "GBPUSD": "🇬🇧🇺🇸", "AUDUSD": "🇦🇺🇺🇸",
    "NZDUSD": "🇳🇿🇺🇸", "USDCAD": "🇺🇸🇨🇦", "USDCHF": "🇺🇸🇨🇭",
    "EURGBP": "🇪🇺🇬🇧",
}

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─── Estado global ─────────────────────────────────────────────────────
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


def tg_send_buttons(chat_id, text, buttons, parse_mode="MarkdownV2"):
    """Envía un mensaje con botones inline."""
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
        log.error(f"Error enviando botones a {chat_id}: {e}")
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


def get_main_buttons():
    """Retorna la grilla de botones principales."""
    return [
        [{"text": "📊 Cuenta", "callback_data": "cmd_cuenta"}, {"text": "ℹ️ Info del Bot", "callback_data": "cmd_info"}],
        [{"text": "📜 Últimas Señales", "callback_data": "cmd_history"}, {"text": "💎 Planes VIP", "callback_data": "cmd_vip"}],
        [{"text": "💰 Precios", "callback_data": "cmd_price"}, {"text": "📈 Activas", "callback_data": "cmd_active"}],
    ]


def tg_broadcast(text, parse_mode="MarkdownV2"):
    """Envía un mensaje a TODOS los suscriptores."""
    subs = db_get_subscribers()
    ok_count = 0

    for chat_id in subs:
        if tg_send(int(chat_id), text, parse_mode):
            ok_count += 1
        time.sleep(0.05)

    log.info(f"Broadcast enviado: {ok_count}/{len(subs)} OK")
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
    except Exception as e:
        log.error(f"Error getUpdates: {e}")
    return []


# ─── ML Scanner ────────────────────────────────────────────────────────
def download_data(ticker, days=120):
    """Descarga datos de Yahoo Finance."""
    import yfinance as yf
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), interval="1d", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index(drop=True)
    return df if all(c in df.columns for c in ["Open", "High", "Low", "Close"]) else None


def run_scan():
    """Escanea todos los pares y retorna lista de señales."""
    log.info("Iniciando scan ML...")

    if not os.path.exists(MODEL_PATH):
        log.error(f"Modelo no encontrado: {MODEL_PATH}")
        return []

    model_artifact = joblib.load(MODEL_PATH)
    sell_artifact = joblib.load(SELL_MODEL_PATH)
    feature_cols = model_artifact["feature_columns"]
    threshold = model_artifact["threshold"]
    main_model = model_artifact["model"]
    sell_model = sell_artifact["model"]

    signals = []

    for pair, config in PAIRS.items():
        log.info(f"Escaneando {pair}...")
        df = download_data(config["ticker"])
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

        X = last[feature_cols].values.reshape(1, -1)

        main_proba = main_model.predict_proba(X)[0]
        main_pred = main_model.predict(X)[0]
        main_conf = main_proba.max()

        sell_proba = sell_model.predict_proba(X)[0]
        sell_pred = sell_model.predict(X)[0]
        sell_conf = sell_proba[1] if len(sell_proba) > 1 else 0

        signal = "HOLD"
        confidence = 0

        if main_pred == 1 and main_conf >= threshold:
            signal, confidence = "BUY", main_conf
        elif sell_pred == 1 and sell_conf >= threshold:
            signal, confidence = "SELL", sell_conf
        elif main_pred == 2 and main_conf >= threshold:
            signal, confidence = "SELL", main_conf

        if signal == "HOLD":
            log.info(f"  {pair}: HOLD")
            continue

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
        risk_usd = DEFAULT_BALANCE * RISK_PCT
        volume = round(risk_usd / (sl_pips * 10), 2)

        signals.append({
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
        })
        log.info(f"  {pair}: {signal} ({confidence:.1%})")

    bot_state["last_scan"] = datetime.now().isoformat()
    bot_state["last_signals"] = signals
    bot_state["total_scans"] += 1
    log.info(f"Scan completo: {len(signals)} señal(es)")
    return signals


def build_signal_message(s):
    """Construye mensaje de señal para Telegram."""
    flag = PAIR_FLAGS.get(s["pair"], "💱")
    emoji = "🟢 BUY" if s["signal"] == "BUY" else "🔴 SELL"

    # Escapar caracteres especiales para MarkdownV2
    def esc(v):
        return str(v).replace(".", "\\.").replace("-", "\\-").replace("+", "\\+")

    text = (
        f"⚡ *SEÑAL ML* — {datetime.now().strftime('%H:%M %d/%m/%Y')}\n\n"
        f"{flag} *{s['pair']}* — {emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 *Entry:*    `{esc(s['entry'])}`\n"
        f"🛑 *SL:*       `{esc(s['sl'])}`  \\({s['sl_pips']:.0f} pips\\)\n"
        f"🎯 *TP:*       `{esc(s['tp'])}`  \\({s['tp_pips']:.0f} pips\\)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *ATR:*      {s['atr_pips']:.0f} pips  \\|  *R:R* 1:1\\.5\n"
        f"🤖 *Confianza:* {s['confidence']:.0%}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Volumen ref:*   {s['volume']:.2f} lotes\n"
        f"⚠️  *Riesgo ref:*   ${s['risk_usd']:.0f}  \\(0\\.5% de $10k\\)\n"
        f"✅ *Si TP:*    \\+${s['risk_usd'] * 1.5:.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Ajustá lotes según tu capital\\._"
    )
    return text


def scan_and_broadcast():
    """Escanea y transmite señales a todos los suscriptores."""
    subs = db_get_subscribers()
    n_subs = len(subs)
    log.info(f"Iniciando scan + broadcast ({n_subs} suscriptores)")

    signals = run_scan()

    if not signals:
        log.info("Sin señales — no se envía broadcast")
        return

    for s in signals:
        # Guardar en Firebase para monitoreo
        db_save_signal(s["pair"], s)
        
        msg = build_signal_message(s)
        sent = tg_broadcast(msg)
        log.info(f"Señal {s['pair']} {s['signal']} enviada a {sent} suscriptores")


# ─── Command handler ───────────────────────────────────────────────────
def handle_command(chat_id, text, first_name, username):
    """Procesa comandos de Telegram."""
    cmd = text.strip().lower().split()[0] if text else ""

    if cmd in ["/start", "start"]:
        db_add_subscriber(chat_id, first_name, username)
        n = len(db_get_subscribers())
        msg = (
            f"✅ *¡Bienvenido\\, {first_name}\\!*\n\n"
            f"Ahora recibirás señales de trading con IA\\.\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *Modelo:* GradientBoosting v7\n"
            f"📊 *Pares:* 27 instrumentos\n"
            f"⏰ *Scan:* Diario al cierre NY\n"
            f"💡 *Confianza mínima:* 60%\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Suscriptores activos: *{n}*\n\n"
            f"_Usá los botones de abajo para navegar ⬇️_"
        )
        tg_send_buttons(chat_id, msg, get_main_buttons())
        log.info(f"/start de {first_name} ({chat_id})")

    elif cmd in ["/stop", "stop"]:
        db_remove_subscriber(chat_id)
        tg_send(chat_id,
                "👋 *Desuscripto\\.*\n\nYa no recibirás más señales\\.\nUsá /start para volver\\.",
                )

    elif cmd in ["/status", "status"]:
        subs = db_get_subscribers()
        last = bot_state.get("last_scan") or "Nunca"
        n_signals = len(bot_state.get("last_signals", []))
        total = bot_state.get("total_scans", 0)
        msg = (
            f"📊 *Estado del Bot*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👥 Suscriptores: *{len(subs)}*\n"
            f"🔍 Último scan: `{last}`\n"
            f"📡 Señales último scan: *{n_signals}*\n"
            f"🔢 Scans totales: *{total}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Bot corriendo 24/7 en Railway_"
        )
        tg_send(chat_id, msg)

    elif cmd in ["/signal", "signal"]:
        tg_send(chat_id, "🔍 *Escaneando mercados\\.\\.\\.*")
        # Scan solo para quien preguntó
        signals = run_scan()
        if signals:
            for s in signals:
                tg_send(chat_id, build_signal_message(s))
        else:
            tg_send(chat_id,
                    "⏸ *Sin señales ahora*\\.\n\nEl modelo está en HOLD para todos los pares\\.")

    elif cmd in ["/active", "active"]:
        active = db_get_active_signals()
        if not active:
            tg_send(chat_id, "📭 *No hay operaciones abiertas en este momento\\.*")
            return
            
        msg = "📈 *Operaciones en Seguimiento*\n━━━━━━━━━━━━━━━━━━━━\n"
        for pair, s in active.items():
            emoji = "🟢 BUY" if s["signal"] == "BUY" else "🔴 SELL"
            msg += f"*{pair}* — {emoji}\n📍 En: `{s['entry']}`\n🛑 SL: `{s['sl']}` | 🎯 TP: `{s['tp']}`\n\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n_Monitoreando cada 5 min_"
        tg_send(chat_id, msg.replace(".", "\\."))

    elif cmd in ["/price", "price"]:
        handle_callback_query_action(chat_id, "cmd_price")

    elif cmd in ["/history", "history"]:
        handle_callback_query_action(chat_id, "cmd_history")

    elif cmd in ["/menu", "menu"]:
        tg_send_buttons(chat_id, "🤖 *Menú Principal*", get_main_buttons())

    else:
        tg_send_buttons(chat_id,
                "❓ *Usá los botones o los comandos:*\n/start /stop /status /signal /price /history /menu",
                get_main_buttons())


def handle_callback_query_action(chat_id, action):
    """Procesa las acciones de los botones inline."""

    if action == "cmd_cuenta":
        subs = db_get_subscribers()
        user = subs.get(str(chat_id), {})
        joined = user.get("joined", "Desconocido")
        msg = (
            f"📊 *Tu Cuenta*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 *Nombre:* {user.get('first_name', 'N/A')}\n"
            f"🆔 *Chat ID:* `{chat_id}`\n"
            f"📅 *Miembro desde:* `{joined[:10] if len(joined) > 10 else joined}`\n"
            f"✅ *Estado:* Activo\n"
            f"💳 *Plan:* Free\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Upgrade a VIP para más features_"
        )
        tg_send_buttons(chat_id, msg.replace(".", "\\."), [
            [{"text": "💎 Ver Planes", "callback_data": "cmd_vip"}, {"text": "🔙 Menú", "callback_data": "cmd_menu"}]
        ])

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
        tg_send_buttons(chat_id, msg, [
            [{"text": "📊 Rendimiento OOS", "callback_data": "cmd_performance"}],
            [{"text": "📜 Ver Historial", "callback_data": "cmd_history"}, {"text": "🔙 Menú", "callback_data": "cmd_menu"}]
        ])

    elif action == "cmd_performance":
        report = {}
        if db:
            doc = db.collection("reports").document("blind_test_v8").get()
            if doc.exists:
                report = doc.to_dict()
        
        if not report:
            tg_send_buttons(chat_id,
                "📊 *Rendimiento Out-of-Sample*\n━━━━━━━━━━━━━━━━━━━━\n\n_Reporte no generado aún\\._\n\nEjecutá `blind_backtest.py` para generar métricas de ingeniería\\.",
                [[{"text": "🔙 Menú", "callback_data": "cmd_menu"}]])
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
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Validación técnica sobre datos nunca vistos_"
        )
        tg_send_buttons(chat_id, msg.replace(".", "\\."), [
            [{"text": "📜 Historial", "callback_data": "cmd_history"}, {"text": "🔙 Menú", "callback_data": "cmd_menu"}]
        ])

    elif action == "cmd_history":
        trades = []
        if db:
            docs = db.collection("backtest_history").order_by("order").limit(10).stream()
            trades = [doc.to_dict() for doc in docs]
        
        if not trades:
            tg_send_buttons(chat_id,
                "📜 *Historial de Operaciones*\n━━━━━━━━━━━━━━━━━━━━\n\n_No hay operaciones registradas aún\\._\n\n_El historial se actualiza con cada backtest\\._",
                [[{"text": "🔙 Menú", "callback_data": "cmd_menu"}]])
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
        
        tg_send_buttons(chat_id, msg.replace(".", "\\."), [
            [{"text": "📊 Cuenta", "callback_data": "cmd_cuenta"}, {"text": "🔙 Menú", "callback_data": "cmd_menu"}]
        ])

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
        tg_send_buttons(chat_id, msg, [
            [{"text": "📊 Cuenta", "callback_data": "cmd_cuenta"}, {"text": "🔙 Menú", "callback_data": "cmd_menu"}]
        ])

    elif action == "cmd_price":
        tg_send(chat_id, "💰 *Consultando precios actuales\\.\\.\\.*")
        msg = "📊 *Precios en Vivo*\n━━━━━━━━━━━━━━━━━━━━\n"
        for pair, config in PAIRS.items():
            df = download_data(config["ticker"], days=5)
            if df is not None and not df.empty:
                last_price = df.iloc[-1]["Close"]
                flag = PAIR_FLAGS.get(pair, "💱")
                msg += f"{flag} *{pair}:* `{last_price:.{config['decimals']}f}`\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n_Datos vía Yahoo Finance_"
        tg_send_buttons(chat_id, msg.replace(".", "\\."), [
            [{"text": "📈 Activas", "callback_data": "cmd_active"}, {"text": "🔙 Menú", "callback_data": "cmd_menu"}]
        ])

    elif action == "cmd_active":
        active = db_get_active_signals()
        if not active:
            tg_send_buttons(chat_id, "📭 *No hay operaciones abiertas en este momento\\.*", [
                [{"text": "📜 Historial", "callback_data": "cmd_history"}, {"text": "🔙 Menú", "callback_data": "cmd_menu"}]
            ])
            return
        msg = "📈 *Operaciones en Seguimiento*\n━━━━━━━━━━━━━━━━━━━━\n"
        for pair, s in active.items():
            emoji = "🟢 BUY" if s["signal"] == "BUY" else "🔴 SELL"
            msg += f"*{pair}* — {emoji}\n📍 En: `{s['entry']}`\n🛑 SL: `{s['sl']}` | 🎯 TP: `{s['tp']}`\n\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n_Monitoreando cada 5 min_"
        tg_send_buttons(chat_id, msg.replace(".", "\\."), [
            [{"text": "💰 Precios", "callback_data": "cmd_price"}, {"text": "🔙 Menú", "callback_data": "cmd_menu"}]
        ])

    elif action == "cmd_menu":
        tg_send_buttons(chat_id, "🤖 *Menú Principal*", get_main_buttons())


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
                if not config: continue
                
                df = download_data(config["ticker"], days=3)
                if df is None or df.empty: continue
                
                curr = df.iloc[-1]["Close"]
                high = df.iloc[-1]["High"]
                low = df.iloc[-1]["Low"]
                
                hit = None
                if s["signal"] == "BUY":
                    if high >= s["tp"]: hit = "TP"
                    elif low <= s["sl"]: hit = "SL"
                else: # SELL
                    if low <= s["tp"]: hit = "TP"
                    elif high >= s["sl"]: hit = "SL"
                
                if hit:
                    # Notificar cierre
                    emoji = "🎯 TP" if hit == "TP" else "🛑 SL"
                    flag = PAIR_FLAGS.get(pair, "💱")
                    msg = (
                        f"🏁 *OPERACIÓN CERRADA* — {pair}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"{flag} *Cierre:* {emoji}\n"
                        f"📍 *Entrada:* `{str(s['entry']).replace('.', '\\.')}`\n"
                        f"🏁 *Salida:*  `{str(curr).replace('.', '\\.')}`\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"{'✅ PROFIT' if hit == 'TP' else '❌ LOSS'}\n"
                    )
                    tg_broadcast(msg)
                    db_close_signal(pair, hit, curr)
            
            time.sleep(300) # Revisa cada 5 min
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
                    handle_callback_query_action(cb_chat_id, cb_data)
                    continue
                
                msg = update.get("message")
                if not msg:
                    continue
                chat_id = msg["chat"]["id"]
                text = msg.get("text", "")
                first_name = msg["chat"].get("first_name", "")
                username = msg["chat"].get("username", "")
                handle_command(chat_id, text, first_name, username)
        except KeyboardInterrupt:
            log.info("Bot detenido.")
            break
        except Exception as e:
            log.error(f"Error en polling: {e}")
            time.sleep(5)


def run_scheduler():
    """Programa el scan diario."""
    scan_time = f"{SCAN_HOUR:02d}:00"
    schedule.every().day.at(scan_time).do(scan_and_broadcast)
    log.info(f"Scan programado diariamente a las {scan_time}")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-now", action="store_true", help="Hacer scan y broadcast ahora")
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
