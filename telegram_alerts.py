"""
Telegram Alerts — ML Trading Bot
==================================
Módulo para enviar alertas al bot de Telegram cuando se detecta una señal.

Configuración inicial:
  1. Abrí Telegram → buscá tu bot → mandá /start
  2. python telegram_alerts.py --setup    ← guarda tu chat_id
  3. python telegram_alerts.py --test     ← prueba el bot

Integración con el scanner:
  python live_scanner.py   (ya integrado automáticamente)
"""
import os
import sys
import json
import argparse
import requests
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────
BOT_TOKEN = "5967657374:AAHX9XuJBmRxIYWn9AgcsCBtTK5mr3O2yTY"
CONFIG_FILE = "telegram_config.json"
# ──────────────────────────────────────────────────────────────────────


def _api(endpoint, params=None, json_data=None):
    """Llama a la API de Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{endpoint}"
    try:
        if json_data:
            r = requests.post(url, json=json_data, timeout=10)
        else:
            r = requests.get(url, params=params, timeout=10)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_chat_id():
    """Devuelve el chat_id guardado (o None si no hay)."""
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    return cfg.get("chat_id")


def save_chat_id(chat_id, username=""):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"chat_id": chat_id, "username": username}, f)


def setup():
    """Detecta automáticamente el chat_id después de que el usuario mande /start."""
    print("─" * 50)
    print("  CONFIGURACIÓN — Telegram Bot")
    print("─" * 50)
    print(f"\n  1. Abrí Telegram")
    print(f"  2. Buscá tu bot y mandá cualquier mensaje (ej: /start)")
    print(f"  3. Esperá...")
    print(f"\n  Esperando mensaje...", end="", flush=True)

    for attempt in range(30):
        result = _api("getUpdates")
        if result.get("ok") and result.get("result"):
            for update in result["result"]:
                msg = update.get("message") or update.get("channel_post")
                if msg:
                    chat = msg["chat"]
                    chat_id = chat["id"]
                    username = chat.get("username") or chat.get("first_name", "")
                    save_chat_id(chat_id, username)
                    print(f"\n\n  ✅ Chat ID detectado: {chat_id}")
                    print(f"  Usuario: {username}")
                    print(f"  Guardado en: {CONFIG_FILE}")
                    return chat_id

        print(".", end="", flush=True)
        import time
        time.sleep(2)

    print(f"\n\n  ⏱️ Timeout — mandá un mensaje al bot e intentá de nuevo.")
    return None


def send_signal_alert(pair, signal, entry, sl, tp, sl_pips, tp_pips,
                      confidence, risk_usd, volume, atr_pips):
    """Envía alerta de señal al Telegram."""
    chat_id = get_chat_id()
    if not chat_id:
        print("  ⚠️ Telegram sin configurar. Ejecutá: python telegram_alerts.py --setup")
        return False

    emoji_signal = "🟢 BUY" if signal == "BUY" else "🔴 SELL"
    emoji_pair = {
        "EURUSD": "🇪🇺🇺🇸", "GBPUSD": "🇬🇧🇺🇸", "AUDUSD": "🇦🇺🇺🇸",
        "NZDUSD": "🇳🇿🇺🇸", "USDCAD": "🇺🇸🇨🇦", "USDCHF": "🇺🇸🇨🇭",
        "EURGBP": "🇪🇺🇬🇧", "EURJPY": "🇪🇺🇯🇵", "EURNZD": "🇪🇺🇳🇿",
    }.get(pair, "💱")

    text = (
        f"⚡ *SEÑAL ML* — {datetime.now().strftime('%H:%M %d/%m/%Y')}\n\n"
        f"{emoji_pair} *{pair}* — {emoji_signal}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 *Entry:*    `{entry}`\n"
        f"🛑 *SL:*       `{sl}`  \\({sl_pips:.0f} pips\\)\n"
        f"🎯 *TP:*       `{tp}`  \\({tp_pips:.0f} pips\\)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *ATR:*      {atr_pips:.0f} pips  \\|  *R:R* 1:1\\.5\n"
        f"🤖 *Confianza:* {confidence:.0%}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Volumen:*   {volume:.2f} lotes\n"
        f"⚠️  *Riesgo:*   ${risk_usd:.0f}\n"
        f"✅ *Si TP:*    \\+${risk_usd * 1.5:.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📁 _Registrado en trade\\_journal.csv_"
    )

    result = _api("sendMessage", json_data={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",
    })

    if result.get("ok"):
        print(f"  📱 Telegram alert enviada")
        return True
    else:
        print(f"  ⚠️ Error Telegram: {result.get('description', result)}")
        return False


def send_no_signals_alert():
    """Envía mensaje de no-señales (opcional, off por defecto)."""
    chat_id = get_chat_id()
    if not chat_id:
        return

    text = (
        f"⏸ *Sin señales* — {datetime.now().strftime('%H:%M %d/%m/%Y')}\n"
        f"El scanner revisó 7 pares y todos están en HOLD\\.\n"
        f"_Próximo scan: mañana al cierre de vela diaria\\._"
    )
    _api("sendMessage", json_data={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",
    })


def send_test_alert():
    """Envía un mensaje de prueba."""
    chat_id = get_chat_id()
    if not chat_id:
        print("  ❌ Chat ID no configurado. Ejecutá: python telegram_alerts.py --setup")
        return False

    text = (
        "✅ *ML Trading Bot conectado correctamente\\!*\n\n"
        "Recibirás alertas cuando el modelo detecte señales\\.\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🤖 _GradientBoosting multi\\-pair v4_\n"
        "📊 _7 pares \\| Threshold: 0\\.60_\n"
        "💰 _Riesgo: 0\\.5% por trade_"
    )

    result = _api("sendMessage", json_data={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",
    })

    if result.get("ok"):
        print(f"  ✅ Mensaje de prueba enviado a chat_id: {chat_id}")
        return True
    else:
        print(f"  ❌ Error: {result.get('description', result)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telegram Alerts")
    parser.add_argument("--setup", action="store_true", help="Configurar chat_id")
    parser.add_argument("--test", action="store_true", help="Enviar mensaje de prueba")
    parser.add_argument("--no-signals", action="store_true", help="Enviar alerta de no-señales")
    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.test:
        send_test_alert()
    elif args.no_signals:
        send_no_signals_alert()
    else:
        parser.print_help()
