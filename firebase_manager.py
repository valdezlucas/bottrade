"""
Firebase Manager — ML Trading Bot
==================================
Módulo para manejar la persistencia en Google Firebase Firestore.

Colecciones:
  - subscribers: { chat_id: { first_name, username, joined } }
  - active_signals: { pair: { signal, entry, sl, tp, status, timestamp } }
  - signals_history: Historial de señales cerradas.

Configuración:
  Requiere el archivo firebase_key.json en la raíz del proyecto.
"""

import logging
import os
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

log = logging.getLogger(__name__)

# Inicializar Firebase
try:
    if not firebase_admin._apps:
        # 1. Intentar desde variable de entorno (para Railway)
        # Buscamos 'FIREBASE_KEY' pero también somos robustos a espacios accidentales en el nombre de la variable
        json_key = os.environ.get("FIREBASE_KEY")

        if not json_key:
            # Buscar si existe una variable que se parezca (ej: ' FIREBASE_KEY')
            for k, v in os.environ.items():
                if k.strip() == "FIREBASE_KEY":
                    json_key = v
                    log.info(f"⚠️ Detectada variable con espacio en el nombre: '{k}'")
                    break

        if json_key:
            import json
            import re

            clean_key = json_key.strip()
            # Robustez extra: Buscar el primer '{' y el último '}' por si se coló basura al copiar
            match = re.search(r"(\{.*\})", clean_key, re.DOTALL)
            if match:
                clean_key = match.group(1)

            try:
                service_account_info = json.loads(clean_key)
                cred = credentials.Certificate(service_account_info)
                log.info(
                    f"🔍 Cargando Firebase desde env var (Longitud: {len(clean_key)})"
                )
            except json.JSONDecodeError as je:
                log.error(f"❌ Error decodificando JSON: {je}")
                log.error(f"   Primeros 20 chars: {clean_key[:20]}...")
                log.error(f"   Últimos 20 chars: ...{clean_key[-20:]}")
                raise je
        # 2. Intentar desde archivo local
        elif os.path.exists("firebase_key.json"):
            cred = credentials.Certificate("firebase_key.json")
            log.info("🔍 Cargando Firebase desde archivo firebase_key.json")

        else:
            env_keys = list(os.environ.keys())
            log.error(
                f"❌ FIREBASE_KEY no encontrada en el entorno. Variables disponibles: {env_keys}"
            )
            raise FileNotFoundError(
                "No se encontró FIREBASE_KEY ni firebase_key.json en la raíz."
            )

        firebase_admin.initialize_app(cred)
    db = firestore.client()
    log.info("🔥 Firebase conectado correctamente")
except Exception as e:
    db = None
    log.error(f"❌ Error conectando a Firebase: {e}")


# ─── Suscriptores ──────────────────────────────────────────────────────
def db_add_subscriber(chat_id, first_name="", username=""):
    if not db:
        return
    doc_ref = db.collection("subscribers").document(str(chat_id))
    doc_ref.set(
        {
            "first_name": first_name,
            "username": username,
            "active": True,
            "joined": datetime.now().isoformat(),
        },
        merge=True,
    )


def db_get_subscribers():
    if not db:
        return {}
    docs = (
        db.collection("subscribers")
        .where(filter=FieldFilter("active", "==", True))
        .stream()
    )
    return {doc.id: doc.to_dict() for doc in docs}


def db_remove_subscriber(chat_id):
    if not db:
        return
    db.collection("subscribers").document(str(chat_id)).update({"active": False})


# ─── Señales ───────────────────────────────────────────────────────────
def db_save_signal(pair, signal_data):
    """Guarda una señal activa para monitoreo."""
    if not db:
        return
    # No usamos .document(pair) para permitir múltiples señales del mismo par si hiciera falta,
    # pero para el bot diario, una por par está bien.
    # Usamos el par como ID para actualizarla fácilmente en el monitoreo.
    doc_ref = db.collection("active_signals").document(pair)
    signal_data["timestamp"] = datetime.utcnow().isoformat()
    signal_data["status"] = "OPEN"
    doc_ref.set(signal_data)
    log.info(f"💾 Señal {pair} guardada en Firebase")


def db_get_active_signals():
    """Retorna todas las señales en estado OPEN."""
    if not db:
        return {}
    docs = (
        db.collection("active_signals")
        .where(filter=FieldFilter("status", "==", "OPEN"))
        .stream()
    )
    return {doc.id: doc.to_dict() for doc in docs}


def db_close_signal(pair, result, exit_price):
    """Mueve una señal al historial cuando toca SL o TP."""
    if not db:
        return
    doc_ref = db.collection("active_signals").document(pair)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        data["status"] = "CLOSED"
        data["result"] = result  # "TP" o "SL"
        data["exit_price"] = exit_price
        data["closed_at"] = datetime.utcnow().isoformat()

        # Guardar en historial y borrar de activas
        db.collection("signals_history").add(data)
        doc_ref.delete()
        log.info(f"🏁 Señal {pair} cerrada ({result}) y movida al historial")


# ─── Gestión de Cuentas (v9) ──────────────────────────────────────────
SIGNAL_COST = 0.50  # USD por señal
TRIAL_DAYS = 15


def db_init_user_account(chat_id, first_name="", username=""):
    """Inicializa la cuenta de un usuario nuevo con trial de 15 días."""
    if not db:
        return
    from datetime import timedelta

    doc_ref = db.collection("subscribers").document(str(chat_id))
    doc = doc_ref.get()

    now = datetime.now()

    # Si ya existe, solo actualizamos nombre/username (no resetear trial)
    if doc.exists:
        data = doc.to_dict()
        update = {"first_name": first_name, "username": username, "active": True}
        # Solo inicializar campos nuevos si no existen
        if "balance" not in data:
            update["balance"] = 0.0
        if "trial_start" not in data:
            update["trial_start"] = now.isoformat()
            update["trial_end"] = (now + timedelta(days=TRIAL_DAYS)).isoformat()
        if "alerts_enabled" not in data:
            update["alerts_enabled"] = True
        if "total_signals_received" not in data:
            update["total_signals_received"] = 0
        if "total_spent" not in data:
            update["total_spent"] = 0.0
        doc_ref.set(update, merge=True)
    else:
        # Usuario completamente nuevo
        doc_ref.set(
            {
                "first_name": first_name,
                "username": username,
                "active": True,
                "joined": now.isoformat(),
                "balance": 0.0,
                "alerts_enabled": True,
                "trial_start": now.isoformat(),
                "trial_end": (now + timedelta(days=TRIAL_DAYS)).isoformat(),
                "total_signals_received": 0,
                "total_spent": 0.0,
            }
        )
    log.info(f"👤 Cuenta inicializada para {first_name} ({chat_id})")


def db_get_user_account(chat_id):
    """Devuelve los datos completos de un usuario."""
    if not db:
        return None
    doc = db.collection("subscribers").document(str(chat_id)).get()
    if doc.exists:
        return doc.to_dict()
    return None


def db_is_trial_active(chat_id):
    """Verifica si el trial de 15 días sigue vigente."""
    user = db_get_user_account(chat_id)
    if not user:
        return False
    trial_end = user.get("trial_end")
    if not trial_end:
        return False
    try:
        end_date = datetime.fromisoformat(trial_end)
        return datetime.now() < end_date
    except:
        return False


def db_toggle_alerts(chat_id):
    """Activa/desactiva alertas. Retorna el nuevo estado."""
    if not db:
        return None
    doc_ref = db.collection("subscribers").document(str(chat_id))
    doc = doc_ref.get()
    if not doc.exists:
        return None

    current = doc.to_dict().get("alerts_enabled", True)
    new_state = not current
    doc_ref.update({"alerts_enabled": new_state})
    return new_state


def db_deposit(chat_id, amount):
    """Suma saldo a la cuenta del usuario. Retorna el nuevo balance."""
    if not db or amount <= 0:
        return None
    doc_ref = db.collection("subscribers").document(str(chat_id))
    doc = doc_ref.get()
    if not doc.exists:
        return None

    data = doc.to_dict()
    new_balance = data.get("balance", 0.0) + amount
    doc_ref.update({"balance": new_balance})

    # Registrar transacción
    db.collection("transactions").add(
        {
            "chat_id": str(chat_id),
            "type": "deposit",
            "amount": amount,
            "balance_after": new_balance,
            "timestamp": datetime.now().isoformat(),
        }
    )
    log.info(
        f"💰 Depósito de ${amount:.2f} para {chat_id}. Nuevo saldo: ${new_balance:.2f}"
    )
    return new_balance


def db_charge_signal(chat_id):
    """Cobra $0.50 por señal. Retorna True si se pudo cobrar."""
    if not db:
        return False
    doc_ref = db.collection("subscribers").document(str(chat_id))
    doc = doc_ref.get()
    if not doc.exists:
        return False

    data = doc.to_dict()
    balance = data.get("balance", 0.0)

    if balance < SIGNAL_COST:
        return False

    new_balance = balance - SIGNAL_COST
    total_spent = data.get("total_spent", 0.0) + SIGNAL_COST
    total_signals = data.get("total_signals_received", 0) + 1

    doc_ref.update(
        {
            "balance": new_balance,
            "total_spent": total_spent,
            "total_signals_received": total_signals,
        }
    )

    # Registrar transacción
    db.collection("transactions").add(
        {
            "chat_id": str(chat_id),
            "type": "signal_charge",
            "amount": -SIGNAL_COST,
            "balance_after": new_balance,
            "timestamp": datetime.now().isoformat(),
        }
    )
    return True


def db_can_receive_signal(chat_id):
    """
    Lógica central de billing:
    Retorna (puede_recibir, razon)
    - ("ok_trial", True) → En periodo de prueba gratuito
    - ("ok_paid", True)  → Cobrado correctamente
    - ("no_alerts", False) → Alertas desactivadas
    - ("no_balance", False) → Sin saldo suficiente
    """
    user = db_get_user_account(chat_id)
    if not user:
        return "no_user", False

    # 1. Alertas desactivadas?
    if not user.get("alerts_enabled", True):
        return "no_alerts", False

    # 2. En trial?
    if db_is_trial_active(chat_id):
        # Incrementar contador sin cobrar
        if db:
            doc_ref = db.collection("subscribers").document(str(chat_id))
            doc_ref.update(
                {"total_signals_received": user.get("total_signals_received", 0) + 1}
            )
        return "ok_trial", True

    # 3. Tiene saldo?
    if db_charge_signal(chat_id):
        return "ok_paid", True

    return "no_balance", False


if __name__ == "__main__":
    # Test simple
    logging.basicConfig(level=logging.INFO)
    if db:
        print("Probando escritura...")
        db_init_user_account(12345, "TestUser", "tester")
        user = db_get_user_account(12345)
        print(f"Usuario: {user}")
        print(f"Trial activo: {db_is_trial_active(12345)}")
        print("✅ Test de conexión exitoso")
    else:
        print("❌ Falló la conexión")
