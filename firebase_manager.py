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
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import os
import logging
from datetime import datetime

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
                log.info(f"🔍 Cargando Firebase desde env var (Longitud: {len(clean_key)})")
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
            log.error(f"❌ FIREBASE_KEY no encontrada en el entorno. Variables disponibles: {env_keys}")
            raise FileNotFoundError("No se encontró FIREBASE_KEY ni firebase_key.json en la raíz.")

        firebase_admin.initialize_app(cred)
    db = firestore.client()
    log.info("🔥 Firebase conectado correctamente")
except Exception as e:
    db = None
    log.error(f"❌ Error conectando a Firebase: {e}")

# ─── Suscriptores ──────────────────────────────────────────────────────
def db_add_subscriber(chat_id, first_name="", username=""):
    if not db: return
    doc_ref = db.collection("subscribers").document(str(chat_id))
    doc_ref.set({
        "first_name": first_name,
        "username": username,
        "active": True,
        "joined": datetime.now().isoformat()
    }, merge=True)

def db_get_subscribers():
    if not db: return {}
    docs = db.collection("subscribers").where(filter=FieldFilter("active", "==", True)).stream()
    return {doc.id: doc.to_dict() for doc in docs}

def db_remove_subscriber(chat_id):
    if not db: return
    db.collection("subscribers").document(str(chat_id)).update({"active": False})

# ─── Señales ───────────────────────────────────────────────────────────
def db_save_signal(pair, signal_data):
    """Guarda una señal activa para monitoreo."""
    if not db: return
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
    if not db: return {}
    docs = db.collection("active_signals").where(filter=FieldFilter("status", "==", "OPEN")).stream()
    return {doc.id: doc.to_dict() for doc in docs}

def db_close_signal(pair, result, exit_price):
    """Mueve una señal al historial cuando toca SL o TP."""
    if not db: return
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

if __name__ == "__main__":
    # Test simple
    logging.basicConfig(level=logging.INFO)
    if db:
        print("Probando escritura...")
        db_add_subscriber(12345, "TestUser", "tester")
        subs = db_get_subscribers()
        print(f"Suscriptores en DB: {len(subs)}")
        print("✅ Test de conexión exitoso")
    else:
        print("❌ Falló la conexión")
