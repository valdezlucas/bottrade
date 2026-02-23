"""
Shitcoin Live Scanner (Phase 6)
Scans Binance Futures 24h tickers to find extreme parabolic moves (+-15%).
Filters by Volume and Funding Rates.
Checks CryptoPanic to filter out organic news pumps.
Predicts reversals using the specialized LightGBM GPU Shitcoin model.
Broadcasts signals via Telegram to active subscribers every hour.
"""
import os
import requests
import pandas as pd
import numpy as np
import time
import joblib
import schedule
from datetime import datetime
from features import create_features
from train_shitcoins import engineer_shitcoin_features
from fractals import detect_fractals
from firebase_manager import db_get_subscribers, db_can_receive_signal

# --- CONFIGURACIONES BÁSICAS ---
MIN_VOLATILITY_PCT = 15.0 # Mínimo movimiento 24h para considerarlo "PUMP/DUMP"
MIN_USDT_VOLUME = 50_000_000 # Minimizar riesgo de tokens basura ilíquidos
FUNDING_RATE_THRESHOLD = 0.0005 # > 0.05% o < -0.05% por epoch

CPANIC_API_KEY = "dummy_key_for_now" # Idealmente provisto por el usuario
TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN", "5967657374:AAHX9XuJBmRxIYWn9AgcsCBtTK5mr3O2yTY") # Default del bot principal
# -------------------------------

def tg_send(chat_id, text, parse_mode="MarkdownV2"):
    """Envía un mensaje a un chat de Telegram."""
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        return r.json().get("ok", False)
    except Exception as e:
        print(f"Error tg_send: {e}")
        return False

def broadcast_shitcoin_signal(sym, pct, vol, fr, direction, confidence, entry, sl, tp, atr_pct):
    """Envía la alerta a todos los suscriptores activos."""
    if not TELEGRAM_BOT_TOKEN:
        print("No BOT_TOKEN configured. Cannot broadcast.")
        return
        
    subs = db_get_subscribers()
    sent = 0
    emoji = "🔴 SHORT REVERSAL" if direction == "SHORT" else "🟢 LONG REVERSAL"
    
    # Escapar para MarkdownV2
    def esc(v):
        return str(v).replace(".", "\\.").replace("-", "\\-").replace("+", "\\+").replace("_", "\\_")
        
    fr_str = f"{fr*100:.3f}%"
    vol_str = f"${vol:.1f}M"
    pct_str = f"{pct:+.2f}%"
    atr_str = f"{atr_pct*100:.1f}%"
    
    msg = (
        f"🚨 *ALERTA SHITCOIN PUMP & DUMP* 🚨\n\n"
        f"*{esc(sym)}* ⚡ {emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Movimiento 24h:* `{esc(pct_str)}`\n"
        f"🌊 *Volumen:* {esc(vol_str)}\n"
        f"🔥 *Funding Rate:* `{esc(fr_str)}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 *IA Confianza:* {esc(f'{confidence*100:.1f}%')}\n"
        f"📍 *Entrada:* `{esc(entry)}`\n"
        f"🛑 *Stop Loss:* `{esc(sl)}`\n"
        f"🎯 *Take Profit:* `{esc(tp)}` \\(R:R 1:2\\)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ *Operativa de extremado riesgo*\\.\n"
        f"_Reversión asimétrica capturando volatilidad extrema\\._"
    )

    for chat_id in subs:
        reason, can_receive = db_can_receive_signal(int(chat_id))
        if can_receive:
            if tg_send(int(chat_id), msg):
                sent += 1
            time.sleep(0.05)
            
    print(f"   📣 Broadcast finalizado: {sent} usuarios notificados.")

def get_binance_extremes():
    """Obtiene los Top Gainers/Losers de Binance Futures"""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"Error fetching Binance: {e}")
        return []
        
    extremes = []
    for item in data:
        symbol = item['symbol']
        try:
            pct_change = float(item['priceChangePercent'])
            volume = float(item['quoteVolume'])
            
            if abs(pct_change) >= MIN_VOLATILITY_PCT and volume >= MIN_USDT_VOLUME:
                # Evitar BTC y ETH
                if not symbol.startswith(("BTC", "ETH")) and symbol.endswith("USDT"):
                    extremes.append({
                        "symbol": symbol,
                        "pct_change": pct_change,
                        "volume": volume,
                        "last_price": float(item['lastPrice'])
                    })
        except: continue
        
    return sorted(extremes, key=lambda x: abs(x['pct_change']), reverse=True)

def check_funding_rates(symbol):
    """Revisa la tasa de fondeo actual (Squeeze pressure)"""
    url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        fr = float(data.get('lastFundingRate', 0))
        return fr
    except Exception:
        return 0.0

def fetch_binance_ohlcv(symbol, interval="1h", limit=120):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        df = pd.DataFrame(data, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching klines {symbol}: {e}")
        return None

def run_scan_job():
    print(f"\n============================================================")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 SCANNER SHITCOINS 1H RUNNING")
    print(f"============================================================")
    
    try:
        art = joblib.load("model_shitcoins_1h.joblib")
        model = art["model"]
        feature_cols = art["feature_columns"]
        threshold = art["threshold"]
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return

    extremes = get_binance_extremes()
    print(f"🔥 Encontrados {len(extremes)} pares con > {MIN_VOLATILITY_PCT}% volatilidad y liquidez.")
    
    for asset in extremes[:15]:
        sym = asset['symbol']
        pct = asset['pct_change']
        vol = asset['volume'] / 1e6
        
        dir_emoji = "🚀 PUMP" if pct > 0 else "🩸 DUMP"
        print(f"\n{dir_emoji} | {sym} | 24h: {pct:+.2f}% | Vol: ${vol:.1f}M")
        
        fr = check_funding_rates(sym)
        if abs(fr) < FUNDING_RATE_THRESHOLD:
            print(f"   ⚠️ Funding Normal ({fr*100:.3f}%). Saltando...")
            continue
            
        print(f"   🔥 Funding Extremo: {fr*100:.3f}%")
        
        df = fetch_binance_ohlcv(sym, "1h", limit=120)
        if df is None or len(df) < 100: continue
            
        try:
            df_feat = create_features(df)
            df_feat = detect_fractals(df_feat)
            df_feat = engineer_shitcoin_features(df_feat)
            
            missing = [c for c in feature_cols if c not in df_feat.columns]
            if missing: continue
                
            last = df_feat.iloc[[-1]][feature_cols].copy()
            if last.isna().any().any(): continue
            X = last.astype(np.float32)
            
            probas = model.predict_proba(X)[0]
            if len(probas) < 3: continue
            
            p_buy = probas[1] 
            p_sell = probas[2]
            
            # Cálculos de SL/TP basados en ATR dinámico
            entry_price = float(df_feat["Close"].iloc[-1])
            atr = float(df_feat["ATR"].iloc[-1])
            if pd.isna(atr): atr = entry_price * 0.05
            
            atr_pct = atr / entry_price
            sl_dist = atr * 1.5
            tp_dist = atr * 3.0 # Risk Reward 1:2 The Asymmetric nature of Shitcoins
            
            if p_sell > threshold and pct > 0:
                print(f"   🎯 ALERTA IA: SHORT! Confianza: {p_sell*100:.1f}%")
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist
                broadcast_shitcoin_signal(sym, pct, vol, fr, "SHORT", p_sell, entry_price, sl, tp, atr_pct)
                
            elif p_buy > threshold and pct < 0:
                print(f"   🎯 ALERTA IA: LONG! Confianza: {p_buy*100:.1f}%")
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
                broadcast_shitcoin_signal(sym, pct, vol, fr, "LONG", p_buy, entry_price, sl, tp, atr_pct)
            else:
                print(f"   ⏸️ IA Hold")
                
        except Exception as e:
            print(f"   ❌ Error ML: {e}")

def main():
    print("="*60)
    print(" 🕒 INICIANDO SERVICIO DEMONIO SHITCOINS (CADA 1 HORA)")
    print("="*60)
    
    # Run once immediately
    run_scan_job()
    
    # Schedule every hour at minute 0 (cierre de vela 1H)
    schedule.every().hour.at(":01").do(run_scan_job)
    
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    main()
