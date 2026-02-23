import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import yfinance as yf
from features import create_features
from fractals import detect_fractals
from ml_dataset import label_data

# Extendemos con los nuevos pares y acciones
EXTENDED_ASSETS = {
    # Forex Base
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "NZDUSD": "NZDUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "EURGBP": "EURGBP=X",
    # Nuevos Forex
    "USDJPY": "USDJPY=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    # Acciones Globales
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "NVDA": "NVDA",
    "GOOGL": "GOOGL",
    "META": "META",
    "TSLA": "TSLA",
    "BRK-B": "BRK-B",
    "JPM": "JPM",
    "KO": "KO",
    "JNJ": "JNJ",
    "PG": "PG",
    "XOM": "XOM",
    "BABA": "BABA",
    "SAN": "SAN"
}

all_data = []

for name, ticker in EXTENDED_ASSETS.items():
    print(f"📥 Descargando {name} ({ticker}) (1D, desde 2010)...")
    df = yf.download(ticker, start="2010-01-01", interval="1d", progress=False)
    
    if df.empty:
        print(f"⚠️ {name} sin datos, saltando.")
        continue
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "Datetime"})
    
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    print(f"   ✅ {len(df)} velas descargadas.")
    
    print(f"   ⚙️ Calculando features y labels...")
    try:
        df = create_features(df)
        df = detect_fractals(df)
        df = label_data(df, lookahead=20, rr=1.5)
        # We append a column to identify the asset later if needed
        df['Asset'] = name
        all_data.append(df)
    except Exception as e:
        print(f"   ❌ Error procesando {name}: {e}")

print(f"\n🔗 Combinando {len(all_data)} activos...")
combined = pd.concat(all_data, ignore_index=True)
combined.to_csv("df_1d_joined.csv", index=False)
print(f"💾 Guardado df_1d_joined.csv con {len(combined)} filas.")
