import pandas as pd
import yfinance as yf

from features import create_features
from fractals import detect_fractals
from ml_dataset import label_data
from train_multi_tf import PAIRS

all_data = []

for pair, config in PAIRS.items():
    print(f"📥 Descargando {pair} (1D, desde 2010)...")
    ticker = config["ticker"]
    df = yf.download(ticker, start="2010-01-01", interval="1d", progress=False)

    if df.empty:
        print(f"⚠️ {pair} sin datos, saltando.")
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "Datetime"})

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    print(f"   ✅ {len(df)} velas descargadas.")

    print(f"   ⚙️ Calculando features y labels...")
    df = create_features(df)
    df = detect_fractals(df)
    df = label_data(df, lookahead=20, rr=1.5)

    all_data.append(df)

print(f"\n🔗 Combinando {len(all_data)} pares...")
combined = pd.concat(all_data, ignore_index=True)
combined.to_csv("df_1d_joined.csv", index=False)
print(f"💾 Guardado df_1d_joined.csv con {len(combined)} filas.")
