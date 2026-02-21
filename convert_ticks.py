"""
Convierte tick data de MetaTrader/Finamark a velas OHLCV.

Formato de entrada (tab-separated):
  <DATE>  <TIME>  <BID>  <ASK>  <LAST>  <VOLUME>  <FLAGS>

Genera un CSV con: Open, High, Low, Close, Volume
"""
import pandas as pd
import sys


def ticks_to_ohlcv(input_path, output_path, timeframe="1h"):
    """
    Convierte ticks a velas OHLCV.

    Args:
        input_path: Path al CSV de ticks
        output_path: Path de salida para el CSV de velas
        timeframe: Timeframe de las velas ('1min', '5min', '15min', '1h', '4h', '1D')
    """
    print(f"Cargando ticks de: {input_path}")

    # Leer con separador tab
    df = pd.read_csv(input_path, sep="\t", header=0)

    # Limpiar nombres de columnas
    df.columns = [c.strip().replace("<", "").replace(">", "") for c in df.columns]

    print(f"Columnas detectadas: {list(df.columns)}")
    print(f"Ticks cargados: {len(df):,}")

    # Crear datetime
    df["datetime"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], format="mixed")
    df = df.set_index("datetime")

    # Usar BID como precio principal (es lo estándar en Forex)
    # Llenar BID vacíos con LAST, luego forward fill
    df["BID"] = pd.to_numeric(df["BID"], errors="coerce")
    df["ASK"] = pd.to_numeric(df["ASK"], errors="coerce")
    df["LAST"] = pd.to_numeric(df["LAST"], errors="coerce")

    df["price"] = df["BID"].fillna(df["LAST"]).ffill()

    # Eliminar filas sin precio
    df = df.dropna(subset=["price"])
    print(f"Ticks con precio valido: {len(df):,}")

    # Resamplear a velas OHLCV
    ohlcv = df["price"].resample(timeframe).agg(
        Open="first",
        High="max",
        Low="min",
        Close="last",
    )

    # Volumen: contar ticks por vela (proxy de volumen en Forex)
    volume = df["price"].resample(timeframe).count()
    ohlcv["Volume"] = volume

    # Eliminar velas vacías (fuera de horario)
    ohlcv = ohlcv.dropna(subset=["Open"]).reset_index(drop=True)

    print(f"\nVelas generadas ({timeframe}): {len(ohlcv):,}")
    print(f"Rango: {ohlcv['Open'].iloc[0]:.5f} - {ohlcv['Close'].iloc[-1]:.5f}")
    print(f"\nPrimeras 3 velas:")
    print(ohlcv.head(3).to_string())

    # Guardar
    ohlcv.to_csv(output_path, index=False)
    print(f"\nGuardado en: {output_path}")

    return ohlcv


if __name__ == "__main__":
    input_file = "NZDUSD_201502201900_202602202028.csv"
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    output_file = f"NZDUSD_{timeframe}.csv"

    ticks_to_ohlcv(input_file, output_file, timeframe=timeframe)
