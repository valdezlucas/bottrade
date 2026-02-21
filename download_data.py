"""
Descarga datos históricos de Forex desde Yahoo Finance.

Yahoo Finance permite:
- Datos diarios (1D): hasta ~20 años
- Datos horarios (1h): hasta ~2 años (730 días)

Para Forex, el ticker es: NZDUSD=X, EURUSD=X, etc.
"""
import yfinance as yf
import pandas as pd
import sys


def download_forex(pair="NZDUSD", interval="1h", period=None, start=None, end=None, output=None):
    """
    Descarga datos OHLCV de un par de Forex.

    Args:
        pair: Par forex (ej: NZDUSD, EURUSD, GBPUSD)
        interval: 1m, 5m, 15m, 1h, 1d, 1wk
        period: Período (ej: 1y, 2y, 5y, 10y, max)
        start: Fecha inicio (YYYY-MM-DD)
        end: Fecha fin (YYYY-MM-DD)
        output: Nombre del archivo de salida
    """
    ticker = f"{pair}=X"

    print(f"Descargando {ticker} — intervalo: {interval}")

    kwargs = {"interval": interval, "auto_adjust": True}

    if start and end:
        kwargs["start"] = start
        kwargs["end"] = end
        print(f"Rango: {start} → {end}")
    elif period:
        kwargs["period"] = period
        print(f"Periodo: {period}")
    else:
        # Defaults por intervalo
        if interval in ["1m", "5m", "15m"]:
            kwargs["period"] = "60d"
        elif interval in ["1h"]:
            kwargs["period"] = "730d"  # Máximo para hourly
        else:
            kwargs["period"] = "max"
        print(f"Periodo: {kwargs['period']} (default para {interval})")

    data = yf.download(ticker, **kwargs)

    if data.empty:
        print("ERROR: No se obtuvieron datos")
        return None

    # Limpiar MultiIndex si existe
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Asegurar nombres de columnas estándar
    rename_map = {}
    for col in data.columns:
        col_lower = col.lower()
        if "open" in col_lower:
            rename_map[col] = "Open"
        elif "high" in col_lower:
            rename_map[col] = "High"
        elif "low" in col_lower:
            rename_map[col] = "Low"
        elif "close" in col_lower:
            rename_map[col] = "Close"
        elif "volume" in col_lower:
            rename_map[col] = "Volume"

    data = data.rename(columns=rename_map)

    # Eliminar filas con datos faltantes
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data.reset_index(drop=True)

    print(f"\nDatos descargados: {len(data):,} velas")
    print(f"Columnas: {list(data.columns)}")
    print(f"Rango precios: {data['Low'].min():.5f} — {data['High'].max():.5f}")
    print(f"\nPrimeras 3:")
    print(data.head(3).to_string())
    print(f"\nUltimas 3:")
    print(data.tail(3).to_string())

    # Guardar
    if output is None:
        output = f"{pair}_{interval}.csv"

    data.to_csv(output, index=False)
    print(f"\nGuardado en: {output}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descargar datos Forex de Yahoo Finance")
    parser.add_argument("--pair", default="NZDUSD", help="Par forex (default: NZDUSD)")
    parser.add_argument("--interval", default="1h", help="Intervalo: 1m, 5m, 15m, 1h, 1d (default: 1h)")
    parser.add_argument("--period", default=None, help="Periodo: 1y, 2y, 5y, max (default: auto)")
    parser.add_argument("--start", default=None, help="Fecha inicio: YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Fecha fin: YYYY-MM-DD")
    parser.add_argument("--output", default=None, help="Archivo de salida")

    args = parser.parse_args()
    download_forex(args.pair, args.interval, args.period, args.start, args.end, args.output)
