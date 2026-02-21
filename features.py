import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD


def create_features(df):
    """
    Genera features técnicos sobre un DataFrame OHLC(V).
    Requiere columnas: Open, High, Low, Close.
    """

    # --- ATR ---
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ATR"] = atr.average_true_range()
    df["ATR_norm"] = df["ATR"] / df["Close"]  # ATR normalizado al precio

    # --- EMAs ---
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA_slope"] = df["EMA20"].diff()
    df["price_vs_EMA20"] = (df["Close"] - df["EMA20"]) / df["ATR"]  # Distancia normalizada
    df["price_vs_EMA50"] = (df["Close"] - df["EMA50"]) / df["ATR"]
    df["EMA_cross"] = df["EMA20"] - df["EMA50"]  # Positivo = bullish

    # --- RSI ---
    rsi = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi.rsi()

    # --- MACD ---
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    # --- Bollinger Bands ---
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    bb_range = bb_high - bb_low
    # Posición relativa del precio dentro de las bandas (0 = banda baja, 1 = banda alta)
    df["BB_position"] = np.where(bb_range > 0, (df["Close"] - bb_low) / bb_range, 0.5)
    df["BB_width"] = bb_range / df["Close"]  # Ancho normalizado

    # --- Volatilidad ---
    df["volatility"] = df["Close"].rolling(20).std()
    df["volatility_norm"] = df["volatility"] / df["Close"]

    # --- Fractal features (distancia al último fractal) ---
    df = _add_fractal_distance_features(df)

    return df


def _add_fractal_distance_features(df):
    """
    Agrega distancia (en barras y en precio normalizado por ATR)
    al último fractal high y low detectado.
    """
    from fractals import detect_fractals

    # Aplicar detección de fractales si no existen las columnas
    if "fractal_high" not in df.columns:
        df = detect_fractals(df)

    df["bars_since_fractal_high"] = np.nan
    df["bars_since_fractal_low"] = np.nan
    df["dist_fractal_high"] = np.nan
    df["dist_fractal_low"] = np.nan

    last_fh_idx = None
    last_fh_price = None
    last_fl_idx = None
    last_fl_price = None

    for i in range(len(df)):
        if df["fractal_high"].iloc[i]:
            last_fh_idx = i
            last_fh_price = df["High"].iloc[i]

        if df["fractal_low"].iloc[i]:
            last_fl_idx = i
            last_fl_price = df["Low"].iloc[i]

        atr_val = df["ATR"].iloc[i]

        if last_fh_idx is not None and atr_val > 0:
            df.iloc[i, df.columns.get_loc("bars_since_fractal_high")] = i - last_fh_idx
            df.iloc[i, df.columns.get_loc("dist_fractal_high")] = (
                last_fh_price - df["Close"].iloc[i]
            ) / atr_val

        if last_fl_idx is not None and atr_val > 0:
            df.iloc[i, df.columns.get_loc("bars_since_fractal_low")] = i - last_fl_idx
            df.iloc[i, df.columns.get_loc("dist_fractal_low")] = (
                df["Close"].iloc[i] - last_fl_price
            ) / atr_val

    return df