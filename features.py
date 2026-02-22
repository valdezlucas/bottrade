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

    # ═══════════════════════════════════════════════════════════════
    #  STRATEGY FEATURES — Basados en análisis profesional
    # ═══════════════════════════════════════════════════════════════

    # --- A. Extremos del mercado (sobreventa / sobrecompra) ---
    df["RSI_extreme_buy"] = (df["RSI"] < 30).astype(int)
    df["RSI_extreme_sell"] = (df["RSI"] > 70).astype(int)
    df["price_at_BB_low"] = (df["BB_position"] < 0.1).astype(int)
    df["price_at_BB_high"] = (df["BB_position"] > 0.9).astype(int)

    # --- B. Desaceleración (velas achicándose en extremos) ---
    hl_range = df["High"] - df["Low"]
    body = (df["Close"] - df["Open"]).abs()
    # Tamaño del cuerpo normalizado por ATR
    df["body_size"] = body / df["ATR"].replace(0, np.nan)
    # ¿El cuerpo se achica? (últimas 3 velas: cuerpo actual < 50% del de hace 3)
    body_3_ago = body.shift(2)
    df["body_shrinking"] = np.where(
        body_3_ago > 0, (body < body_3_ago * 0.5).astype(int), 0
    )
    # Rango total de la vela normalizado
    df["candle_range"] = hl_range / df["ATR"].replace(0, np.nan)
    # Ratio de mechas — detecta rechazo (hammer, shooting star)
    df["upper_wick_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / (hl_range + 1e-10)
    df["lower_wick_ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / (hl_range + 1e-10)

    # --- C. Fibonacci Retracement (posición en swing reciente) ---
    df["swing_high_20"] = df["High"].rolling(20).max()
    df["swing_low_20"] = df["Low"].rolling(20).min()
    swing_range = df["swing_high_20"] - df["swing_low_20"]
    df["fib_position"] = np.where(
        swing_range > 0,
        (df["Close"] - df["swing_low_20"]) / swing_range,
        0.5,
    )
    # ¿Estamos en zona Fibonacci clave?
    df["at_fib_618"] = ((df["fib_position"] > 0.55) & (df["fib_position"] < 0.68)).astype(int)
    df["at_fib_382"] = ((df["fib_position"] > 0.33) & (df["fib_position"] < 0.45)).astype(int)
    # Limpiar columnas auxiliares
    df.drop(columns=["swing_high_20", "swing_low_20"], inplace=True)

    # --- D. EMA50 Confluencia (pendiente + proximidad) ---
    df["EMA50_slope"] = df["EMA50"].diff(3) / df["ATR"].replace(0, np.nan)  # pendiente 3 barras
    df["EMA50_touch"] = (
        (df["Close"] - df["EMA50"]).abs() < df["ATR"] * 0.3
    ).astype(int)
    df["EMA50_slope_pos"] = (df["EMA50_slope"] > 0).astype(int)

    # --- Fractal features (distancia al último fractal) ---
    df = _add_fractal_distance_features(df)

    # --- E. Cambio de estructura (HH/HL ↔ LH/LL via fractales) ---
    df = _add_structure_features(df)

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


def _add_structure_features(df):
    """
    Detecta cambio de estructura del mercado usando fractales.
    HH/HL = estructura alcista, LH/LL = estructura bajista.
    structure_change = momento en que cambia de una a otra.
    """
    df["structure_bullish"] = 0
    df["structure_bearish"] = 0
    df["structure_change"] = 0

    # Recolectar fractales como series de precios
    fh_prices = []  # fractal highs history: (idx, price)
    fl_prices = []  # fractal lows history: (idx, price)

    prev_direction = 0  # 0=undefined, 1=bullish, -1=bearish

    for i in range(len(df)):
        if df["fractal_high"].iloc[i]:
            fh_prices.append(df["High"].iloc[i])
        if df["fractal_low"].iloc[i]:
            fl_prices.append(df["Low"].iloc[i])

        if len(fh_prices) >= 2 and len(fl_prices) >= 2:
            hh = fh_prices[-1] > fh_prices[-2]  # Higher High
            hl = fl_prices[-1] > fl_prices[-2]   # Higher Low
            lh = fh_prices[-1] < fh_prices[-2]   # Lower High
            ll = fl_prices[-1] < fl_prices[-2]    # Lower Low

            if hh and hl:
                current_dir = 1  # bullish
                df.iloc[i, df.columns.get_loc("structure_bullish")] = 1
            elif lh and ll:
                current_dir = -1  # bearish
                df.iloc[i, df.columns.get_loc("structure_bearish")] = 1
            else:
                current_dir = prev_direction  # mixed, keep previous

            if prev_direction != 0 and current_dir != prev_direction:
                df.iloc[i, df.columns.get_loc("structure_change")] = 1

            prev_direction = current_dir

    # Forward-fill structure signals (persist between fractals)
    df["structure_bullish"] = df["structure_bullish"].replace(0, np.nan)
    df["structure_bullish"] = df["structure_bullish"].ffill().fillna(0).astype(int)
    df["structure_bearish"] = df["structure_bearish"].replace(0, np.nan)
    df["structure_bearish"] = df["structure_bearish"].ffill().fillna(0).astype(int)

    return df