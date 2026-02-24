import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator


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
    df["price_vs_EMA20"] = (df["Close"] - df["EMA20"]) / df[
        "ATR"
    ]  # Distancia normalizada
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
    df["upper_wick_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / (
        hl_range + 1e-10
    )
    df["lower_wick_ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / (
        hl_range + 1e-10
    )

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
    df["at_fib_618"] = (
        (df["fib_position"] > 0.55) & (df["fib_position"] < 0.68)
    ).astype(int)
    df["at_fib_382"] = (
        (df["fib_position"] > 0.33) & (df["fib_position"] < 0.45)
    ).astype(int)
    # Limpiar columnas auxiliares
    df.drop(columns=["swing_high_20", "swing_low_20"], inplace=True)

    # --- D. EMA50 Confluencia (pendiente + proximidad) ---
    df["EMA50_slope"] = df["EMA50"].diff(3) / df["ATR"].replace(
        0, np.nan
    )  # pendiente 3 barras
    df["EMA50_touch"] = ((df["Close"] - df["EMA50"]).abs() < df["ATR"] * 0.3).astype(
        int
    )
    df["EMA50_slope_pos"] = (df["EMA50_slope"] > 0).astype(int)

    # --- F. Nuevos Features de Sensibilidad (Phase 2 Refinement) ---
    # ADX: Fuerza de la tendencia (0-100)
    adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ADX"] = adx.adx()
    df["ADX_pos"] = adx.adx_pos()
    df["ADX_neg"] = adx.adx_neg()

    # Stochastic RSI: Muy sensible a sobrecompra/sobreventa rápida
    stoch_rsi = StochRSIIndicator(close=df["Close"], window=14, smooth1=3, smooth2=3)
    df["StochRSI"] = stoch_rsi.stochrsi()
    df["StochRSI_K"] = stoch_rsi.stochrsi_k()
    df["StochRSI_D"] = stoch_rsi.stochrsi_d()

    # MFI (Money Flow Index): RSI pesado por volumen
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        mfi = MFIIndicator(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            volume=df["Volume"],
            window=14,
        )
        df["MFI"] = mfi.money_flow_index()
    else:
        df["MFI"] = df["RSI"]  # Fallback

    # Bollinger Band Squeeze: Detecta compresión de volatilidad
    # Ratio entre BB_width actual y su media de 100 periodos
    df["BB_squeeze"] = df["BB_width"] / df["BB_width"].rolling(100).mean().replace(
        0, np.nan
    )

    # --- Fractal features (distancia al último fractal) ---
    df = _add_fractal_distance_features(df)

    # --- E. Cambio de estructura (HH/HL ↔ LH/LL via fractales) ---
    df = _add_structure_features(df)

    # --- G. Seasonality / Time Features ---
    if "Datetime" in df.columns:
        dt = pd.to_datetime(df["Datetime"])

        # Day of week (0 = Monday, 6 = Sunday)
        # Using 7 days for full circularity, though forex/stocks trade 5 days
        day_of_week = dt.dt.dayofweek
        df["sin_day"] = np.sin(2 * np.pi * day_of_week / 7)
        df["cos_day"] = np.cos(2 * np.pi * day_of_week / 7)

        # Month of year (1-12)
        month = dt.dt.month
        df["sin_month"] = np.sin(2 * np.pi * month / 12)
        df["cos_month"] = np.cos(2 * np.pi * month / 12)

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
            last_fh_idx = i - 2  # El pico real fue hace 2 velas
            last_fh_price = df["High"].iloc[last_fh_idx]

        if df["fractal_low"].iloc[i]:
            last_fl_idx = i - 2  # El piso real fue hace 2 velas
            last_fl_price = df["Low"].iloc[last_fl_idx]

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
            fh_prices.append(df["High"].iloc[i - 2])  # Guardar precio real del pico
        if df["fractal_low"].iloc[i]:
            fl_prices.append(df["Low"].iloc[i - 2])  # Guardar precio real del valle

        if len(fh_prices) >= 2 and len(fl_prices) >= 2:
            hh = fh_prices[-1] > fh_prices[-2]  # Higher High
            hl = fl_prices[-1] > fl_prices[-2]  # Higher Low
            lh = fh_prices[-1] < fh_prices[-2]  # Lower High
            ll = fl_prices[-1] < fl_prices[-2]  # Lower Low

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
