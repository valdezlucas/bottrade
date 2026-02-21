import numpy as np


def label_data(df, lookahead=20, rr=1.5):
    """
    Etiqueta cada fila con:
      0 = HOLD
      1 = BUY  (oportunidad larga)
      2 = SELL (oportunidad corta)

    Criterio normalizado por ATR:
      - BUY:  potencial alcista > rr * ATR  Y  riesgo bajista < ATR
      - SELL: potencial bajista > rr * ATR  Y  riesgo alcista < ATR

    También calcula avg_win y avg_loss potencial por señal para 
    poder calcular expectancy en la evaluación.
    """

    df["target"] = 0
    df["potential_win"] = 0.0
    df["potential_loss"] = 0.0

    for i in range(len(df) - lookahead):
        entry = df["Close"].iloc[i]
        atr = df["ATR"].iloc[i]

        if atr <= 0 or np.isnan(atr):
            continue

        future_slice_high = df["High"].iloc[i + 1 : i + 1 + lookahead]
        future_slice_low = df["Low"].iloc[i + 1 : i + 1 + lookahead]

        future_max = future_slice_high.max()
        future_min = future_slice_low.min()

        # Potencial BUY
        buy_reward = future_max - entry
        buy_risk = entry - future_min

        # Potencial SELL
        sell_reward = entry - future_min
        sell_risk = future_max - entry

        # BUY: reward significativo y riesgo controlado
        if buy_reward > rr * atr and buy_risk < atr:
            df.iat[i, df.columns.get_loc("target")] = 1
            df.iat[i, df.columns.get_loc("potential_win")] = buy_reward
            df.iat[i, df.columns.get_loc("potential_loss")] = buy_risk

        # SELL: reward significativo y riesgo controlado
        elif sell_reward > rr * atr and sell_risk < atr:
            df.iat[i, df.columns.get_loc("target")] = 2
            df.iat[i, df.columns.get_loc("potential_win")] = sell_reward
            df.iat[i, df.columns.get_loc("potential_loss")] = sell_risk

    return df