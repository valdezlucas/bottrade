from patterns import detect_double_top


def multi_timeframe_signal(df_d1, df_h1, df_m5):

    # Dirección D1
    last_d1 = df_d1.iloc[-1]
    direction = None

    if last_d1["Close"] > last_d1["Open"]:
        direction = "bullish"
    else:
        direction = "bearish"

    # Patrón H1
    double_tops = detect_double_top(df_h1)

    if direction == "bearish" and len(double_tops) > 0:

        # Trigger M5
        last_m5 = df_m5.iloc[-1]

        if last_m5["Close"] < last_m5["Low"].rolling(5).min().iloc[-1]:
            return "SELL"

    return None
