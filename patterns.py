def detect_double_top(df, tolerance=0.002):
    tops = df[df["fractal_high"]]
    patterns = []

    for i in range(len(tops)-1):
        price1 = tops.iloc[i]["High"]
        price2 = tops.iloc[i+1]["High"]

        if abs(price1 - price2) / price1 < tolerance:
            patterns.append(tops.index[i+1])

    return patterns