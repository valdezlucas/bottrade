import pandas as pd

def detect_fractals(df):
    df["fractal_high"] = False
    df["fractal_low"] = False
    
    for i in range(2, len(df)-2):
        if df["High"][i] > df["High"][i-1] and df["High"][i] > df["High"][i-2] \
        and df["High"][i] > df["High"][i+1] and df["High"][i] > df["High"][i+2]:
            df.at[i, "fractal_high"] = True
            
        if df["Low"][i] < df["Low"][i-1] and df["Low"][i] < df["Low"][i-2] \
        and df["Low"][i] < df["Low"][i+1] and df["Low"][i] < df["Low"][i+2]:
            df.at[i, "fractal_low"] = True
            
    return df