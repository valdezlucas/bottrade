import pandas as pd

def detect_fractals(df):
    df["fractal_high"] = False
    df["fractal_low"] = False
    
    for i in range(2, len(df)-2):
        if df["High"].iloc[i] > df["High"].iloc[i-1] and df["High"].iloc[i] > df["High"].iloc[i-2] \
        and df["High"].iloc[i] > df["High"].iloc[i+1] and df["High"].iloc[i] > df["High"].iloc[i+2]:
            df.iloc[i, df.columns.get_loc("fractal_high")] = True
            
        if df["Low"].iloc[i] < df["Low"].iloc[i-1] and df["Low"].iloc[i] < df["Low"].iloc[i-2] \
        and df["Low"].iloc[i] < df["Low"].iloc[i+1] and df["Low"].iloc[i] < df["Low"].iloc[i+2]:
            df.iloc[i, df.columns.get_loc("fractal_low")] = True
            
    return df