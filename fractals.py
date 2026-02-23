import pandas as pd

def detect_fractals(df):
    df["fractal_high"] = False
    df["fractal_low"] = False
    
    # A fractal high is formed when the high of bar i is strictly greater than 
    # the highs of i-2, i-1, i+1, and i+2.
    # CRITICAL: We only KNOW it formed at the close of bar i+2. 
    # Therefore, the signal must be attributed to bar i+2.
    
    for i in range(2, len(df)-2):
        if df["High"].iloc[i] > df["High"].iloc[i-1] and df["High"].iloc[i] > df["High"].iloc[i-2] \
        and df["High"].iloc[i] > df["High"].iloc[i+1] and df["High"].iloc[i] > df["High"].iloc[i+2]:
            df.iloc[i+2, df.columns.get_loc("fractal_high")] = True
            
        if df["Low"].iloc[i] < df["Low"].iloc[i-1] and df["Low"].iloc[i] < df["Low"].iloc[i-2] \
        and df["Low"].iloc[i] < df["Low"].iloc[i+1] and df["Low"].iloc[i] < df["Low"].iloc[i+2]:
            df.iloc[i+2, df.columns.get_loc("fractal_low")] = True
            
    return df