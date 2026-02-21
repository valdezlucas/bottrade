import pandas as pd
from features import create_features

def load_data(path):
    df = pd.read_csv(path)
    df = create_features(df)
    return df