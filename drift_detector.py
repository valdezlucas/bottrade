import numpy as np
import pandas as pd
from scipy.stats import entropy

def hist_probs(series, bins=50):
    counts, _ = np.histogram(series, bins=bins, density=True)
    probs = counts + 1e-12
    return probs / probs.sum()

def kl_divergence(p_series, ref_series, bins=50):
    p = hist_probs(p_series, bins)
    q = hist_probs(ref_series, bins)
    return entropy(p, q)  # KL(p || q)

def rolling_kl(df_features, ref_period_end_idx, window=250, bins=50, threshold=0.2):
    """
    df_features: DataFrame index=timestamp cols=features
    ref_period_end_idx: index position (int) marking end of reference window (e.g., end of validation)
    window: window size to compute rolling divergence
    threshold: alert threshold
    returns DataFrame of KL per feature and alerts
    """
    ref = df_features.iloc[:ref_period_end_idx]
    kl_df = pd.DataFrame(index=df_features.index[ref_period_end_idx+window-1:], columns=df_features.columns, dtype=float)

    print(f"Calculando Divergencia KL sobre {len(df_features.columns)} features...")
    for i in range(ref_period_end_idx+window-1, len(df_features)):
        window_slice = df_features.iloc[i-window+1:i+1]
        for col in df_features.columns:
            kl = kl_divergence(window_slice[col], ref[col], bins=bins)
            kl_df.iloc[i - (ref_period_end_idx+window-1), kl_df.columns.get_loc(col)] = kl

    alerts = (kl_df > threshold).any(axis=1)
    return kl_df, alerts

if __name__ == "__main__":
    import joblib
    try:
        # Ejemplo de test
        print("Cargando features guardadas en modelo para test de Drift...")
        buy_art = joblib.load('model_multi.joblib')
        features = buy_art['feature_columns']
        
        df = pd.read_csv('df_1d_joined.csv')
        df = df.dropna(subset=features)
        
        # Simulamos que los últimos N dias (~2024-2025) son OOS
        split_idx = int(len(df) * 0.8) # 80% REF, 20% TEST
        df_feat = df[features]
        
        # Test KL rápido (solo 5 ventanas) para validar el script
        test_df = pd.concat([df_feat.iloc[:split_idx], df_feat.iloc[split_idx:split_idx+255]])
        kl_df, alerts = rolling_kl(test_df, ref_period_end_idx=split_idx, window=250, threshold=0.2)
        
        latest_kl = kl_df.iloc[-1].sort_values(ascending=False).head(5)
        print("\nTop 5 features con mayor KL divergence al final del test:")
        print(latest_kl)
        print(f"\nAlerta general de Drift activada: {alerts.iloc[-1]}")
    except Exception as e:
        print(f"No se pudo correr el test de ejemplo: {e}")
