import joblib
import pandas as pd

from data import load_data
from fractals import detect_fractals


def load_model(model_path="model.joblib"):
    """Carga el modelo entrenado y su metadata."""
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["feature_columns"], artifact["threshold"]


def predict(path, model_path="model.joblib"):
    """
    Carga datos nuevos, aplica features y genera predicciones.

    Solo devuelve señales BUY/SELL que superen el threshold óptimo.
    """
    model, feature_cols, threshold = load_model(model_path)

    print(f"📂 Cargando datos: {path}")
    df = load_data(path)
    df = detect_fractals(df)

    # Verificar que existan las columnas necesarias
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en los datos: {missing}")

    # Limpiar NaN
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X = df[feature_cols].values
    probas = model.predict_proba(X)
    predictions = probas.argmax(axis=1)
    max_probs = probas.max(axis=1)

    labels_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

    # Filtrar por threshold
    signals = []
    for i in range(len(df)):
        pred = predictions[i]
        prob = max_probs[i]

        if pred != 0 and prob >= threshold:
            signals.append(
                {
                    "index": i,
                    "signal": labels_map[pred],
                    "probability": round(prob, 4),
                    "close": df["Close"].iloc[i],
                }
            )

    print(f"\n🎯 Señales encontradas (threshold={threshold}):")
    print(f"   Total filas analizadas: {len(df)}")
    print(f"   Señales generadas: {len(signals)}")

    if signals:
        print(f"\n   Últimas señales:")
        for s in signals[-10:]:
            emoji = "🟢" if s["signal"] == "BUY" else "🔴"
            print(
                f"   {emoji} [{s['index']}] {s['signal']} @ {s['close']:.5f} "
                f"(prob: {s['probability']:.2%})"
            )

    return signals, df
