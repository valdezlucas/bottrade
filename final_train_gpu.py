import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

# 1. Cargar Datos
df = pd.read_csv("df_1d_joined.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])

# 2. Filtro Temporal (Entrenamos con todo hasta el cierre de 2021)
df_final = df[df['Datetime'].dt.year <= 2021].copy()

EXCLUDE_COLS = ["Open", "High", "Low", "Close", "Volume", "target", "potential_win", "potential_loss", "fractal_high", "fractal_low", "Datetime", "Date", "Asset"]
feature_cols = [c for c in df_final.columns if c not in EXCLUDE_COLS]

X = df_final[feature_cols].values
y = df_final['target'].values

print(f"🚀 Entrenando modelo final CAMPEÓN (Trial 67) con {len(df_final)} filas...")

# 3. Configuración del Modelo Ganador (Trial 67)
best_lgbm = lgb.LGBMClassifier(
    device="gpu",           # <--- USANDO TU GPU
    gpu_platform_id=0,
    gpu_device_id=0,
    learning_rate=0.061024,
    n_estimators=155,
    num_leaves=79,
    min_data_in_leaf=100,
    feature_fraction=0.8382,
    bagging_fraction=0.6932,
    lambda_l1=3.1021e-07,
    lambda_l2=9.3739e-07,
    class_weight="balanced",
    random_state=42,
    verbose=-1
)

# 4. Calibración (Indispensable para usar los thresholds de Optuna)
model_final = CalibratedClassifierCV(best_lgbm, method='isotonic', cv=3)
model_final.fit(X, y)

# 5. Guardar
joblib.dump(model_final, "modelo_final_trading.joblib")
joblib.dump(feature_cols, "feature_cols.joblib")

print("\n✅ ¡Modelo guardado con éxito!")
print(f"📊 Thresholds para OOS: Buy {0.4702} | Sell {0.5080}")