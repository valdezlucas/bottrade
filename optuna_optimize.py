"""
Optuna Bayesian Optimization for Trading Pipeline - GPU ENABLED
=============================================================
Versión optimizada para Rei: Menos carga de CPU, uso de GPU en LightGBM.
"""

import argparse
import os
import numpy as np
import pandas as pd
import optuna
import joblib
import warnings
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("⚠️ LightGBM no está instalado. Ejecutar: pip install lightgbm")

warnings.filterwarnings('ignore')

EXCLUDE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "target", "potential_win", "potential_loss",
    "fractal_high", "fractal_low",
    "Datetime", "Date",
]

def apply_stochastic_pnl(raw_pnl_usd, risk_usd):
    slippage = np.random.normal(0.10 * risk_usd, 0.05 * risk_usd)
    return raw_pnl_usd - slippage

def fast_simulate(preds, probas, targets, buy_th, sell_th):
    base_risk_pct = 0.005 
    max_risk_pct = 0.01   
    alpha = 0.5
    balance = 10000.0
    
    trades = []
    equity = [balance]
    peak = balance
    max_dd = 0.0
    
    for i in range(len(preds)):
        prob_buy = probas[i, 1]
        prob_sell = probas[i, 2]
        
        is_buy = prob_buy >= buy_th
        is_sell = prob_sell >= sell_th
        
        if not is_buy and not is_sell:
            continue
            
        if is_buy and not is_sell:
            dir_pred, conf, th = 1, prob_buy, buy_th
        elif is_sell and not is_buy:
            dir_pred, conf, th = 2, prob_sell, sell_th
        else:
            if prob_buy - buy_th > prob_sell - sell_th:
                dir_pred, conf, th = 1, prob_buy, buy_th
            else:
                dir_pred, conf, th = 2, prob_sell, sell_th
                
        if conf > th and th < 1.0:
            scaled = base_risk_pct * ((conf - th) / (1.0 - th)) * alpha + base_risk_pct
        else:
            scaled = base_risk_pct
            
        adj_risk = min(scaled, max_risk_pct)
        risk_usd = balance * adj_risk
        
        if dir_pred == targets[i]:
            raw_pnl = risk_usd * 1.5 
        else:
            raw_pnl = -risk_usd
            
        pnl_net = apply_stochastic_pnl(raw_pnl, risk_usd) - 7.0
        balance += pnl_net
        equity.append(balance)
        
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak
        if dd > max_dd:
            max_dd = dd
        trades.append(pnl_net)
        
    expectancy = np.mean(trades) if trades else 0
    gross_profits = sum(t for t in trades if t > 0)
    gross_losses = abs(sum(t for t in trades if t <= 0))
    pf = gross_profits / gross_losses if gross_losses > 0 else 99.0
    
    returns = np.diff(equity) / equity[:-1]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 and np.std(returns) > 0 else 0
    
    return {
        "trades_count": len(trades),
        "expectancy_per_trade": expectancy,
        "max_drawdown_pct": max_dd * 100,
        "pf": pf,
        "sharpe": sharpe,
        "equity_curve": equity
    }

class TradingObjective:
    def __init__(self, df_train, df_valid, feature_cols, seed):
        self.df_train = df_train
        self.df_valid = df_valid
        self.seed = seed
        self.feature_cols = feature_cols
        self.X_train_full = self.df_train[self.feature_cols].values
        self.y_train_full = self.df_train['target'].values
        self.df_valid = self.df_valid.sort_values("Datetime").reset_index(drop=True)
        self.X_valid_full = self.df_valid[self.feature_cols].values
        self.y_valid_full = self.df_valid['target'].values

    def __call__(self, trial):
        # Mantenemos ambos, pero bajamos n_jobs para que no trabe la PC
        model_type = trial.suggest_categorical("model_type", ["RandomForest", "LightGBM"])
        
        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("rf_n_estimators", 50, 250),
                max_depth=trial.suggest_int("rf_max_depth", 3, 12),
                max_features="sqrt",
                min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 20, 100),
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=2 # <--- NO USAR -1 PARA NO CONGELAR PC
            )
        else:
            model = lgb.LGBMClassifier(
                device="gpu",           # <--- USA TU PLACA DE VIDEO
                gpu_platform_id=0,
                gpu_device_id=0,
                learning_rate=trial.suggest_float("lgb_learning_rate", 1e-3, 0.05, log=True),
                n_estimators=trial.suggest_int("lgb_n_estimators", 50, 250),
                num_leaves=trial.suggest_int("lgb_num_leaves", 20, 80),
                min_data_in_leaf=trial.suggest_int("lgb_min_data_in_leaf", 20, 100),
                feature_fraction=trial.suggest_float("lgb_feature_fraction", 0.5, 0.9),
                bagging_fraction=trial.suggest_float("lgb_bagging_fraction", 0.5, 0.9),
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=2,               # <--- DEJA NÚCLEOS LIBRES PARA WINDOWS
                verbose=-1
            )
            
        buy_th = trial.suggest_float("buy_threshold", 0.48, 0.55)
        sell_th = trial.suggest_float("sell_threshold", 0.48, 0.55)
        
        tscv = TimeSeriesSplit(n_splits=3)
        fold_scores = []
        fold_metrics = []
        
        for fold, (val_train_idx, val_test_idx) in enumerate(tscv.split(self.X_valid_full)):
            X_train_fold = np.vstack([self.X_train_full, self.X_valid_full[val_train_idx]])
            y_train_fold = np.concatenate([self.y_train_full, self.y_valid_full[val_train_idx]])
            X_test_fold = self.X_valid_full[val_test_idx]
            y_test_fold = self.y_valid_full[val_test_idx]
            
            # Calibración Isotonic (Consistente con main.py)
            calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated.fit(X_train_fold, y_train_fold)
            
            probas = calibrated.predict_proba(X_test_fold)
            preds = calibrated.predict(X_test_fold)
            
            sim_metrics = fast_simulate(preds, probas, y_test_fold, buy_th, sell_th)
            
            if sim_metrics["trades_count"] < 5:
                obj_score = -5000.0
            else:
                # Score: Expectativa - penalización por Drawdown
                obj_score = sim_metrics["expectancy_per_trade"] - (0.3 * sim_metrics["max_drawdown_pct"])
                
            fold_scores.append(obj_score)
            fold_metrics.append(sim_metrics)
            
            trial.report(obj_score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(fold_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_features_train", type=str, required=True)
    parser.add_argument("--path_features_valid", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="optuna_results")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    df_train = pd.read_csv(args.path_features_train)
    df_valid = pd.read_csv(args.path_features_valid)
    
    if 'Datetime' in df_train.columns:
        df_train['Datetime'] = pd.to_datetime(df_train['Datetime'])
        df_train = df_train[df_train['Datetime'].dt.year <= 2018]
        
    if 'Datetime' in df_valid.columns:
        df_valid['Datetime'] = pd.to_datetime(df_valid['Datetime'])
        df_valid = df_valid[(df_valid['Datetime'].dt.year >= 2019) & (df_valid['Datetime'].dt.year <= 2021)]
        
    feature_cols = [c for c in df_train.columns if c not in EXCLUDE_COLS]
    df_train = df_train.dropna(subset=feature_cols + ['target'])
    df_valid = df_valid.dropna(subset=feature_cols + ['target'])
    
    # --- PERSISTENCIA: Crea archivo .db para no perder progreso ---
    study_name = "trading_optimization_rei"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True,
        direction="maximize", 
        sampler=TPESampler(seed=args.seed), 
        pruner=MedianPruner(n_startup_trials=10)
    )
    
    print(f"🚀 Iniciando/Retomando Optuna en {storage_name}")
    objective = TradingObjective(df_train, df_valid, feature_cols, args.seed)
    study.optimize(objective, n_trials=args.n_trials)
    
    # (Resto del código de guardado de resultados igual al original...)
    print(f"🏆 Mejor Score: {study.best_trial.value:.4f}")
    joblib.dump(study.best_params, os.path.join(args.out_dir, "best_params.joblib"))

if __name__ == "__main__":
    main()