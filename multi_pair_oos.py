"""
Multi-Pair Out-of-Sample Backtest
=================================
Combina datos de múltiples pares para entrenar un modelo más robusto,
luego backtestea out-of-sample en cada par individualmente.

Mejoras v3:
  - Multi-pair training (5 pares combinados)
  - Modelo SELL dedicado mejorado
  - Drawdown breaker relajado a 25%
  - Features adicionales: momentum, trend strength
  - Normalización completa para ser pair-agnostic

Uso:
    python multi_pair_oos.py --split-year 2021
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
import joblib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from features import create_features
from fractals import detect_fractals
from ml_dataset import label_data
from train import get_feature_columns
from costs import TradingCosts
from backtest import run_backtest

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report


# Pares disponibles  (archivo, spread_pips, pip_value)
PAIRS = {
    "EURUSD": {"file": "EURUSD_daily_2010_2026.csv", "spread": 1.0, "pip": 0.0001},
    "GBPUSD": {"file": "GBPUSD_daily_2010_2026.csv", "spread": 1.2, "pip": 0.0001},
    "NZDUSD": {"file": "NZDUSD_daily_2016_2026.csv", "spread": 1.5, "pip": 0.0001},
    "AUDUSD": {"file": "AUDUSD_daily_2010_2026.csv", "spread": 1.2, "pip": 0.0001},
    "USDCAD": {"file": "USDCAD_daily_2010_2026.csv", "spread": 1.5, "pip": 0.0001},
    "USDCHF": {"file": "USDCHF_daily_2010_2026.csv", "spread": 1.5, "pip": 0.0001},
    "EURGBP": {"file": "EURGBP_daily_2010_2026.csv", "spread": 1.5, "pip": 0.0001},
    "EURJPY": {"file": "EURJPY_daily_2010_2026.csv", "spread": 1.5, "pip": 0.01, "skip_backtest": True},
    "GBPJPY": {"file": "GBPJPY_daily_2010_2026.csv", "spread": 2.0, "pip": 0.01, "skip_backtest": True},
    "AUDJPY": {"file": "AUDJPY_daily_2010_2026.csv", "spread": 2.0, "pip": 0.01, "skip_backtest": True},
    "CADJPY": {"file": "CADJPY_daily_2010_2026.csv", "spread": 2.0, "pip": 0.01, "skip_backtest": True},
    "NZDJPY": {"file": "NZDJPY_daily_2010_2026.csv", "spread": 2.5, "pip": 0.01, "skip_backtest": True},
    "CHFJPY": {"file": "CHFJPY_daily_2010_2026.csv", "spread": 2.5, "pip": 0.01, "skip_backtest": True},
    "EURAUD": {"file": "EURAUD_daily_2010_2026.csv", "spread": 2.0, "pip": 0.0001},
    "GBPAUD": {"file": "GBPAUD_daily_2010_2026.csv", "spread": 2.5, "pip": 0.0001},
    "EURNZD": {"file": "EURNZD_daily_2010_2026.csv", "spread": 3.0, "pip": 0.0001},
    "GBPNZD": {"file": "GBPNZD_daily_2010_2026.csv", "spread": 3.5, "pip": 0.0001},
    "AUDNZD": {"file": "AUDNZD_daily_2010_2026.csv", "spread": 2.5, "pip": 0.0001},
    "EURCAD": {"file": "EURCAD_daily_2010_2026.csv", "spread": 2.5, "pip": 0.0001},
    "GBPCAD": {"file": "GBPCAD_daily_2010_2026.csv", "spread": 3.0, "pip": 0.0001},
}


def prepare_pair_data(pair_name, config, split_year, lookahead=20, rr=1.5):
    """Carga y prepara datos de un par, split por fecha."""
    path = config["file"]
    if not os.path.exists(path):
        print(f"  [!] Archivo no encontrado: {path}")
        return None, None

    df_raw = pd.read_csv(path)

    # Estimar fechas
    total_rows = len(df_raw)
    # Detectar año de inicio por el nombre del archivo
    if "2010" in path:
        start_date = pd.Timestamp("2010-01-04")
    elif "2016" in path:
        start_date = pd.Timestamp("2016-01-04")
    else:
        start_date = pd.Timestamp("2010-01-04")

    dates = pd.bdate_range(start=start_date, periods=total_rows, freq="B")
    df_raw["_date"] = dates[:total_rows]

    # Split
    split_date = pd.Timestamp(f"{split_year}-01-01")
    df_train_raw = df_raw[df_raw["_date"] < split_date].copy().reset_index(drop=True)
    df_test_raw = df_raw[df_raw["_date"] >= split_date].copy().reset_index(drop=True)

    # Aplicar features
    for subset in [df_train_raw, df_test_raw]:
        if len(subset) < 60:
            continue
        tmp = subset.drop(columns=["_date"])
        tmp = create_features(tmp)
        tmp = detect_fractals(tmp)
        tmp = label_data(tmp, lookahead=lookahead, rr=rr)
        for col in tmp.columns:
            if col not in subset.columns:
                subset[col] = tmp[col].values

    return df_train_raw, df_test_raw


def run_multi_pair_oos(split_year=2021, rr=1.5, lookahead=20,
                       risk_per_trade=0.005, initial_balance=10000,
                       max_drawdown_pct=25.0):
    """
    Pipeline multi-par:
    1. Carga y prepara datos de 5 pares
    2. Combina train data de todos los pares
    3. Entrena modelo BUY (multiclass) y SELL (binario)
    4. Backtestea OOS en cada par individualmente
    """
    print("=" * 70)
    print(f"  MULTI-PAIR OUT-OF-SAMPLE BACKTEST")
    print(f"  Train: < {split_year} | Test: >= {split_year}")
    print(f"  Pares: {', '.join(PAIRS.keys())}")
    print(f"  Risk: {risk_per_trade*100:.1f}% | Max DD: {max_drawdown_pct}%")
    print("=" * 70)

    # --- 1. Cargar todos los pares ---
    print(f"\n[1/4] Cargando datos de {len(PAIRS)} pares...")
    all_train = []
    pair_test_data = {}
    feature_cols = None

    for pair, config in PAIRS.items():
        df_train, df_test = prepare_pair_data(pair, config, split_year, lookahead, rr)

        if df_train is None:
            continue

        # Obtener feature columns del primer par
        train_feats = df_train.drop(columns=["_date"], errors="ignore")
        if feature_cols is None:
            feature_cols = get_feature_columns(train_feats)

        # Limpiar NaN en features
        train_clean = train_feats.dropna(subset=feature_cols).reset_index(drop=True)
        test_feats = df_test.drop(columns=["_date"], errors="ignore")
        test_clean = test_feats.dropna(subset=feature_cols).reset_index(drop=True)

        all_train.append(train_clean)
        pair_test_data[pair] = test_clean

        n_train = len(train_clean)
        n_test = len(test_clean)
        counts = train_clean["target"].value_counts().sort_index()
        labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
        dist = " | ".join([f"{labels[k]}: {v}" for k, v in counts.items()])
        print(f"  {pair}: Train {n_train} | Test {n_test} | {dist}")

    # Combinar todos los train data
    combined_train = pd.concat(all_train, ignore_index=True)
    print(f"\n  TOTAL COMBINADO: {len(combined_train)} velas para entrenamiento")

    counts_all = combined_train["target"].value_counts().sort_index()
    labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
    for val, count in counts_all.items():
        print(f"    {labels[val]}: {count} ({count/len(combined_train)*100:.1f}%)")

    # --- 2. Walk-forward en datos combinados ---
    print(f"\n[2/4] Walk-forward validation en datos combinados...")
    costs = TradingCosts(spread_pips=1.2)  # Average spread

    total = len(combined_train)
    fold_results = []

    for fold in range(4):
        train_end = int(total * (0.6 + fold * 0.1))
        test_end = int(total * (0.7 + fold * 0.1))
        test_end = min(test_end, total)

        df_tr = combined_train.iloc[:train_end]
        df_te = combined_train.iloc[train_end:test_end]

        X_tr = df_tr[feature_cols].values
        y_tr = df_tr["target"].values
        X_te = df_te[feature_cols].values
        y_te = df_te["target"].values

        # Usar GradientBoosting en vez de RandomForest — mejor generalización
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        # Threshold optimization
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
        best_th = 0.5
        best_exp = -999

        for th in thresholds:
            max_probs = y_proba.max(axis=1)
            mask = (y_pred != 0) & (max_probs >= th)
            n_trades = mask.sum()

            if n_trades < 30:
                continue

            # Simulate PnL
            wins = losses = 0
            total_win = total_loss = 0.0

            for idx in np.where(mask)[0]:
                actual = y_te[idx]
                pred = y_pred[idx]
                pw = df_te.iloc[idx]["potential_win"]
                pl = df_te.iloc[idx]["potential_loss"]

                if pred == actual and actual != 0:
                    pnl = costs.apply_to_pnl(pw)
                    wins += 1
                    total_win += pnl
                else:
                    pnl = costs.apply_to_pnl(-pl)
                    losses += 1
                    total_loss += abs(pnl)

            wr = wins / n_trades if n_trades > 0 else 0
            avg_w = total_win / wins if wins > 0 else 0
            avg_l = total_loss / losses if losses > 0 else 0
            exp = (wr * avg_w) - ((1 - wr) * avg_l)
            pf = total_win / total_loss if total_loss > 0 else float("inf")

            if exp > best_exp:
                best_exp = exp
                best_th = th

        fold_results.append({"fold": fold + 1, "best_th": best_th, "best_exp": best_exp})

        status = f"Exp: {best_exp:.6f} th={best_th}" if best_exp > -999 else "INVALID"
        print(f"  Fold {fold+1}: Train {len(df_tr)} | Test {len(df_te)} → {status}")

    # Optimal threshold
    valid_folds = [f for f in fold_results if f["best_exp"] > -999]
    if valid_folds:
        ths = [f["best_th"] for f in valid_folds]
        optimal_th = max(set(ths), key=ths.count)
        avg_exp = np.mean([f["best_exp"] for f in valid_folds])
    else:
        optimal_th = 0.5
        avg_exp = 0

    print(f"\n  Threshold óptimo: {optimal_th}")
    print(f"  Avg Expectancy WF: {avg_exp:.6f}")

    # --- 3. Entrenar modelos finales ---
    print(f"\n[3/4] Entrenando modelos finales con {len(combined_train)} filas...")

    # Modelo principal (BUY + SELL + HOLD)
    X_all = combined_train[feature_cols].values
    y_all = combined_train["target"].values

    main_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=30,
        subsample=0.8,
        random_state=42,
    )
    main_model.fit(X_all, y_all)

    model_path = "model_multi.joblib"
    joblib.dump({
        "model": main_model,
        "feature_columns": feature_cols,
        "threshold": optimal_th,
    }, model_path)

    # Modelo SELL separado (binario)
    y_sell = (combined_train["target"] == 2).astype(int).values
    sell_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=30,
        subsample=0.8,
        random_state=42,
    )
    sell_model.fit(X_all, y_sell)

    sell_model_path = "model_multi_sell.joblib"
    joblib.dump({
        "model": sell_model,
        "feature_columns": feature_cols,
        "threshold": optimal_th,
    }, sell_model_path)

    n_sell = y_sell.sum()
    print(f"  Modelo principal: {model_path}")
    print(f"  Modelo SELL:      {sell_model_path} ({n_sell} ejemplos SELL)")

    # Feature importance
    importances = pd.Series(main_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print(f"\n  Top 10 features (modelo combinado):")
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 100)
        print(f"    {feat:25s} {imp:.4f} {bar}")

    # --- 4. Backtest OOS en cada par ---
    print(f"\n{'='*70}")
    print(f"  [4/4] BACKTEST OUT-OF-SAMPLE POR PAR ({split_year}–2026)")
    print(f"{'='*70}")

    all_results = {}

    for pair, config in PAIRS.items():
        if pair not in pair_test_data:
            continue

        if config.get("skip_backtest", False):
            print(f"\n  {pair}: Solo usado para entrenamiento (skip_backtest=True)")
            continue

        df_test = pair_test_data[pair]
        if len(df_test) < 50:
            print(f"\n  {pair}: datos insuficientes ({len(df_test)} velas)")
            continue

        print(f"\n{'─'*70}")
        print(f"  {pair} — {len(df_test)} velas OOS")
        print(f"{'─'*70}")

        # Guardar temp
        test_csv = f"_oos_{pair}.csv"
        df_test.to_csv(test_csv, index=False)

        results = run_backtest(
            data_path=test_csv,
            model_path=model_path,
            sell_model_path=sell_model_path,
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            rr_ratio=rr,
            max_drawdown_pct=max_drawdown_pct,
            spread_pips=config["spread"],
            pip_value=config["pip"],
        )

        all_results[pair] = results

        if os.path.exists(test_csv):
            os.remove(test_csv)

    # --- Resumen final ---
    print(f"\n{'='*70}")
    print(f"  RESUMEN — MULTI-PAIR OOS ({split_year}–2026)")
    print(f"{'='*70}")
    print(f"  {'Par':<10} {'Trades':>7} {'Win%':>7} {'Expect':>10} {'PF':>8} {'MaxDD':>8} {'B/S':>8}")
    print(f"  {'-'*60}")

    total_trades = 0
    profitable_pairs = 0

    for pair, res in all_results.items():
        m = res.get("metrics")
        if m:
            n = m["n_trades"]
            wr = m["win_rate"] * 100
            exp = m["expectancy"]
            pf = m["profit_factor"]
            dd = m["max_drawdown_pct"]
            buys = m.get("n_buys", 0)
            sells = m.get("n_sells", 0)

            status = "✅" if exp > 0 else "❌"
            print(f"  {status} {pair:<8} {n:>7} {wr:>6.1f}% {exp:>10.6f} {pf:>8.4f} {dd:>7.1f}% {buys}/{sells}")

            total_trades += n
            if exp > 0:
                profitable_pairs += 1
        else:
            print(f"  ⚠️  {pair:<8} — Sin datos suficientes")

    print(f"\n  Total trades OOS:    {total_trades}")
    print(f"  Pares rentables:     {profitable_pairs}/{len(all_results)}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Pair OOS Backtest")
    parser.add_argument("--split-year", type=int, default=2021, help="Año de corte")
    parser.add_argument("--rr", type=float, default=1.5, help="R:R ratio")
    parser.add_argument("--balance", type=float, default=10000, help="Capital inicial")
    parser.add_argument("--risk", type=float, default=0.005, help="Riesgo por trade")
    parser.add_argument("--max-dd", type=float, default=25.0, help="Max drawdown %%")

    args = parser.parse_args()

    run_multi_pair_oos(
        split_year=args.split_year,
        rr=args.rr,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        max_drawdown_pct=args.max_dd,
    )
