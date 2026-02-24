import numpy as np
import pandas as pd


def mc_bootstrap(trades_df, n_runs=10000, seed=42):
    """
    trades_df: DataFrame con columnas ['pnl_usd']
    """
    if "pnl_usd" not in trades_df.columns:
        raise ValueError("El DataFrame debe contener la columna 'pnl_usd'")

    rng = np.random.default_rng(seed)
    results = []
    initial_balance = 10000

    for _ in range(n_runs):
        sample = rng.choice(
            trades_df["pnl_usd"].values, size=len(trades_df), replace=True
        )
        equity = np.cumsum(sample) + initial_balance
        results.append(
            {
                "total_pnl": equity[-1] - initial_balance,
                "max_dd": (np.maximum.accumulate(equity) - equity).max(),
                "min_equity": equity.min(),
            }
        )

    res_df = pd.DataFrame(results)
    out = {
        "total_pnl_mean": res_df["total_pnl"].mean(),
        "total_pnl_median": res_df["total_pnl"].median(),
        "total_pnl_p1": res_df["total_pnl"].quantile(0.01),
        "total_pnl_p5": res_df["total_pnl"].quantile(0.05),
        "total_pnl_p95": res_df["total_pnl"].quantile(0.95),
        "max_dd_mean": res_df["max_dd"].mean(),
        "max_dd_p95": res_df["max_dd"].quantile(0.95),
        "max_dd_p99": res_df["max_dd"].quantile(0.99),
        "prob_ruina_20pct": (res_df["min_equity"] < (initial_balance * 0.8)).mean()
        * 100,
        "prob_ruina_50pct": (res_df["min_equity"] < (initial_balance * 0.5)).mean()
        * 100,
    }
    return out


if __name__ == "__main__":
    try:
        df = pd.read_csv("oos_trades.csv")
        print("📊 Ejecutando Monte Carlo Bootstrap (10,000 runs)...")
        print(f"   Trades originales: {len(df)}")
        metrics = mc_bootstrap(df, n_runs=10000)

        print(f"\n{'='*50}")
        print(f"  RESULTADOS MONTE CARLO (10k) - USD Base: $10,000")
        print(f"{'='*50}")
        for k, v in metrics.items():
            if "prob" in k:
                print(f"   {k:>20}: {v:.2f}%")
            else:
                print(f"   {k:>20}: ${v:.2f}")
    except FileNotFoundError:
        print(
            "❌ Archivo oos_trades.csv no encontrado. Ejecuta true_oos_test.py primero."
        )
