"""CLI: walk-forward backtest a completed season and report accuracy vs. baselines. Logic lives in fpl_agent.models.evaluate."""

from __future__ import annotations

import pandas as pd

from fpl_agent.models.evaluate import (
    MIN_EVAL_GW,
    accuracy_report,
    build_fixed_squad,
    gameweek_points_with_captain,
    naive_prior_points,
    recency_split,
    simulate_cumulative_points,
    target_season_player_pool,
    walk_forward_backtest,
)
from fpl_agent.storage.repository import PROCESSED_DIR

if __name__ == "__main__":
    import sys

    target_season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    training_table = pd.read_parquet(PROCESSED_DIR / "training_table.parquet")

    print(f"Running walk-forward backtest for {target_season} (from GW{MIN_EVAL_GW})...")
    predictions = walk_forward_backtest(training_table, target_season)

    print("\nAccuracy (all evaluated gameweeks):")
    for label, metrics in accuracy_report(predictions).items():
        print(f"  {label}: MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  rank_corr={metrics['mean_rank_correlation']:.3f}")

    pool = target_season_player_pool(training_table, target_season, predictions["GW"].min())
    cumulative = simulate_cumulative_points(predictions, pool)
    print("\nCumulative points (fixed squad, no transfers, captained weekly by each strategy):")
    for label, points in cumulative.items():
        print(f"  {label}: {points:.0f}")
