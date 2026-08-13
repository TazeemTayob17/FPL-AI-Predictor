"""Computes rolling 3/5/10-gameweek form windows from a per-player-per-gameweek stats table."""

from __future__ import annotations

import pandas as pd

WINDOWS = (3, 5, 10)

ROLLING_STAT_COLUMNS = (
    "total_points", "minutes", "goals_scored", "assists", "bps", "clean_sheets", "saves",
    "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    "defensive_contribution",
)


def add_rolling_features(
    gw_table: pd.DataFrame, group_keys: list[str], stat_columns: tuple[str, ...] = ROLLING_STAT_COLUMNS
) -> pd.DataFrame:
    """Adds a mean-of-last-N-games column per stat/window, using only games strictly before the current row."""
    table = gw_table.sort_values(group_keys + ["GW"]).reset_index(drop=True)
    group_cols = [table[key] for key in group_keys]
    for column in stat_columns:
        if column not in table.columns:
            continue
        shifted = table.groupby(group_keys, sort=False)[column].shift(1)
        for window in WINDOWS:
            rolled = shifted.groupby(group_cols).rolling(window, min_periods=1).mean()
            table[f"{column}_roll{window}"] = rolled.reset_index(level=list(range(len(group_keys))), drop=True)
    return table
