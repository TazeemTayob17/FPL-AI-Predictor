"""Checks rolling-window features use only strictly-prior games and never leak the current or future row."""

import pandas as pd
import pytest

from fpl_agent.features.rolling_stats import add_minutes_volatility, add_rolling_features

DATA = pd.DataFrame(
    {
        "season": ["s"] * 8,
        "element": [1, 1, 1, 1, 2, 2, 2, 2],
        "GW": [1, 2, 3, 4, 1, 2, 3, 4],
        "total_points": [2, 4, 6, 8, 10, 10, 10, 10],
    }
)


def test_first_game_has_no_rolling_value():
    """A player's very first game has no prior games, so every rolling column must be NaN, not 0."""
    result = add_rolling_features(DATA, group_keys=["season", "element"], stat_columns=("total_points",))
    first_games = result[result["GW"] == 1]
    assert first_games["total_points_roll3"].isna().all()


def test_rolling_mean_uses_only_prior_games():
    """Player 1's GW4 roll3 must be mean(2,4,6)=4.0 - the prior three games, excluding GW4's own value of 8."""
    result = add_rolling_features(DATA, group_keys=["season", "element"], stat_columns=("total_points",))
    row = result[(result["element"] == 1) & (result["GW"] == 4)].iloc[0]
    assert row["total_points_roll3"] == 4.0


def test_rolling_stats_do_not_cross_player_boundaries():
    """Player 2's rolling stats must never be influenced by player 1's values."""
    result = add_rolling_features(DATA, group_keys=["season", "element"], stat_columns=("total_points",))
    row = result[(result["element"] == 2) & (result["GW"] == 4)].iloc[0]
    assert row["total_points_roll3"] == 10.0


def test_missing_stat_columns_are_skipped_without_error():
    """A requested stat column absent from the input frame is silently skipped, not a KeyError."""
    result = add_rolling_features(DATA, group_keys=["season", "element"], stat_columns=("total_points", "does_not_exist"))
    assert "does_not_exist_roll3" not in result.columns


def test_minutes_volatility_matches_the_standard_deviation_of_prior_games():
    """A player alternating 90/0/90 minutes should show GW4 volatility equal to std([90, 0, 90]), not including GW4 itself."""
    data = pd.DataFrame({"season": ["s"] * 4, "element": [1] * 4, "GW": [1, 2, 3, 4], "minutes": [90, 0, 90, 0]})
    result = add_minutes_volatility(data, group_keys=["season", "element"], window=10)
    row = result[result["GW"] == 4].iloc[0]
    assert row["minutes_volatility"] == pytest.approx(pd.Series([90, 0, 90]).std())


def test_minutes_volatility_is_zero_for_a_consistent_starter():
    """A player who always plays 90 minutes must show zero volatility, not NaN or a spurious nonzero value."""
    data = pd.DataFrame({"season": ["s"] * 4, "element": [1] * 4, "GW": [1, 2, 3, 4], "minutes": [90, 90, 90, 90]})
    result = add_minutes_volatility(data, group_keys=["season", "element"], window=10)
    assert (result["minutes_volatility"] == 0.0).all()
