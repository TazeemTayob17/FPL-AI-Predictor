"""Checks per-fixture rows collapse correctly into one row per player per gameweek, including double-gameweek summing."""

import pandas as pd

from fpl_agent.features.build_features import collapse_to_gameweek

DOUBLE_GAMEWEEK_ROWS = pd.DataFrame(
    {
        "season": ["s", "s"],
        "element": [1, 1],
        "GW": [5, 5],
        "name": ["Player One", "Player One"],
        "position": ["GK", "GK"],
        "team": ["Arsenal", "Arsenal"],
        "value": [50, 51],
        "total_points": [4, 6],
        "minutes": [90, 90],
    }
)


def test_double_gameweek_points_are_summed_not_duplicated():
    """A player with two fixtures in one gameweek must produce ONE row with points summed (4+6=10), not two rows."""
    result = collapse_to_gameweek(DOUBLE_GAMEWEEK_ROWS)
    assert len(result) == 1
    assert result.iloc[0]["total_points"] == 10
    assert result.iloc[0]["minutes"] == 180


def test_double_gameweek_takes_the_latest_value_not_the_sum():
    """Price (value) is a snapshot, not additive - the second fixture's value (51) should win, not 50+51."""
    result = collapse_to_gameweek(DOUBLE_GAMEWEEK_ROWS)
    assert result.iloc[0]["value"] == 51


def test_position_labels_are_normalized_across_vaastav_seasons():
    """vaastav uses "GK" (not FPL's own "GKP") and a stray "AM" label in one season - both must normalize."""
    rows = pd.DataFrame(
        {
            "season": ["s", "s"], "element": [1, 2], "GW": [1, 1],
            "name": ["A", "B"], "position": ["GK", "AM"], "team": ["X", "Y"],
            "total_points": [2, 3], "minutes": [90, 90],
        }
    )
    result = collapse_to_gameweek(rows)
    assert set(result["position"]) == {"GKP", "MID"}


def test_collapses_a_live_element_summary_shaped_frame_with_no_name_position_or_team_columns():
    """A live element-summary pull has no name/position/team columns - only build_current_features's later merge adds them."""
    rows = pd.DataFrame(
        {"season": ["current", "current"], "element": [1, 1], "GW": [5, 5], "total_points": [4, 6], "minutes": [90, 90]}
    )
    result = collapse_to_gameweek(rows)
    assert len(result) == 1
    assert result.iloc[0]["total_points"] == 10
    assert "position" not in result.columns
