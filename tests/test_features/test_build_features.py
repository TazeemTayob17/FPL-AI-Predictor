# Checks per-fixture rows collapse correctly into one row per player per gameweek, including double-gameweek summing.

import pandas as pd

from fpl_agent.features.build_features import EXPECTED_ROLL_COLUMNS, build_current_features, collapse_to_gameweek

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


# A player with two fixtures in one gameweek must produce ONE row with points summed (4+6=10), not two rows.
def test_double_gameweek_points_are_summed_not_duplicated():
    result = collapse_to_gameweek(DOUBLE_GAMEWEEK_ROWS)
    assert len(result) == 1
    assert result.iloc[0]["total_points"] == 10
    assert result.iloc[0]["minutes"] == 180


# Price (value) is a snapshot, not additive - the second fixture's value (51) should win, not 50+51.
def test_double_gameweek_takes_the_latest_value_not_the_sum():
    result = collapse_to_gameweek(DOUBLE_GAMEWEEK_ROWS)
    assert result.iloc[0]["value"] == 51


# vaastav uses "GK" (not FPL's own "GKP") and a stray "AM" label in one season - both must normalize.
def test_position_labels_are_normalized_across_vaastav_seasons():
    rows = pd.DataFrame(
        {
            "season": ["s", "s"], "element": [1, 2], "GW": [1, 1],
            "name": ["A", "B"], "position": ["GK", "AM"], "team": ["X", "Y"],
            "total_points": [2, 3], "minutes": [90, 90],
        }
    )
    result = collapse_to_gameweek(rows)
    assert set(result["position"]) == {"GKP", "MID"}


# A live element-summary pull has no name/position/team columns - only build_current_features's later merge adds them.
def test_collapses_a_live_element_summary_shaped_frame_with_no_name_position_or_team_columns():
    rows = pd.DataFrame(
        {"season": ["current", "current"], "element": [1, 1], "GW": [5, 5], "total_points": [4, 6], "minutes": [90, 90]}
    )
    result = collapse_to_gameweek(rows)
    assert len(result) == 1
    assert result.iloc[0]["total_points"] == 10
    assert "position" not in result.columns


FIXTURES_CURRENT = pd.DataFrame(
    {"event": [1], "team_h": [1], "team_a": [2], "team_h_difficulty": [3], "team_a_difficulty": [3]}
)
TEAMS_CURRENT = pd.DataFrame(
    {
        "id": [1, 2], "strength_attack_home": [1000, 1000], "strength_attack_away": [1000, 1000],
        "strength_defence_home": [1000, 1000], "strength_defence_away": [1000, 1000],
    }
)


# Regression test: with zero players having any history, roll columns must still exist (as NaN), not vanish.
def test_build_current_features_keeps_every_roll_column_when_no_player_has_any_history_yet():
    players = pd.DataFrame([{"player_id": 1, "team_id": 1, "position": "FWD", "now_cost_million": 8.0}])
    result = build_current_features({1: pd.DataFrame()}, players, FIXTURES_CURRENT, TEAMS_CURRENT, target_gameweek=1)
    for column in EXPECTED_ROLL_COLUMNS:
        assert column in result.columns


# A mixed pool (some players with real history, others brand new) must still carry every roll column for everyone.
def test_build_current_features_keeps_every_roll_column_when_some_players_have_history_and_others_dont():
    players = pd.DataFrame(
        [
            {"player_id": 1, "team_id": 1, "position": "FWD", "now_cost_million": 8.0},
            {"player_id": 2, "team_id": 2, "position": "FWD", "now_cost_million": 5.0},
        ]
    )
    history = pd.DataFrame({"GW": [1], "total_points": [6], "minutes": [90]})
    result = build_current_features({1: history, 2: pd.DataFrame()}, players, FIXTURES_CURRENT, TEAMS_CURRENT, target_gameweek=1)
    for column in EXPECTED_ROLL_COLUMNS:
        assert column in result.columns


# "value" has a live equivalent (now_cost_million), so a player with no history still gets a real value, not NaN.
def test_build_current_features_backfills_value_from_now_cost_million_when_history_has_none():
    players = pd.DataFrame([{"player_id": 1, "team_id": 1, "position": "FWD", "now_cost_million": 8.0}])
    result = build_current_features({1: pd.DataFrame()}, players, FIXTURES_CURRENT, TEAMS_CURRENT, target_gameweek=1)
    assert result.loc[0, "value"] == 80  # tenths of a million, matching vaastav's units


# "selected" has no unit-consistent live equivalent, so it must be left NaN, not substituted with a wrong value.
def test_build_current_features_leaves_selected_missing_rather_than_wrongly_scaled():
    players = pd.DataFrame([{"player_id": 1, "team_id": 1, "position": "FWD", "now_cost_million": 8.0}])
    result = build_current_features({1: pd.DataFrame()}, players, FIXTURES_CURRENT, TEAMS_CURRENT, target_gameweek=1)
    assert "selected" in result.columns
    assert pd.isna(result.loc[0, "selected"])


# Regression test: LightGBM rejects object-dtype columns even when every value is missing; must be numeric NaN.
def test_build_current_features_missing_columns_are_numeric_dtype_not_object():
    players = pd.DataFrame([{"player_id": 1, "team_id": 1, "position": "FWD", "now_cost_million": 8.0}])
    result = build_current_features({1: pd.DataFrame()}, players, FIXTURES_CURRENT, TEAMS_CURRENT, target_gameweek=1)
    for column in [*EXPECTED_ROLL_COLUMNS, "selected"]:
        assert result[column].dtype.kind in "fc", f"{column} has non-numeric dtype {result[column].dtype}"
