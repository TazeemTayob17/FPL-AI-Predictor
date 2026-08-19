# Checks per-fixture rows collapse correctly into one row per player per gameweek, including double-gameweek summing.

import pandas as pd

from fpl_agent.features.build_features import EXPECTED_ROLL_COLUMNS, build_current_features, collapse_to_gameweek
from fpl_agent.features.team_strength import ROLL_COLUMNS as TEAM_STRENGTH_ROLL_COLUMNS

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


TEAMS_CURRENT_WITH_CODE = pd.DataFrame(
    {
        "id": [1, 2], "code": [100, 200],
        "strength_attack_home": [1000, 1000], "strength_attack_away": [1000, 1000],
        "strength_defence_home": [1000, 1000], "strength_defence_away": [1000, 1000],
    }
)
HISTORICAL_FIXTURES = pd.DataFrame(
    {
        "season": ["2025-26", "2025-26"], "event": [37, 38], "team_h": [1, 1], "team_a": [2, 2],
        "team_h_score": [5, 0], "team_a_score": [1, 3], "finished": [True, True],
    }
)
HISTORICAL_TEAMS = pd.DataFrame({"season": ["2025-26", "2025-26"], "id": [1, 2], "code": [100, 200]})


# When historical fixtures/teams are supplied, a team's real last-season defensive form carries into GW1 instead of starting blank.
def test_build_current_features_carries_team_strength_from_last_season():
    players = pd.DataFrame([{"player_id": 1, "team_id": 1, "position": "FWD", "now_cost_million": 8.0}])
    result = build_current_features(
        {1: pd.DataFrame()}, players, FIXTURES_CURRENT, TEAMS_CURRENT_WITH_CODE, target_gameweek=1,
        historical_fixtures=HISTORICAL_FIXTURES, historical_teams=HISTORICAL_TEAMS,
    )
    assert result.loc[0, "team_goals_conceded_roll3"] == 1


# Without historical data, team-strength columns still exist (as NaN) instead of being missing entirely and crashing the trained model's column lookup.
def test_build_current_features_team_strength_columns_are_nan_without_historical_data():
    players = pd.DataFrame([{"player_id": 1, "team_id": 1, "position": "FWD", "now_cost_million": 8.0}])
    result = build_current_features({1: pd.DataFrame()}, players, FIXTURES_CURRENT, TEAMS_CURRENT, target_gameweek=1)
    for column in TEAM_STRENGTH_ROLL_COLUMNS:
        assert column in result.columns
        assert pd.isna(result.loc[0, column])


PREVIOUS_SEASON_FORM = pd.DataFrame([{"player_id": 1, **{col: 5.0 for col in EXPECTED_ROLL_COLUMNS}}])


# A player with no current-season history yet gets last season's rolling form as a GW1 starting point, not blank NaN.
def test_build_current_features_uses_previous_season_form_when_no_current_history():
    players = pd.DataFrame([{"player_id": 1, "team_id": 1, "position": "FWD", "now_cost_million": 8.0}])
    result = build_current_features(
        {1: pd.DataFrame()}, players, FIXTURES_CURRENT, TEAMS_CURRENT, target_gameweek=1,
        previous_season_form=PREVIOUS_SEASON_FORM,
    )
    assert result.loc[0, "total_points_roll3"] == 5.0


# Once a player has a real current-season game, the real (if still sparse) rolling pipeline is used instead of the previous-season fallback.
def test_build_current_features_prefers_real_current_history_over_previous_season_fallback():
    players = pd.DataFrame([{"player_id": 1, "team_id": 1, "position": "FWD", "now_cost_million": 8.0}])
    history = pd.DataFrame({"GW": [1], "total_points": [9], "minutes": [90]})
    result = build_current_features(
        {1: history}, players, FIXTURES_CURRENT, TEAMS_CURRENT, target_gameweek=1,
        previous_season_form=PREVIOUS_SEASON_FORM,
    )
    assert pd.isna(result.loc[0, "total_points_roll3"])
