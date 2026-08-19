# Checks team-level goals-scored/conceded/clean-sheet rolling form is computed correctly and carries across a season boundary.

import pandas as pd

from fpl_agent.features.team_strength import (
    ROLL_COLUMNS,
    add_team_strength_rolling,
    build_team_match_results,
    latest_team_strength,
)

FIXTURES = pd.DataFrame(
    [
        {"season": "s1", "event": 1, "team_h": 1, "team_a": 2, "team_h_score": 2, "team_a_score": 0, "finished": True},
        {"season": "s1", "event": 2, "team_h": 2, "team_a": 1, "team_h_score": 1, "team_a_score": 1, "finished": True},
        {"season": "s1", "event": 3, "team_h": 1, "team_a": 2, "team_h_score": 0, "team_a_score": 3, "finished": False},
    ]
)
TEAMS = pd.DataFrame([{"season": "s1", "id": 1, "code": 100}, {"season": "s1", "id": 2, "code": 200}])


# Both the home and away side of a finished fixture produce one row each, with goals and clean-sheet correctly assigned per side.
def test_build_team_match_results_extracts_goals_and_clean_sheets_for_both_sides():
    result = build_team_match_results(FIXTURES, TEAMS)
    assert len(result) == 4
    team100_gw1 = result[(result["code"] == 100) & (result["GW"] == 1)].iloc[0]
    assert team100_gw1["team_goals_scored"] == 2
    assert team100_gw1["team_goals_conceded"] == 0
    assert team100_gw1["team_clean_sheet"] == 1


# GW3 is not finished yet and must be excluded entirely, not counted as a 0-0.
def test_build_team_match_results_excludes_unfinished_fixtures():
    result = build_team_match_results(FIXTURES, TEAMS)
    assert (result["GW"] == 3).sum() == 0


# Regression test: a double-gameweek (2 fixtures, same team, same GW) must collapse into ONE row with summed goals, not 2 rows - 2 rows would silently double-count every player on that team in that gameweek downstream.
def test_build_team_match_results_collapses_a_double_gameweek_into_one_row():
    dgw_fixtures = pd.DataFrame(
        [
            {"season": "s1", "event": 1, "team_h": 1, "team_a": 2, "team_h_score": 2, "team_a_score": 0, "finished": True},
            {"season": "s1", "event": 1, "team_h": 3, "team_a": 1, "team_h_score": 1, "team_a_score": 1, "finished": True},
        ]
    )
    dgw_teams = pd.DataFrame([{"season": "s1", "id": 1, "code": 100}, {"season": "s1", "id": 2, "code": 200}, {"season": "s1", "id": 3, "code": 300}])
    result = build_team_match_results(dgw_fixtures, dgw_teams)
    team100_rows = result[(result["code"] == 100) & (result["GW"] == 1)]
    assert len(team100_rows) == 1
    assert team100_rows.iloc[0]["team_goals_scored"] == 3
    assert team100_rows.iloc[0]["team_goals_conceded"] == 1


# A team's rolling form must chain across a season boundary via its stable club code, not reset to blank at GW1 of a new season.
def test_team_strength_rolling_carries_across_season_boundary_via_club_code():
    match_results = pd.DataFrame(
        [
            {"season": "2024-25", "GW": 38, "code": 100, "team_goals_scored": 1, "team_goals_conceded": 3, "team_clean_sheet": 0},
            {"season": "2025-26", "GW": 1, "code": 100, "team_goals_scored": 2, "team_goals_conceded": 1, "team_clean_sheet": 0},
        ]
    )
    result = add_team_strength_rolling(match_results)
    new_season_row = result[(result["season"] == "2025-26") & (result["GW"] == 1)].iloc[0]
    assert new_season_row["team_goals_conceded_roll3"] == 3


# latest_team_strength must pick the chronologically last row per team, even when that row is in an earlier season than others in the table.
def test_latest_team_strength_returns_most_recent_row_regardless_of_season():
    table = add_team_strength_rolling(
        pd.DataFrame(
            [
                {"season": "2024-25", "GW": 38, "code": 100, "team_goals_scored": 1, "team_goals_conceded": 3, "team_clean_sheet": 0},
                {"season": "2025-26", "GW": 5, "code": 100, "team_goals_scored": 2, "team_goals_conceded": 1, "team_clean_sheet": 0},
            ]
        )
    )
    expected = table[(table["season"] == "2025-26") & (table["GW"] == 5)].iloc[0]
    actual = latest_team_strength(table)
    actual_row = actual[actual["code"] == 100].iloc[0]
    for column in ROLL_COLUMNS:
        assert actual_row[column] == expected[column]
