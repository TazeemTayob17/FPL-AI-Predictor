# Checks team-level xG-for/xG-against is correctly aggregated from player-level expected_goals and correctly paired via fixtures, and that rolling form carries across a season boundary exactly like team_strength.py's actual-goals version.

import pandas as pd

from fpl_agent.features.team_xg_strength import (
    ROLL_COLUMNS,
    add_team_xg_strength_rolling,
    build_team_xg_against,
    build_team_xg_for,
    build_team_xg_strength_table,
    latest_team_xg_strength,
)

TEAMS = pd.DataFrame(
    [
        {"season": "s1", "id": 1, "code": 100, "name": "Team100"},
        {"season": "s1", "id": 2, "code": 200, "name": "Team200"},
    ]
)
GW_ROWS = pd.DataFrame(
    [
        {"season": "s1", "GW": 1, "team": "Team100", "expected_goals": 1.5},
        {"season": "s1", "GW": 1, "team": "Team100", "expected_goals": 0.3},
        {"season": "s1", "GW": 1, "team": "Team200", "expected_goals": 0.5},
    ]
)
FIXTURES = pd.DataFrame([{"season": "s1", "event": 1, "team_h": 1, "team_a": 2, "finished": True}])


# Two players on the same team in the same gameweek must sum, not overwrite - the team's total xG for that match.
def test_build_team_xg_for_sums_player_level_expected_goals_to_team_level():
    result = build_team_xg_for(GW_ROWS, TEAMS)
    team100 = result[(result["code"] == 100) & (result["GW"] == 1)].iloc[0]
    assert team100["team_xg_for"] == 1.8


# A team's xG-against for a fixture is exactly the opponent's xG-for that same fixture - what the defense actually faced.
def test_build_team_xg_against_uses_opponent_xg_for():
    xg_for = build_team_xg_for(GW_ROWS, TEAMS)
    xg_against = build_team_xg_against(xg_for, FIXTURES, TEAMS)
    team100_against = xg_against[(xg_against["code"] == 100) & (xg_against["GW"] == 1)].iloc[0]
    team200_against = xg_against[(xg_against["code"] == 200) & (xg_against["GW"] == 1)].iloc[0]
    assert team100_against["team_xg_against"] == 0.5
    assert team200_against["team_xg_against"] == 1.8


# Older seasons without expected_goals in the data must degrade gracefully to an empty-but-correctly-columned frame, not crash.
def test_build_team_xg_strength_table_handles_missing_expected_goals_column():
    gw_rows_no_xg = GW_ROWS.drop(columns=["expected_goals"])
    result = build_team_xg_strength_table(gw_rows_no_xg, FIXTURES, TEAMS)
    assert result.empty
    assert set(ROLL_COLUMNS).issubset(result.columns)


# A team's rolling xG form must chain across a season boundary via its stable club code, not reset to blank at GW1 of a new season.
def test_team_xg_strength_rolling_carries_across_season_boundary_via_club_code():
    match_results = pd.DataFrame(
        [
            {"season": "2024-25", "GW": 38, "code": 100, "team_xg_for": 1.0, "team_xg_against": 2.5},
            {"season": "2025-26", "GW": 1, "code": 100, "team_xg_for": 1.2, "team_xg_against": 0.8},
        ]
    )
    result = add_team_xg_strength_rolling(match_results)
    new_season_row = result[(result["season"] == "2025-26") & (result["GW"] == 1)].iloc[0]
    assert new_season_row["team_xg_against_roll3"] == 2.5


# latest_team_xg_strength must pick the chronologically last row per team, even across a season boundary.
def test_latest_team_xg_strength_returns_most_recent_row_regardless_of_season():
    table = add_team_xg_strength_rolling(
        pd.DataFrame(
            [
                {"season": "2024-25", "GW": 38, "code": 100, "team_xg_for": 1.0, "team_xg_against": 2.5},
                {"season": "2025-26", "GW": 5, "code": 100, "team_xg_for": 1.2, "team_xg_against": 0.8},
            ]
        )
    )
    expected = table[(table["season"] == "2025-26") & (table["GW"] == 5)].iloc[0]
    actual_row = latest_team_xg_strength(table)
    actual_row = actual_row[actual_row["code"] == 100].iloc[0]
    for column in ROLL_COLUMNS:
        assert actual_row[column] == expected[column]
