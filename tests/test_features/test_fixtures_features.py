"""Checks DGW/BGW fixture counting and home/away-conditional opponent-strength resolution."""

import pandas as pd

from fpl_agent.features.fixtures_features import gameweek_fixture_counts, team_fixture_difficulty

FIXTURES = pd.DataFrame(
    {
        "event": [1, 1, 2],
        "team_h": [1, 3, 1],
        "team_a": [2, 4, 3],
        "team_h_difficulty": [3, 2, 4],
        "team_a_difficulty": [5, 4, 1],
    }
)

TEAMS = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "strength_attack_home": [1300, 1200, 1100, 1000],
        "strength_attack_away": [1250, 1150, 1050, 950],
        "strength_defence_home": [1310, 1210, 1110, 1010],
        "strength_defence_away": [1260, 1160, 1060, 960],
    }
)


def test_gameweek_fixture_counts_flags_double_gameweek():
    """Team 1 plays twice in GW2 (once as team_h, once via a second fixture) - a double gameweek."""
    extra = pd.concat([FIXTURES, pd.DataFrame({"event": [2], "team_h": [5], "team_a": [1], "team_h_difficulty": [3], "team_a_difficulty": [3]})])
    counts = gameweek_fixture_counts(extra)
    team1_gw2 = counts[(counts["team"] == 1) & (counts["event"] == 2)]["fixture_count"].iloc[0]
    assert team1_gw2 == 2


def test_gameweek_fixture_counts_blank_team_is_simply_absent():
    """A team with no fixture that gameweek has no row at all - blanks are absences, not zeros."""
    counts = gameweek_fixture_counts(FIXTURES)
    assert counts[(counts["team"] == 2) & (counts["event"] == 2)].empty


def test_team_fixture_difficulty_uses_home_rating_for_home_team():
    """Team 1 at home in GW1 (vs team 2) must use team_h_difficulty (3), not team_a_difficulty (5)."""
    result = team_fixture_difficulty(FIXTURES, TEAMS)
    row = result[(result["team"] == 1) & (result["event"] == 1)].iloc[0]
    assert row["difficulty"] == 3
    assert bool(row["was_home"]) is True


def test_team_fixture_difficulty_resolves_opponent_strength_by_venue():
    """Team 1 (home) faces team 2 playing away, so team 2's AWAY strength ratings apply, not its home ones."""
    result = team_fixture_difficulty(FIXTURES, TEAMS)
    row = result[(result["team"] == 1) & (result["event"] == 1)].iloc[0]
    assert row["opponent_attack_strength"] == 1150
    assert row["opponent_defence_strength"] == 1160
