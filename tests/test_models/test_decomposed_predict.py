# Hand-calculable checks that recombine_points applies the real FPL scoring formula correctly, given fixed component predictions (no trained models involved).

import math

import pandas as pd
import pytest

from fpl_agent.models.decomposed_predict import recombine_points

BASE_ROW = {
    "position": "DEF", "p_any_minutes": 1.0, "p_sixty_plus_minutes": 1.0, "minutes_when_played": 90.0,
    "goals_per90": 0.0, "assists_per90": 0.0, "saves_per90": 0.0, "bonus_per90": 0.0,
    "p_defensive_threshold": 0.0, "team_expected_goals_conceded": 0.0,
    "own_goals_roll10": 0.0, "penalties_missed_roll10": 0.0, "yellow_cards_roll10": 0.0,
    "red_cards_roll10": 0.0, "penalties_saved_roll10": 0.0,
}


def _row(**overrides) -> pd.DataFrame:
    return pd.DataFrame([{**BASE_ROW, **overrides}])


# A defender nailed on for 90 minutes with a guaranteed clean sheet (0 expected goals conceded) scores exactly 2 (appearance) + 4 (clean sheet) = 6.
def test_defender_certain_appearance_and_clean_sheet():
    result = recombine_points(_row())
    assert result.iloc[0] == pytest.approx(6.0)


# A forward nailed on for 90 minutes averaging 1 goal and 0.5 assists per 90 scores 2 (appearance) + 4 (goal) + 1.5 (assist) = 7.5. Forwards get 0 clean-sheet points, so clean-sheet probability shouldn't matter for this position.
def test_forward_goal_and_assist_rate():
    result = recombine_points(_row(position="FWD", goals_per90=1.0, assists_per90=0.5, team_expected_goals_conceded=5.0))
    assert result.iloc[0] == pytest.approx(7.5)


# A defender meeting the defensive-contribution threshold with certainty adds exactly 2 points.
def test_defensive_contribution_adds_its_point_value_when_certain():
    with_dc = recombine_points(_row(p_defensive_threshold=1.0)).iloc[0]
    without_dc = recombine_points(_row(p_defensive_threshold=0.0)).iloc[0]
    assert with_dc - without_dc == pytest.approx(2.0)


# A goalkeeper facing 2.0 expected goals conceded: clean-sheet points shrink to exp(-2)*4, and the goals-conceded penalty is exactly -1 per 2 expected goals conceded (continuous approximation), i.e. -1.0 here.
def test_goalkeeper_goals_conceded_penalty_and_reduced_clean_sheet_probability():
    result = recombine_points(_row(position="GKP", team_expected_goals_conceded=2.0)).iloc[0]
    expected = 2.0 + math.exp(-2.0) * 4.0 - 1.0
    assert result == pytest.approx(expected)


# A player who's a doubt (50% chance of any minutes) has all rate-based and appearance contributions scaled down accordingly - appearance points alone should be 0.5 + 0.5 = 1.0, not 2.0.
def test_rotation_doubt_scales_down_appearance_points():
    result = recombine_points(_row(p_any_minutes=0.5, p_sixty_plus_minutes=0.5)).iloc[0]
    assert result == pytest.approx(1.0 + math.exp(0.0) * 0.5 * 4.0)  # appearance(1.0) + clean_sheet(p_cs=1 since 0 xGC * p_sixty(0.5) * 4)


# Missing component columns must default to a neutral 0/90 rather than raising, since a fresh features table won't always carry every optional column.
def test_missing_component_columns_default_gracefully():
    minimal = pd.DataFrame([{"position": "MID", "p_any_minutes": 1.0, "p_sixty_plus_minutes": 1.0, "minutes_when_played": 90.0}])
    result = recombine_points(minimal)
    assert result.iloc[0] == pytest.approx(2.0 + 1.0)  # appearance(2) + clean sheet (p_cs=1, MID clean sheet = 1)
