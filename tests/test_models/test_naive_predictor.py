"""Checks the naive predictor's season-label logic and name-based matching against last season's totals."""

import pandas as pd

from fpl_agent.models.naive_predictor import predict_naive_points, previous_season_label

LAST_SEASON = pd.DataFrame(
    {
        "player_name": ["erling haaland", "mohamed salah"],
        "last_season_points": [220, 260],
    }
)


def test_previous_season_label_steps_back_one_year():
    """Checks "2026-27" maps to "2025-26", the season the naive predictor should read from."""
    assert previous_season_label("2026-27") == "2025-26"
    assert previous_season_label("2025-26") == "2024-25"


def test_predict_naive_points_matches_by_full_name():
    """Checks a player present in last season's data gets their prior total as predicted_points."""
    players = pd.DataFrame({"web_name_full": ["Erling Haaland"], "position": ["FWD"]})
    result = predict_naive_points(players, last_season=LAST_SEASON)
    assert result.loc[0, "predicted_points"] == 220


def test_predict_naive_points_falls_back_to_zero_for_unmatched_players():
    """Checks a player absent from last season's data (e.g. newly promoted) gets a 0.0 fallback, not a crash."""
    players = pd.DataFrame({"web_name_full": ["Brand New Signing"], "position": ["MID"]})
    result = predict_naive_points(players, last_season=LAST_SEASON)
    assert result.loc[0, "predicted_points"] == 0.0


def test_predict_naive_points_matching_is_case_and_whitespace_insensitive():
    """Checks matching tolerates case differences between the FPL API and the vaastav CSV name spelling."""
    players = pd.DataFrame({"web_name_full": ["  MOHAMED SALAH  "], "position": ["MID"]})
    result = predict_naive_points(players, last_season=LAST_SEASON)
    assert result.loc[0, "predicted_points"] == 260
