"""Checks the shrinkage estimator and final-standings logic behind the pre-season cold-start predictor."""

import pandas as pd
import pytest

from fpl_agent.models import cold_start
from fpl_agent.models.cold_start import (
    blended_points_per_90,
    build_opening_difficulty,
    default_weak_team_prior,
    final_standings,
    predict_cold_start_points,
)

SHRINKAGE_MINUTES = 450.0


def test_blended_pp90_is_exactly_the_prior_at_zero_minutes():
    """With zero minutes there is no actual signal at all, so the blend must equal the position average exactly."""
    result = blended_points_per_90(pd.Series([0.0]), pd.Series([0.0]), pd.Series([4.0]))
    assert result.iloc[0] == 4.0


def test_blended_pp90_mostly_trusts_a_full_seasons_own_rate():
    """With a full season of minutes at a 6.0 pp90 rate, the blend should sit close to 6.0, only slightly pulled to the prior."""
    result = blended_points_per_90(pd.Series([2280.0]), pd.Series([34200.0]), pd.Series([2.0]))
    assert result.iloc[0] == pytest.approx(5.948, abs=0.01)


def test_final_standings_awards_three_points_for_a_win():
    """A 2-0 home win must give the home team 3 points and the away team 0."""
    fixtures = pd.DataFrame({"team_h": [1], "team_a": [2], "team_h_score": [2.0], "team_a_score": [0.0]})
    standings = final_standings(fixtures)
    assert standings.loc[1] == 3
    assert standings.loc[2] == 0


def test_final_standings_awards_one_point_each_for_a_draw():
    """A 1-1 draw must give both teams 1 point."""
    fixtures = pd.DataFrame({"team_h": [1], "team_a": [2], "team_h_score": [1.0], "team_a_score": [1.0]})
    standings = final_standings(fixtures)
    assert standings.loc[1] == 1
    assert standings.loc[2] == 1


def test_final_standings_ignores_unplayed_fixtures():
    """A fixture with no score yet (NaN) must not contribute points to either team."""
    fixtures = pd.DataFrame({"team_h": [1, 3], "team_a": [2, 4], "team_h_score": [2.0, None], "team_a_score": [0.0, None]})
    standings = final_standings(fixtures)
    assert 3 not in standings.index
    assert 4 not in standings.index


def test_default_weak_team_prior_is_empty_when_no_historical_fixtures_data_exists(monkeypatch, tmp_path):
    """No all_seasons_fixtures/teams parquet on disk yet (e.g. before vaastav_sync has run) must degrade gracefully, not crash."""
    monkeypatch.setattr(cold_start, "PROCESSED_DIR", tmp_path)
    prior_stats = pd.DataFrame({"player_name": ["a"], "team": ["Weak FC"], "total_points": [10], "minutes": [900], "appearances": [10]})
    result = default_weak_team_prior("2026-27", current_positions=pd.DataFrame(columns=["player_name", "position"]), prior_stats=prior_stats)
    assert result.empty


def test_default_weak_team_prior_computes_the_bottom_tier_average_from_real_files(monkeypatch, tmp_path):
    """With real historical fixtures/teams on disk, the bottom-5 tier's per-position average must be computed, not left empty."""
    monkeypatch.setattr(cold_start, "PROCESSED_DIR", tmp_path)

    # 6 teams; team 6 wins everything (top), teams 1-5 lose everything (bottom tier).
    fixtures = pd.DataFrame(
        {
            "season": ["2025-26"] * 5,
            "team_h": [6, 6, 6, 6, 6],
            "team_a": [1, 2, 3, 4, 5],
            "team_h_score": [3, 3, 3, 3, 3],
            "team_a_score": [0, 0, 0, 0, 0],
        }
    )
    teams = pd.DataFrame({"season": ["2025-26"] * 6, "id": [1, 2, 3, 4, 5, 6], "name": ["A", "B", "C", "D", "E", "F"]})
    fixtures.to_parquet(tmp_path / "all_seasons_fixtures.parquet", index=False)
    teams.to_parquet(tmp_path / "all_seasons_teams.parquet", index=False)

    prior_stats = pd.DataFrame(
        {
            "player_name": ["weak striker", "top striker"],
            "team": ["A", "F"],
            "total_points": [20, 200],
            "minutes": [900, 900],
            "appearances": [10, 10],
        }
    )
    current_positions = pd.DataFrame({"player_name": ["weak striker", "top striker"], "position": ["FWD", "FWD"]})

    result = default_weak_team_prior("2026-27", current_positions, prior_stats)

    assert not result.empty
    assert result.loc[result["position"] == "FWD", "weak_team_prior_pp_gw"].iloc[0] == pytest.approx(2.0)  # 20 pts / 10 apps


def test_predict_cold_start_points_no_longer_zeroes_debut_players(monkeypatch):
    """Regression test for the bug found in review: debut/promoted-club players must not silently get predicted_points=0."""
    monkeypatch.setattr(cold_start, "default_weak_team_prior", lambda season_label, current_positions, prior_stats: pd.DataFrame(
        [{"position": "FWD", "weak_team_prior_pp_gw": 3.0}]
    ))
    players = pd.DataFrame(
        [{"web_name_full": "Brand New Signing", "position": "FWD", "team_id": 1, "team_name": "Newly Promoted FC"}]
    )
    result = predict_cold_start_points(players, season_label="2026-27", prior_stats=pd.DataFrame(columns=["player_name", "team", "total_points", "minutes", "appearances"]))
    assert result.loc[0, "predicted_points"] == pytest.approx(3.0)


def test_predict_cold_start_points_debut_player_is_zero_only_when_truly_no_data_available(monkeypatch, tmp_path):
    """With no historical data anywhere (fresh checkout, no vaastav_sync yet), the old graceful-zero fallback must still hold."""
    monkeypatch.setattr(cold_start, "PROCESSED_DIR", tmp_path)
    players = pd.DataFrame(
        [{"web_name_full": "Brand New Signing", "position": "FWD", "team_id": 1, "team_name": "Newly Promoted FC"}]
    )
    result = predict_cold_start_points(players, season_label="2026-27", prior_stats=pd.DataFrame(columns=["player_name", "team", "total_points", "minutes", "appearances"]))
    assert result.loc[0, "predicted_points"] == 0.0


def test_build_opening_difficulty_averages_only_within_the_window():
    """A team's difficulty must average only their first `window_gws` fixtures, ignoring later ones."""
    fixtures_current = pd.DataFrame(
        {
            "event": [1, 2, 3, 4],
            "team_h": [1, 1, 1, 1],
            "team_a": [2, 2, 2, 2],
            "team_h_difficulty": [2, 4, 5, 5],
            "team_a_difficulty": [3, 3, 3, 3],
        }
    )
    teams_current = pd.DataFrame(
        {
            "id": [1, 2], "strength_attack_home": [1000, 1000], "strength_attack_away": [1000, 1000],
            "strength_defence_home": [1000, 1000], "strength_defence_away": [1000, 1000],
        }
    )
    result = build_opening_difficulty(fixtures_current, teams_current, window_gws=2)
    team_1_difficulty = result.loc[result["team_id"] == 1, "difficulty"].iloc[0]
    assert team_1_difficulty == pytest.approx(3.0)  # (2 + 4) / 2, GW3-4's 5s excluded
