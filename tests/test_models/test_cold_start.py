"""Checks the shrinkage estimator and final-standings logic behind the pre-season cold-start predictor."""

import pandas as pd
import pytest

from fpl_agent.models.cold_start import blended_points_per_90, final_standings

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
