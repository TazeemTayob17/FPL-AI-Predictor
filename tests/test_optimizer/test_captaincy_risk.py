"""Checks the captaincy risk score (injury doubt + minutes volatility) and its graceful degradation when data is missing."""

import pandas as pd

from fpl_agent.optimizer.captaincy import captaincy_candidates, compute_risk_score

STARTING_XI = pd.DataFrame(
    {
        "web_name": ["fwd1", "gk1", "mid1"],
        "predicted_points": [55, 50, 42],
        "chance_of_playing_next_round": [100, 50, 100],
        "minutes_volatility": [0.0, 45.0, 0.0],
    }
)


def test_fully_fit_zero_volatility_player_has_near_zero_risk():
    """A fully fit, always-90-minutes player must score close to 0 risk, not just 'low'."""
    score = compute_risk_score(pd.Series([100]), pd.Series([0.0]))
    assert score.iloc[0] == 0.0


def test_doubtful_and_volatile_player_scores_higher_risk_than_a_safe_one():
    """A 50% chance-of-playing, highly volatile player must score clearly higher risk than a nailed-on starter."""
    doubtful = compute_risk_score(pd.Series([50]), pd.Series([45.0]))
    safe = compute_risk_score(pd.Series([100]), pd.Series([0.0]))
    assert doubtful.iloc[0] > safe.iloc[0]


def test_risk_score_is_clipped_to_the_0_1_range():
    """Even an extreme volatility value must not push the risk score above 1."""
    score = compute_risk_score(pd.Series([0]), pd.Series([9000.0]))
    assert 0.0 <= score.iloc[0] <= 1.0


def test_missing_chance_of_playing_defaults_to_fully_fit():
    """A player with no injury news (NaN chance_of_playing) should not be penalized as if doubtful."""
    score = compute_risk_score(pd.Series([None]), pd.Series([0.0]))
    assert score.iloc[0] == 0.0


def test_captaincy_candidates_ranks_by_predicted_points_with_risk_attached():
    """Candidates must be ranked highest-predicted-points first, with a risk_score column present."""
    ranked = captaincy_candidates(STARTING_XI)
    assert ranked["web_name"].tolist() == ["fwd1", "gk1", "mid1"]
    assert "risk_score" in ranked.columns
    assert ranked.iloc[1]["risk_score"] > ranked.iloc[0]["risk_score"]


def test_captaincy_candidates_degrades_gracefully_without_risk_columns():
    """A starting XI lacking chance_of_playing/minutes_volatility (e.g. from the naive predictor) must not crash."""
    plain_xi = pd.DataFrame({"web_name": ["a", "b"], "predicted_points": [10, 5]})
    ranked = captaincy_candidates(plain_xi)
    assert "risk_score" not in ranked.columns
    assert ranked["web_name"].tolist() == ["a", "b"]
