"""Checks captaincy selection picks the top two predicted_points players in the starting XI."""

import pandas as pd

from fpl_agent.optimizer.captaincy import choose_captaincy

STARTING_XI = pd.DataFrame(
    {
        "web_name": ["fwd1", "gk1", "mid1", "def1", "def2"],
        "predicted_points": [55, 50, 42, 40, 38],
    }
)


def test_choose_captaincy_picks_highest_predicted_points_as_captain():
    """Captain is the single highest predicted_points player in the XI, regardless of position."""
    captain, _ = choose_captaincy(STARTING_XI)
    assert captain["web_name"] == "fwd1"
    assert captain["predicted_points"] == 55


def test_choose_captaincy_picks_second_highest_as_vice_captain():
    """Vice-captain is the runner-up by predicted_points, distinct from the captain."""
    _, vice_captain = choose_captaincy(STARTING_XI)
    assert vice_captain["web_name"] == "gk1"
    assert vice_captain["predicted_points"] == 50
