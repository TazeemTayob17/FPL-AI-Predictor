"""Checks that bootstrap-static rows get normalized into a readable players table."""

from fpl_agent.storage.repository import _build_players_frame

SAMPLE_BOOTSTRAP = {
    "elements": [
        {
            "id": 1, "first_name": "Erling", "second_name": "Haaland", "web_name": "Haaland",
            "team": 1, "element_type": 4, "now_cost": 150, "form": "5.2", "total_points": 20,
            "selected_by_percent": "45.3", "status": "a", "news": "",
            "chance_of_playing_this_round": None, "chance_of_playing_next_round": None,
        },
    ],
    "teams": [{"id": 1, "name": "Man City", "short_name": "MCI"}],
    "element_types": [
        {"id": 1, "singular_name_short": "GKP"}, {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"}, {"id": 4, "singular_name_short": "FWD"},
    ],
}


def test_build_players_frame_joins_team_and_position():
    """Checks that team name and position get attached and cost is converted to millions."""
    players = _build_players_frame(SAMPLE_BOOTSTRAP)
    row = players.iloc[0]
    assert row["team_short"] == "MCI"
    assert row["position"] == "FWD"
    assert row["now_cost_million"] == 15.0
    assert row["form"] == 5.2
