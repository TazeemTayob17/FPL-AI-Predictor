# Checks next_fixture_by_team reads only the locally cached parquet files (no live calls) and correctly pairs each team with its opponent/home-away/difficulty.

import pandas as pd

from fpl_agent.ui.components import fixtures as fixtures_module
from fpl_agent.ui.components.fixtures import next_fixture_by_team


def _write_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures_module, "PROCESSED_DIR", tmp_path)
    pd.DataFrame(
        [
            {"player_id": 1, "team_id": 1, "team_short": "ARS"},
            {"player_id": 2, "team_id": 2, "team_short": "CHE"},
        ]
    ).to_parquet(tmp_path / "players_current.parquet")


# Each side of a fixture gets the opponent's short name and the correct home/away flag.
def test_next_fixture_by_team_pairs_home_and_away_correctly(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch)
    pd.DataFrame(
        [{"event": 5, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4, "finished": False}]
    ).to_parquet(tmp_path / "fixtures_current.parquet")

    result = next_fixture_by_team()
    assert result[1] == {"opponent_short": "CHE", "is_home": True, "difficulty": 2}
    assert result[2] == {"opponent_short": "ARS", "is_home": False, "difficulty": 4}


# A team's earliest unplayed fixture must win over a later one, and finished fixtures must be excluded entirely.
def test_next_fixture_by_team_picks_the_earliest_unplayed_fixture(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch)
    pd.DataFrame(
        [
            {"event": 4, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3, "finished": True},
            {"event": 5, "team_h": 2, "team_a": 1, "team_h_difficulty": 5, "team_a_difficulty": 1, "finished": False},
            {"event": 6, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 2, "finished": False},
        ]
    ).to_parquet(tmp_path / "fixtures_current.parquet")

    result = next_fixture_by_team()
    assert result[1]["difficulty"] == 1  # GW5's team_a_difficulty for team 1, not GW6's
    assert result[1]["is_home"] is False


# Missing cache files must degrade to an empty dict, not raise - the dashboard shouldn't crash before the first refresh.
def test_next_fixture_by_team_returns_empty_dict_when_no_cache_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures_module, "PROCESSED_DIR", tmp_path)
    assert next_fixture_by_team() == {}
