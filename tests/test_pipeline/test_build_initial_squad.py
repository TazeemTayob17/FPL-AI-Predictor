# Checks build_initial_squad can use already-fetched players/bootstrap instead of reading local cache files - this is what fixed a real production crash on the stateless Cloud deployment, where data/raw's cached bootstrap JSON never exists.

import pandas as pd
import pytest

from fpl_agent.pipeline import build_initial_squad as module
from fpl_agent.pipeline.build_initial_squad import build_initial_squad

PLAYERS = pd.DataFrame(
    [{"player_id": 1, "web_name": "p1", "position": "GKP", "now_cost_million": 5.0, "team_name": "A"}]
)
BOOTSTRAP = {"events": []}


# When players/bootstrap are passed in directly, no local files should be touched at all.
def test_build_initial_squad_uses_passed_in_players_and_bootstrap_without_touching_disk(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not read local cache files when players/bootstrap are already provided")

    monkeypatch.setattr(module, "load_latest_json", _fail_if_called)
    monkeypatch.setattr(
        module, "predict_horizon_points", lambda players, bootstrap, horizon_gws: (players.assign(predicted_points=10.0), False)
    )
    monkeypatch.setattr(module, "select_squad_and_starting_xi", lambda pool: (pool, pool, pool.iloc[0:0]))

    squad, starting_xi, bench, all_players, used_cold_start = build_initial_squad(PLAYERS, BOOTSTRAP)
    assert used_cold_start is False
    assert list(all_players["web_name"]) == ["p1"]


# Without a cached bootstrap payload and no players/bootstrap passed in, the old clear error must still fire - a real "run the refresh first" case, not a silent failure.
def test_build_initial_squad_raises_a_clear_error_when_nothing_is_available(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(module, "load_latest_json", lambda subdir: None)

    with pytest.raises(FileNotFoundError, match="players_current.parquet"):
        build_initial_squad()
