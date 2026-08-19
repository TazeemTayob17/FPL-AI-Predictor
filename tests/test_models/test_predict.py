# Checks the cold-start-vs-trained-model routing threshold logic and horizon-prediction wiring, with no live network calls.

import pandas as pd
import pytest

from fpl_agent.models import predict as predict_module
from fpl_agent.models.predict import completed_gameweeks_this_season, fetch_player_histories, should_use_cold_start

BOOTSTRAP_PRE_SEASON = {"events": [{"id": 1, "finished": False}, {"id": 2, "finished": False}]}
BOOTSTRAP_MID_SEASON = {"events": [{"id": 1, "finished": True}, {"id": 2, "finished": True}, {"id": 3, "finished": False}]}
PLAYERS = pd.DataFrame([{"player_id": 1, "web_name": "p1", "predicted_points": 4.0}])


# Pre-season, with no events finished yet, the count must be 0.
def test_completed_gameweeks_counts_only_finished_events():
    assert completed_gameweeks_this_season(BOOTSTRAP_PRE_SEASON) == 0


# Two finished events and one still upcoming must count as 2, not 3.
def test_completed_gameweeks_reflects_finished_events():
    assert completed_gameweeks_this_season(BOOTSTRAP_MID_SEASON) == 2


# Fewer completed gameweeks than the configured threshold must route to cold-start.
def test_should_use_cold_start_below_threshold():
    assert should_use_cold_start(completed_gameweeks=2, threshold=4) is True


# Once enough gameweeks have completed, routing must switch away from cold-start.
def test_should_use_cold_start_at_or_above_threshold():
    assert should_use_cold_start(completed_gameweeks=4, threshold=4) is False
    assert should_use_cold_start(completed_gameweeks=5, threshold=4) is False


# The raw element-summary payload names the gameweek field "round"; build_current_features expects "GW".
def test_fetch_player_histories_renames_round_to_gw(monkeypatch):
    monkeypatch.setattr(
        predict_module.fpl_api, "get_element_summary",
        lambda player_id, max_age_hours=None: {"history": [{"round": 3, "total_points": 6}, {"round": 4, "total_points": 2}]},
    )
    players = pd.DataFrame([{"player_id": 1}])

    histories = fetch_player_histories(players)

    assert list(histories[1]["GW"]) == [3, 4]
    assert "round" not in histories[1].columns


# A brand-new player with an empty history list must produce an empty frame, not raise.
def test_fetch_player_histories_handles_a_player_with_no_history_yet(monkeypatch):
    monkeypatch.setattr(predict_module.fpl_api, "get_element_summary", lambda player_id, max_age_hours=None: {"history": []})
    players = pd.DataFrame([{"player_id": 1}])

    histories = fetch_player_histories(players)

    assert histories[1].empty


# fetch_player_histories must forward its freshness window to get_element_summary so daily manual runs reuse today's pull.
def test_fetch_player_histories_passes_max_age_hours_through(monkeypatch):
    seen_max_age = []
    monkeypatch.setattr(
        predict_module.fpl_api, "get_element_summary",
        lambda player_id, max_age_hours=None: seen_max_age.append(max_age_hours) or {"history": []},
    )
    players = pd.DataFrame([{"player_id": 1}])

    fetch_player_histories(players, max_age_hours=6.0)

    assert seen_max_age == [6.0]


# Cold-start's flat per-GW rate should be multiplied by the horizon, not left as a single-GW number.
def test_predict_horizon_points_scales_cold_start_rate_by_horizon(monkeypatch, tmp_path):
    monkeypatch.setattr(predict_module, "REGISTRY_PATH", tmp_path / "no_registry_here.json")
    monkeypatch.setattr(
        predict_module, "predict_points",
        lambda players, bootstrap, fixtures_current=None, teams_current=None, registry_path=None: (PLAYERS.copy(), True),
    )
    result, used_cold_start = predict_module.predict_horizon_points(PLAYERS, BOOTSTRAP_PRE_SEASON, horizon_gws=5)
    assert used_cold_start is True
    assert result["horizon_points"].iloc[0] == pytest.approx(20.0)


# Past cold-start with a registered model, horizon_points must sum per-GW trained-model predictions, with blank GWs zeroed.
def test_predict_horizon_points_sums_the_trained_model_across_the_horizon_and_zeroes_blank_gameweeks(monkeypatch, tmp_path):
    (tmp_path / "registry.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(predict_module, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(predict_module, "PROCESSED_DIR", tmp_path)
    pd.DataFrame(
        {"event": [10, 11], "team_h": [1, 1], "team_a": [2, 2], "team_h_difficulty": [3, 3], "team_a_difficulty": [3, 3]}
    ).to_parquet(tmp_path / "fixtures_current.parquet")
    monkeypatch.setattr(predict_module, "infer_current_gameweek", lambda bootstrap: 10)
    monkeypatch.setattr(predict_module, "fetch_player_histories", lambda players: {})

    bootstrap_mid_season = {
        "events": [{"id": i, "finished": True} for i in range(1, 6)],
        "teams": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
    }

    def fake_predict_points(players, bootstrap, player_histories=None, fixtures_current=None, teams_current=None, target_gameweek=None, registry_path=None, historical_fixtures=None, historical_teams=None, previous_season_form=None):
        if target_gameweek == 10:
            return pd.DataFrame([{"player_id": 1, "predicted_points": 5.0, "fixture_count": 1}]), False
        return pd.DataFrame([{"player_id": 1, "predicted_points": 3.0, "fixture_count": 0}]), False

    monkeypatch.setattr(predict_module, "predict_points", fake_predict_points)

    players = pd.DataFrame([{"player_id": 1, "web_name": "p1"}])
    predictions, used_cold_start = predict_module.predict_horizon_points(players, bootstrap_mid_season, horizon_gws=2)

    assert used_cold_start is False
    row = predictions.loc[predictions["player_id"] == 1].iloc[0]
    assert row["horizon_points"] == pytest.approx(5.0)  # GW10's 5.0 + GW11's blank-zeroed 0.0, not 5.0 + 3.0
    assert row["predicted_points"] == pytest.approx(2.5)  # horizon_points / horizon_gws


# A missing current/next event (season over) must fail clearly rather than loop with gw=None.
def test_predict_horizon_points_raises_when_the_season_has_no_current_or_next_gameweek(monkeypatch, tmp_path):
    (tmp_path / "registry.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(predict_module, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(predict_module, "infer_current_gameweek", lambda bootstrap: None)

    bootstrap_mid_season = {"events": [{"id": i, "finished": True} for i in range(1, 6)], "teams": []}
    with pytest.raises(ValueError):
        predict_module.predict_horizon_points(PLAYERS, bootstrap_mid_season, horizon_gws=5)
