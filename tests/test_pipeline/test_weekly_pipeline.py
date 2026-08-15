"""Checks weekly_pipeline's orchestration: horizon scaling (cold-start and trained-model), live-mode branching, and run logging."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_agent.optimizer.constraints import SquadRules
from fpl_agent.pipeline import weekly_pipeline
from fpl_agent.storage.db import get_connection, init_db

PLAYERS = pd.DataFrame([{"player_id": 1, "web_name": "p1", "predicted_points": 4.0}])
BOOTSTRAP_PRE_SEASON = {"events": [{"id": 1, "finished": False}]}
BOOTSTRAP_MID_SEASON = {
    "events": [{"id": i, "finished": True} for i in range(1, 6)],
    "teams": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
}

RULES = SquadRules(
    budget_million=100.0, squad_size=4,
    squad_composition={"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},
    starting_xi_size=4, formation_min={"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},
    max_players_per_club=2, captain_multiplier=2, triple_captain_multiplier=3,
    free_transfers_per_week=1, free_transfers_max_banked=5, points_hit_per_extra_transfer=-4,
)

PREDICTIONS = pd.DataFrame(
    [
        {"player_id": 1, "web_name": "gk1", "team_name": "A", "position": "GKP", "now_cost_million": 5.0, "horizon_points": 20.0},
        {"player_id": 2, "web_name": "def1", "team_name": "B", "position": "DEF", "now_cost_million": 5.0, "horizon_points": 15.0},
        {"player_id": 3, "web_name": "mid1", "team_name": "C", "position": "MID", "now_cost_million": 5.0, "horizon_points": 15.0},
        {"player_id": 4, "web_name": "fwd1", "team_name": "D", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 15.0},
        {"player_id": 5, "web_name": "fwd_great", "team_name": "E", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 40.0},
    ]
)

SNAPSHOT = {
    "gameweek": 5, "bank": 0.0, "squad_value": 20.0, "free_transfers": 1, "chips_used": [],
    "picks": [{"player_id": pid} for pid in (1, 2, 3, 4)],
}


def _raise(*_args, **_kwargs):
    raise RuntimeError("source unavailable")


def _no_op_run_logging(monkeypatch):
    monkeypatch.setattr(weekly_pipeline, "_start_run", lambda run_type: 1)
    monkeypatch.setattr(weekly_pipeline, "_finish_run", lambda *a, **k: None)


def test_predict_horizon_points_scales_cold_start_rate_by_horizon(monkeypatch):
    """Cold-start's flat per-GW rate should be multiplied by the horizon, not left as a single-GW number."""
    monkeypatch.setattr(weekly_pipeline, "predict_points", lambda players, bootstrap: (PLAYERS.copy(), True))
    result, used_cold_start = weekly_pipeline.predict_horizon_points(PLAYERS, BOOTSTRAP_PRE_SEASON, horizon_gws=5)
    assert used_cold_start is True
    assert result["horizon_points"].iloc[0] == pytest.approx(20.0)


def test_predict_horizon_points_sums_the_trained_model_across_the_horizon_and_zeroes_blank_gameweeks(monkeypatch, tmp_path):
    """Past cold-start with a registered model, horizon_points must sum per-GW trained-model predictions, with blank GWs zeroed."""
    (tmp_path / "registry.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(weekly_pipeline, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(weekly_pipeline, "PROCESSED_DIR", tmp_path)
    pd.DataFrame(
        {"event": [10, 11], "team_h": [1, 1], "team_a": [2, 2], "team_h_difficulty": [3, 3], "team_a_difficulty": [3, 3]}
    ).to_parquet(tmp_path / "fixtures_current.parquet")
    monkeypatch.setattr(weekly_pipeline.repository, "infer_current_gameweek", lambda bootstrap: 10)
    monkeypatch.setattr(weekly_pipeline, "fetch_player_histories", lambda players: {})

    def fake_predict_points(players, bootstrap, player_histories=None, fixtures_current=None, teams_current=None, target_gameweek=None):
        if target_gameweek == 10:
            return pd.DataFrame([{"player_id": 1, "predicted_points": 5.0, "fixture_count": 1}]), False
        return pd.DataFrame([{"player_id": 1, "predicted_points": 3.0, "fixture_count": 0}]), False

    monkeypatch.setattr(weekly_pipeline, "predict_points", fake_predict_points)

    players = pd.DataFrame([{"player_id": 1, "web_name": "p1"}])
    predictions, used_cold_start = weekly_pipeline.predict_horizon_points(players, BOOTSTRAP_MID_SEASON, horizon_gws=2)

    assert used_cold_start is False
    row = predictions.loc[predictions["player_id"] == 1].iloc[0]
    assert row["horizon_points"] == pytest.approx(5.0)  # GW10's 5.0 + GW11's blank-zeroed 0.0, not 5.0 + 3.0
    assert row["predicted_points"] == pytest.approx(2.5)  # horizon_points / horizon_gws


def test_predict_horizon_points_raises_when_the_season_has_no_current_or_next_gameweek(monkeypatch, tmp_path):
    """A missing current/next event (season over) must fail clearly rather than loop with gw=None."""
    (tmp_path / "registry.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(weekly_pipeline, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(weekly_pipeline.repository, "infer_current_gameweek", lambda bootstrap: None)

    with pytest.raises(ValueError):
        weekly_pipeline.predict_horizon_points(PLAYERS, BOOTSTRAP_MID_SEASON, horizon_gws=5)


def test_fetch_player_histories_renames_round_to_gw(monkeypatch):
    """The raw element-summary payload names the gameweek field 'round'; build_current_features expects 'GW'."""
    monkeypatch.setattr(
        weekly_pipeline.fpl_api, "get_element_summary",
        lambda player_id: {"history": [{"round": 3, "total_points": 6}, {"round": 4, "total_points": 2}]},
    )
    players = pd.DataFrame([{"player_id": 1}])

    histories = weekly_pipeline.fetch_player_histories(players)

    assert list(histories[1]["GW"]) == [3, 4]
    assert "round" not in histories[1].columns


def test_fetch_player_histories_handles_a_player_with_no_history_yet(monkeypatch):
    """A brand-new player with an empty history list must produce an empty frame, not raise."""
    monkeypatch.setattr(weekly_pipeline.fpl_api, "get_element_summary", lambda player_id: {"history": []})
    players = pd.DataFrame([{"player_id": 1}])

    histories = weekly_pipeline.fetch_player_histories(players)

    assert histories[1].empty


def test_refresh_and_enrich_news_continues_when_rss_and_scrape_both_fail(monkeypatch):
    """A flaky news source must not abort the whole refresh - the raw player/bootstrap data still comes back."""
    monkeypatch.setattr(weekly_pipeline.refresh_data, "run_refresh", lambda: (PLAYERS, BOOTSTRAP_PRE_SEASON))
    monkeypatch.setattr(weekly_pipeline.news_rss, "build_news_items", _raise)
    monkeypatch.setattr(weekly_pipeline.news_scrape, "fetch_injuries_page", _raise)

    result_players, result_bootstrap = weekly_pipeline.refresh_and_enrich_news()
    assert result_players is PLAYERS
    assert result_bootstrap is BOOTSTRAP_PRE_SEASON


def test_run_live_returns_initial_squad_recommendation_when_no_live_squad_yet(monkeypatch):
    """Pre-deadline (sync_team returns None), run_live must fall back to the initial-squad recommendation, not crash."""
    _no_op_run_logging(monkeypatch)
    squad = pd.DataFrame([{"web_name": "p1"}])
    monkeypatch.setattr(weekly_pipeline, "refresh_and_enrich_news", lambda: (PLAYERS, BOOTSTRAP_PRE_SEASON))
    monkeypatch.setattr(weekly_pipeline, "get_team_id", lambda: 123)
    monkeypatch.setattr(weekly_pipeline, "sync_team", lambda team_id: None)
    monkeypatch.setattr(weekly_pipeline, "build_initial_squad", lambda: (squad, squad, squad))

    result = weekly_pipeline.run_live()
    assert result["mode"] == "initial_squad"
    assert result["squad"] is squad


def test_run_live_recommends_a_transfer_and_captain_against_the_real_live_squad(monkeypatch):
    """With a live squad synced, run_live must wire real predictions through the real optimizer/captaincy logic."""
    _no_op_run_logging(monkeypatch)
    monkeypatch.setattr(weekly_pipeline, "refresh_and_enrich_news", lambda: (PREDICTIONS, BOOTSTRAP_PRE_SEASON))
    monkeypatch.setattr(weekly_pipeline, "get_team_id", lambda: 123)
    monkeypatch.setattr(weekly_pipeline, "sync_team", lambda team_id: SNAPSHOT)
    monkeypatch.setattr(weekly_pipeline, "load_rules", lambda: RULES)
    monkeypatch.setattr(
        weekly_pipeline, "predict_horizon_points", lambda players, bootstrap, horizon_gws: (PREDICTIONS.copy(), False)
    )

    result = weekly_pipeline.run_live(team_id=123)

    assert result["mode"] == "live"
    assert result["gameweek"] == 5
    assert result["recommendation"]["num_transfers"] == 1
    assert result["recommendation"]["bought"]["web_name"].iloc[0] == "fwd_great"
    assert result["captain"]["web_name"] == "fwd_great"


def test_run_live_logs_a_failed_run_and_reraises_on_error(monkeypatch):
    """A pipeline error must be recorded in run_history (for the future staleness indicator) and still propagate."""
    logged = {}
    monkeypatch.setattr(weekly_pipeline, "_start_run", lambda run_type: 7)
    monkeypatch.setattr(weekly_pipeline, "_finish_run", lambda run_id, status, details="": logged.update(run_id=run_id, status=status))
    monkeypatch.setattr(weekly_pipeline, "refresh_and_enrich_news", _raise)

    with pytest.raises(RuntimeError):
        weekly_pipeline.run_live(team_id=123)

    assert logged == {"run_id": 7, "status": "failed"}


def test_start_and_finish_run_round_trip_through_run_history(tmp_path):
    """_start_run/_finish_run must write a real, queryable run_history row - this is the table Phase 0 created but nothing wrote to."""
    db_path = tmp_path / "test.db"
    init_db(db_path)

    run_id = weekly_pipeline._start_run("weekly_pipeline_live", db_path=db_path)
    weekly_pipeline._finish_run(run_id, "success", "did the thing", db_path=db_path)

    conn = get_connection(db_path)
    try:
        row = dict(conn.execute("SELECT * FROM run_history WHERE id = ?", (run_id,)).fetchone())
    finally:
        conn.close()
    assert row["status"] == "success"
    assert row["details"] == "did the thing"
    assert row["finished_at"] is not None
