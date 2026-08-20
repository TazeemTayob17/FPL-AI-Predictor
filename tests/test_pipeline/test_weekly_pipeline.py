# Checks weekly_pipeline's orchestration: horizon scaling (cold-start and trained-model), live-mode branching, and run logging.

from __future__ import annotations

import pandas as pd
import pytest

from fpl_agent.optimizer.constraints import SquadRules
from fpl_agent.optimizer.season_planner import SeasonPlan
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

EMPTY_CHIP_SCORES = pd.DataFrame(columns=["GW", "bench_boost_score", "free_hit_score", "triple_captain_candidate", "triple_captain_score"])
BALANCED_POSTURE = {"posture": "balanced", "rank": None, "points_behind_leader": None, "reasoning": "no league configured"}
FAKE_PLAN = SeasonPlan(
    squad_classified=pd.DataFrame([{"player_id": pid, "classification": "rotational"} for pid in (1, 2, 3, 4)]),
    chip_window_scores=EMPTY_CHIP_SCORES, wildcard_signal={"suggest_wildcard": False, "gap": 0.0}, risk_posture=BALANCED_POSTURE,
)
FAKE_CHIP_SUGGESTIONS = {"available_chips": set(), "suggestions": [], "urgency_warning": None}

# no-op helpers used across the tests below
def _raise(*_args, **_kwargs):
    raise RuntimeError("source unavailable")

def _no_op_run_logging(monkeypatch):
    monkeypatch.setattr(weekly_pipeline, "_start_run", lambda run_type: 1)
    monkeypatch.setattr(weekly_pipeline, "_finish_run", lambda *a, **k: None)


# A flaky news source must not abort the whole refresh - the raw player/bootstrap data still comes back.
def test_refresh_and_enrich_news_continues_when_rss_and_scrape_both_fail(monkeypatch):
    monkeypatch.setattr(weekly_pipeline.refresh_data, "run_refresh", lambda: (PLAYERS, BOOTSTRAP_PRE_SEASON))
    monkeypatch.setattr(weekly_pipeline.news_rss, "build_news_items", _raise)
    monkeypatch.setattr(weekly_pipeline.news_scrape, "fetch_injuries_page", _raise)

    result_players, result_bootstrap = weekly_pipeline.refresh_and_enrich_news()
    assert result_players is PLAYERS
    assert result_bootstrap is BOOTSTRAP_PRE_SEASON


# Pre-deadline (sync_team returns None), run_live must fall back to the initial-squad recommendation, not crash.
def test_run_live_returns_initial_squad_recommendation_when_no_live_squad_yet(monkeypatch):
    _no_op_run_logging(monkeypatch)
    squad = pd.DataFrame([{"web_name": "p1"}])
    all_players = pd.DataFrame([{"web_name": "p1"}, {"web_name": "p2"}])
    monkeypatch.setattr(weekly_pipeline, "refresh_and_enrich_news", lambda: (PLAYERS, BOOTSTRAP_PRE_SEASON))
    monkeypatch.setattr(weekly_pipeline, "get_team_id", lambda: 123)
    monkeypatch.setattr(weekly_pipeline, "sync_team", lambda team_id: None)
    monkeypatch.setattr(weekly_pipeline, "build_initial_squad", lambda: (squad, squad, squad, all_players, True))

    result = weekly_pipeline.run_live()
    assert result["mode"] == "initial_squad"
    assert result["squad"] is squad
    assert result["all_players"] is all_players
    assert result["used_cold_start"] is True


# With a live squad synced, run_live must wire real predictions through the real optimizer/captaincy logic.
def test_run_live_recommends_a_transfer_and_captain_against_the_real_live_squad(monkeypatch):
    _no_op_run_logging(monkeypatch)
    monkeypatch.setattr(weekly_pipeline, "refresh_and_enrich_news", lambda: (PREDICTIONS, BOOTSTRAP_PRE_SEASON))
    monkeypatch.setattr(weekly_pipeline, "get_team_id", lambda: 123)
    monkeypatch.setattr(weekly_pipeline, "sync_team", lambda team_id: SNAPSHOT)
    monkeypatch.setattr(weekly_pipeline, "load_rules", lambda: RULES)
    monkeypatch.setattr(
        weekly_pipeline, "predict_horizon_points", lambda players, bootstrap, horizon_gws: (PREDICTIONS.copy(), False)
    )
    monkeypatch.setattr(weekly_pipeline, "load_fixtures_and_teams_current", lambda bootstrap: (None, None))
    monkeypatch.setattr(weekly_pipeline, "_build_season_plan_and_chip_suggestions", lambda *a, **k: (FAKE_PLAN, FAKE_CHIP_SUGGESTIONS))

    result = weekly_pipeline.run_live(team_id=123)

    assert result["mode"] == "live"
    assert result["gameweek"] == 5
    assert result["recommendation"]["num_transfers"] == 1
    assert result["recommendation"]["bought"]["web_name"].iloc[0] == "fwd_great"
    assert result["captain"]["web_name"] == "fwd_great"
    assert result["season_plan"] is FAKE_PLAN
    assert result["chip_suggestions"] is FAKE_CHIP_SUGGESTIONS


# core_player_ids must actually reach recommend_transfers - a marginal core-player sale must get blocked.
def test_run_live_threads_core_player_ids_into_the_transfer_recommendation(monkeypatch):
    _no_op_run_logging(monkeypatch)
    marginal_predictions = PREDICTIONS.copy()
    marginal_predictions.loc[marginal_predictions["web_name"] == "fwd_great", "horizon_points"] = 16.0  # was 40.0

    plan_with_fwd1_core = SeasonPlan(
        squad_classified=pd.DataFrame([{"player_id": pid, "classification": "core" if pid == 4 else "rotational"} for pid in (1, 2, 3, 4)]),
        chip_window_scores=EMPTY_CHIP_SCORES, wildcard_signal={"suggest_wildcard": False, "gap": 0.0}, risk_posture=BALANCED_POSTURE,
    )

    monkeypatch.setattr(weekly_pipeline, "refresh_and_enrich_news", lambda: (marginal_predictions, BOOTSTRAP_PRE_SEASON))
    monkeypatch.setattr(weekly_pipeline, "get_team_id", lambda: 123)
    monkeypatch.setattr(weekly_pipeline, "sync_team", lambda team_id: SNAPSHOT)
    monkeypatch.setattr(weekly_pipeline, "load_rules", lambda: RULES)
    monkeypatch.setattr(
        weekly_pipeline, "predict_horizon_points", lambda players, bootstrap, horizon_gws: (marginal_predictions.copy(), False)
    )
    monkeypatch.setattr(weekly_pipeline, "load_fixtures_and_teams_current", lambda bootstrap: (None, None))
    monkeypatch.setattr(
        weekly_pipeline, "_build_season_plan_and_chip_suggestions", lambda *a, **k: (plan_with_fwd1_core, FAKE_CHIP_SUGGESTIONS)
    )

    result = weekly_pipeline.run_live(team_id=123)

    assert result["recommendation"]["num_transfers"] == 0  # blocked: marginal +1 gain doesn't clear the 2.0 pt core penalty


# Integration check (not mocked) that team_gameweek_context's real output columns match what's expected downstream.
def test_build_season_plan_and_chip_suggestions_wires_real_column_names_correctly(monkeypatch):
    monkeypatch.setattr(weekly_pipeline, "_load_mini_league_standings", lambda: (None, None))

    squad = pd.DataFrame(
        [
            {
                "player_id": 1, "web_name": "p1", "team_id": 1, "position": "FWD", "predicted_points": 6.0,
                "now_cost_million": 8.0, "chance_of_playing_next_round": 100,
            }
        ]
    )
    fixtures = pd.DataFrame({"event": [5, 6], "team_h": [1, 1], "team_a": [2, 2], "team_h_difficulty": [3, 3], "team_a_difficulty": [3, 3]})
    teams = pd.DataFrame(
        {
            "id": [1, 2], "strength_attack_home": [1000, 1000], "strength_attack_away": [1000, 1000],
            "strength_defence_home": [1000, 1000], "strength_defence_away": [1000, 1000],
        }
    )

    plan, chip_report = weekly_pipeline._build_season_plan_and_chip_suggestions(squad, fixtures, teams, current_gw=5, team_id=None, chips_used=[])

    assert set(plan.squad_classified["classification"]) <= {"core", "rotational"}
    assert list(plan.chip_window_scores["GW"])[:2] == [5, 6]
    assert len(chip_report["suggestions"]) == len(plan.chip_window_scores)


# A pipeline error must be recorded in run_history (for the future staleness indicator) and still propagate.
def test_run_live_logs_a_failed_run_and_reraises_on_error(monkeypatch):
    logged = {}
    monkeypatch.setattr(weekly_pipeline, "_start_run", lambda run_type: 7)
    monkeypatch.setattr(weekly_pipeline, "_finish_run", lambda run_id, status, details="": logged.update(run_id=run_id, status=status))
    monkeypatch.setattr(weekly_pipeline, "refresh_and_enrich_news", _raise)

    with pytest.raises(RuntimeError):
        weekly_pipeline.run_live(team_id=123)

    assert logged == {"run_id": 7, "status": "failed"}


# refresh_shared_predictions must bundle everything the per-visitor step needs, without depending on any one team.
def test_refresh_shared_predictions_returns_the_expected_shared_bundle(monkeypatch):
    monkeypatch.setattr(weekly_pipeline, "_load_horizon_gws", lambda: 5)
    monkeypatch.setattr(weekly_pipeline, "refresh_and_enrich_news", lambda: (PLAYERS, BOOTSTRAP_PRE_SEASON))
    monkeypatch.setattr(weekly_pipeline, "predict_horizon_points", lambda players, bootstrap, horizon_gws: (PREDICTIONS.copy(), True))
    monkeypatch.setattr(weekly_pipeline, "load_fixtures_and_teams_current", lambda bootstrap: ("fixtures", "teams"))

    shared = weekly_pipeline.refresh_shared_predictions()

    assert shared["used_cold_start"] is True
    assert shared["horizon_gws"] == 5
    assert shared["fixtures_current"] == "fixtures"
    assert shared["teams_current"] == "teams"
    assert list(shared["predictions"]["web_name"]) == list(PREDICTIONS["web_name"])


SHARED = {
    "predictions": PREDICTIONS, "used_cold_start": False, "fixtures_current": None, "teams_current": None, "horizon_gws": 5,
}


# Pre-deadline (sync_team returns None), build_visitor_recommendation must fall back to the initial-squad recommendation, exactly like run_live.
def test_build_visitor_recommendation_returns_initial_squad_when_no_live_squad_yet(monkeypatch):
    squad = pd.DataFrame([{"web_name": "p1"}])
    all_players = pd.DataFrame([{"web_name": "p1"}, {"web_name": "p2"}])
    monkeypatch.setattr(weekly_pipeline, "sync_team", lambda team_id: None)
    monkeypatch.setattr(weekly_pipeline, "build_initial_squad", lambda: (squad, squad, squad, all_players, True))

    result = weekly_pipeline.build_visitor_recommendation(123, SHARED)
    assert result["mode"] == "initial_squad"
    assert result["squad"] is squad


# With a live squad synced, build_visitor_recommendation must produce the same shape of live recommendation run_live does, but from the already-computed shared predictions (no predict_horizon_points call).
def test_build_visitor_recommendation_builds_a_live_recommendation_from_shared_predictions(monkeypatch):
    monkeypatch.setattr(weekly_pipeline, "sync_team", lambda team_id: SNAPSHOT)
    monkeypatch.setattr(weekly_pipeline, "load_rules", lambda: RULES)
    captured = {}

    def fake_plan(*args, **kwargs):
        captured["mini_league_ids"] = kwargs.get("mini_league_ids")
        return FAKE_PLAN, FAKE_CHIP_SUGGESTIONS

    monkeypatch.setattr(weekly_pipeline, "_build_season_plan_and_chip_suggestions", fake_plan)

    result = weekly_pipeline.build_visitor_recommendation(123, SHARED, mini_league_ids=[9999])

    assert result["mode"] == "live"
    assert result["recommendation"]["bought"]["web_name"].iloc[0] == "fwd_great"
    assert result["captain"]["web_name"] == "fwd_great"
    assert captured["mini_league_ids"] == [9999]  # this visitor's own league IDs reached the season plan, not settings.yaml's


# A visitor's session override must actually change the predictions their recommendation is built from - the whole point of it being per-visitor.
def test_build_visitor_recommendation_applies_this_visitors_session_overrides(monkeypatch):
    monkeypatch.setattr(weekly_pipeline, "sync_team", lambda team_id: SNAPSHOT)
    monkeypatch.setattr(weekly_pipeline, "load_rules", lambda: RULES)
    monkeypatch.setattr(weekly_pipeline, "_build_season_plan_and_chip_suggestions", lambda *a, **k: (FAKE_PLAN, FAKE_CHIP_SUGGESTIONS))

    doubted = weekly_pipeline.build_visitor_recommendation(123, SHARED, session_overrides={5: {"chance_of_playing_next_round": "0"}})
    # fwd_great (player_id 5) is ruled out by this visitor's own doubt, so its points collapse to 0 in their view only.
    assert doubted["all_players"].loc[doubted["all_players"]["player_id"] == 5, "horizon_points"].iloc[0] == 0.0

    baseline = weekly_pipeline.build_visitor_recommendation(123, SHARED)
    assert baseline["all_players"].loc[baseline["all_players"]["player_id"] == 5, "horizon_points"].iloc[0] == 40.0


# _start_run/_finish_run must write a real, queryable run_history row - this is the table Phase 0 created but nothing wrote to.
def test_start_and_finish_run_round_trip_through_run_history(tmp_path):
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
