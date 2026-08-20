# Single entrypoint wiring refresh -> news enrichment -> predict -> optimize -> cache, for live runs and replay.

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from fpl_agent.features.build_features import team_gameweek_context
from fpl_agent.features.rolling_stats import add_minutes_volatility
from fpl_agent.ingestion import fpl_api, news_rss, news_scrape
from fpl_agent.models.evaluate import (
    HORIZON,
    MIN_SQUAD_FORMATION_GW,
    actual_points_following,
    build_horizon_points,
    target_season_player_pool,
    train_as_of,
)
from fpl_agent.models.live_adjustments import apply_session_overrides
from fpl_agent.models.predict import SETTINGS_PATH, load_fixtures_and_teams_current, predict_horizon_points
from fpl_agent.models.train import feature_columns
from fpl_agent.optimizer import chip_strategy, season_planner
from fpl_agent.optimizer.captaincy import captaincy_candidates, choose_captaincy
from fpl_agent.optimizer.constraints import load_rules
from fpl_agent.optimizer.squad_optimizer import select_squad, select_starting_xi
from fpl_agent.optimizer.transfer_optimizer import recommend_transfers
from fpl_agent.pipeline import refresh_data
from fpl_agent.pipeline.build_initial_squad import build_initial_squad, print_squad_report
from fpl_agent.pipeline.sync_team import sync_team
from fpl_agent.storage import repository
from fpl_agent.storage.db import DEFAULT_DB_PATH, get_connection
from fpl_agent.storage.repository import PROCESSED_DIR
from fpl_agent.utils.dates import load_chip_config
from fpl_agent.utils.env import get_team_id

SEASON_PLAN_LOOKAHEAD_GWS = 15
LAST_GAMEWEEK_OF_SEASON = 38

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Inserts a "running" row into run_history and returns its id, so the dashboard can show run status/history later.
def _start_run(run_type: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("INSERT INTO run_history (run_type, status) VALUES (?, 'running')", (run_type,))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


# Marks a run_history row finished with its outcome.
def _finish_run(run_id: int, status: str, details: str = "", db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE run_history SET status = ?, details = ?, finished_at = datetime('now') WHERE id = ?",
            (status, details, run_id),
        )
        conn.commit()
    finally:
        conn.close()


# Reads the configured near-term prediction horizon (in gameweeks) from settings.yaml.
def _load_horizon_gws() -> int:
    with SETTINGS_PATH.open(encoding="utf-8") as f:
        return int(yaml.safe_load(f)["model"]["near_term_horizon_gws"])


# Pulls the latest player/fixture data, then layers RSS and scrape news on top; news failures don't abort the run.
def refresh_and_enrich_news() -> tuple[pd.DataFrame, dict]:
    players, bootstrap = refresh_data.run_refresh()
    try:
        rss_rows = news_rss.build_news_items(players)
        if not rss_rows.empty:
            repository.enrich_player_status_from_rss(rss_rows)
    except Exception as exc:
        print(f"RSS enrichment failed ({exc}); continuing with existing player_status.", file=sys.stderr)
    try:
        html = news_scrape.fetch_injuries_page()
        scrape_rows = news_scrape.parse_injuries(html, players)
        if not scrape_rows.empty:
            repository.enrich_player_status_from_scrape(scrape_rows)
    except Exception as exc:
        print(f"Scrape enrichment failed ({exc}); continuing with existing player_status.", file=sys.stderr)
    return players, bootstrap


# Fetches standings for the first configured mini_league_id, or (None, None) if none are configured. mini_league_ids overrides the shared settings.yaml value - used to pass a visitor's own session-scoped league IDs in the multi-user path.
def _load_mini_league_standings(
    mini_league_ids: list[int] | None = None, settings_path: Path = SETTINGS_PATH
) -> tuple[pd.DataFrame | None, int | None]:
    league_ids = mini_league_ids
    if league_ids is None:
        with settings_path.open(encoding="utf-8") as f:
            league_ids = yaml.safe_load(f)["team"].get("mini_league_ids") or []
    if not league_ids:
        return None, None
    league_id = int(league_ids[0])
    try:
        raw = fpl_api.get_league_standings(league_id)
    except Exception as exc:
        print(f"Mini-league standings fetch failed ({exc}); continuing without rank-aware risk posture.", file=sys.stderr)
        return None, None
    return repository.normalize_league_standings(raw, league_id), league_id


# Builds the season plan and, from it, a rule-aware chip suggestion report - shared by run_live and run_replay. mini_league_ids overrides the shared settings.yaml value (see _load_mini_league_standings).
def _build_season_plan_and_chip_suggestions(
    squad: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, current_gw: int, team_id: int | None,
    chips_used: list[dict], mini_league_ids: list[int] | None = None,
) -> tuple[season_planner.SeasonPlan, dict]:
    gw_range = range(current_gw, min(current_gw + SEASON_PLAN_LOOKAHEAD_GWS, LAST_GAMEWEEK_OF_SEASON + 1))
    context = team_gameweek_context(fixtures, teams)
    standings, _league_id = (_load_mini_league_standings(mini_league_ids) if team_id is not None else (None, None))

    plan = season_planner.build_season_plan(
        squad, context, context, gw_range, standings=standings, team_id=team_id,
        gws_remaining_in_season=LAST_GAMEWEEK_OF_SEASON + 1 - current_gw,
    )
    chip_config = load_chip_config()
    suggestions = chip_strategy.build_chip_suggestions(
        gw_range, plan.chip_window_scores, plan.wildcard_signal, chips_used, current_gw, chip_config["first_half_deadline_gw"]
    )
    return plan, suggestions


# The shared, expensive half of a live run: refresh + news enrichment + the full player-pool prediction, independent of any one visitor's team. Meant to be run once (locally, by the app owner) and cached for every visitor to read - see pipeline/recommendation_cache.py's shared-cache functions.
def refresh_shared_predictions(horizon_gws: int | None = None) -> dict:
    horizon_gws = horizon_gws or _load_horizon_gws()
    players, bootstrap = refresh_and_enrich_news()
    predictions, used_cold_start = predict_horizon_points(players, bootstrap, horizon_gws)
    fixtures_current, teams_current = load_fixtures_and_teams_current(bootstrap)
    return {
        "players": players, "bootstrap": bootstrap, "predictions": predictions, "used_cold_start": used_cold_start,
        "fixtures_current": fixtures_current, "teams_current": teams_current, "horizon_gws": horizon_gws,
    }


# The cheap, per-visitor half of a live run: syncs this visitor's real squad and builds their transfer/captaincy/chip recommendation against the already-computed shared predictions. Fast enough to call live on every page load - no API calls beyond sync_team's 3 lightweight ones, no model inference. differential_aggressiveness/mini_league_ids/session_overrides are that visitor's own session-scoped preferences (see ui/components/session.py), never shared settings.yaml/global state.
def build_visitor_recommendation(
    team_id: int, shared: dict, mini_league_ids: list[int] | None = None, session_overrides: dict[int, dict] | None = None,
    rules=None,
) -> dict:
    rules = rules or load_rules()
    horizon_gws = shared["horizon_gws"]
    predictions = apply_session_overrides(shared["predictions"], session_overrides or {})
    snapshot = sync_team(team_id)

    if snapshot is None:
        squad, starting_xi, bench, all_players, used_cold_start = build_initial_squad()
        return {
            "mode": "initial_squad", "squad": squad, "starting_xi": starting_xi, "bench": bench,
            "all_players": all_players, "used_cold_start": used_cold_start,
        }

    picks_ids = {pick["player_id"] for pick in snapshot["picks"]}
    current_squad = predictions[predictions["player_id"].isin(picks_ids)].reset_index(drop=True)

    season_plan, chip_suggestions = _build_season_plan_and_chip_suggestions(
        current_squad, shared["fixtures_current"], shared["teams_current"], snapshot["gameweek"], team_id,
        snapshot["chips_used"], mini_league_ids=mini_league_ids,
    )

    recommendation = recommend_transfers(
        current_squad, predictions, snapshot["bank"], snapshot["free_transfers"], rules,
        core_player_ids=season_plan.core_player_ids,
    )
    kept = current_squad[~current_squad["player_id"].isin(recommendation["dropped"]["player_id"])]
    resulting_squad = pd.concat([kept, recommendation["bought"]], ignore_index=True)
    xi_input = resulting_squad.assign(predicted_points=resulting_squad["horizon_points"] / horizon_gws)
    starting_xi, bench = select_starting_xi(xi_input, rules)
    captain, vice_captain = choose_captaincy(starting_xi)

    return {
        "mode": "live",
        "gameweek": snapshot["gameweek"],
        "used_cold_start": shared["used_cold_start"],
        "bank": snapshot["bank"],
        "free_transfers": snapshot["free_transfers"],
        "recommendation": recommendation,
        "starting_xi": starting_xi,
        "bench": bench,
        "captain": captain,
        "vice_captain": vice_captain,
        "season_plan": season_plan,
        "chip_suggestions": chip_suggestions,
        "all_players": predictions,
    }


# Runs the full live pipeline: refresh, news enrichment, team sync, and either an initial-squad or transfer/captaincy recommendation.
def run_live(team_id: int | None = None) -> dict:
    run_id = _start_run("weekly_pipeline_live")
    try:
        rules = load_rules()
        horizon_gws = _load_horizon_gws()
        players, bootstrap = refresh_and_enrich_news()
        resolved_team_id = team_id or get_team_id()
        snapshot = sync_team(resolved_team_id)

        if snapshot is None:
            squad, starting_xi, bench, all_players, used_cold_start = build_initial_squad()
            _finish_run(run_id, "success", "pre-deadline: no live squad yet, returned initial squad recommendation")
            return {
                "mode": "initial_squad", "squad": squad, "starting_xi": starting_xi, "bench": bench,
                "all_players": all_players, "used_cold_start": used_cold_start,
            }

        predictions, used_cold_start = predict_horizon_points(players, bootstrap, horizon_gws)
        picks_ids = {pick["player_id"] for pick in snapshot["picks"]}
        current_squad = predictions[predictions["player_id"].isin(picks_ids)].reset_index(drop=True)

        fixtures_current, teams_current = load_fixtures_and_teams_current(bootstrap)
        season_plan, chip_suggestions = _build_season_plan_and_chip_suggestions(
            current_squad, fixtures_current, teams_current, snapshot["gameweek"], resolved_team_id, snapshot["chips_used"]
        )

        recommendation = recommend_transfers(
            current_squad, predictions, snapshot["bank"], snapshot["free_transfers"], rules,
            core_player_ids=season_plan.core_player_ids,
        )
        kept = current_squad[~current_squad["player_id"].isin(recommendation["dropped"]["player_id"])]
        resulting_squad = pd.concat([kept, recommendation["bought"]], ignore_index=True)
        xi_input = resulting_squad.assign(predicted_points=resulting_squad["horizon_points"] / horizon_gws)
        starting_xi, bench = select_starting_xi(xi_input, rules)
        captain, vice_captain = choose_captaincy(starting_xi)

        result = {
            "mode": "live",
            "gameweek": snapshot["gameweek"],
            "used_cold_start": used_cold_start,
            "bank": snapshot["bank"],
            "free_transfers": snapshot["free_transfers"],
            "recommendation": recommendation,
            "starting_xi": starting_xi,
            "bench": bench,
            "captain": captain,
            "vice_captain": vice_captain,
            "season_plan": season_plan,
            "chip_suggestions": chip_suggestions,
            "all_players": predictions,
        }
        _finish_run(run_id, "success", f"GW{snapshot['gameweek']} live recommendation generated")
        return result
    except Exception as exc:
        _finish_run(run_id, "failed", str(exc))
        raise


# Prints a human-readable live recommendation report.
def print_live_report(result: dict) -> None:
    if result["mode"] == "initial_squad":
        print("Your live squad isn't available yet (the first gameweek deadline hasn't passed).")
        print("Here's a recommended squad to enter yourself before the deadline:\n")
        print_squad_report(result["squad"], result["starting_xi"], result["bench"])
        return

    print(f"GW{result['gameweek']} live recommendation")
    print(f"Using {'pre-season cold-start priors' if result['used_cold_start'] else 'the trained prediction model'}.")
    print(f"Bank: £{result['bank']:.1f}m | Free transfers: {result['free_transfers']}\n")

    print("Transfer recommendation:")
    for line in result["recommendation"]["reasoning"]:
        print(f"  - {line}")

    print(f"\nCaptain: {result['captain']['web_name']} ({result['captain']['predicted_points']:.1f} pts)")
    print(f"Vice-captain: {result['vice_captain']['web_name']} ({result['vice_captain']['predicted_points']:.1f} pts)")

    print_season_plan_and_chip_suggestions(result["season_plan"], result["chip_suggestions"])


# Prints the season plan's risk posture and the upcoming chip suggestions - shared by live and replay reports.
def print_season_plan_and_chip_suggestions(plan, chip_report: dict) -> None:
    print(f"\nMini-league posture: {plan.risk_posture['posture']} - {plan.risk_posture['reasoning']}")
    if plan.wildcard_signal.get("suggest_wildcard"):
        print(f"Wildcard signal: fixtures notably worse than average (gap {plan.wildcard_signal['gap']:.2f}).")

    upcoming = [s for s in chip_report["suggestions"] if s["chip"]]
    if upcoming:
        print("Chip suggestions:")
        for s in upcoming:
            print(f"  GW{s['GW']}: {s['chip_label']} - {s['reasoning']}")
    else:
        print("Chip suggestions: none of the scanned gameweeks meet the suggestion threshold.")
    if chip_report.get("urgency_warning"):
        print(f"  ! {chip_report['urgency_warning']}")


# Replays a past gameweek: recommends a transfer and captain as of as_of_gw, against what actually happened.
def run_replay(target_season: str, as_of_gw: int, horizon: int = HORIZON) -> dict:
    run_id = _start_run("weekly_pipeline_replay")
    try:
        rules = load_rules()
        table = pd.read_parquet(PROCESSED_DIR / "training_table.parquet")
        columns = feature_columns(table)

        first_gw = MIN_SQUAD_FORMATION_GW
        early_models = train_as_of(table, target_season, first_gw, columns)
        first_gw_points = build_horizon_points(table, target_season, first_gw, 1, early_models, columns)
        pool = target_season_player_pool(table, target_season, first_gw)
        candidates = pool.merge(first_gw_points[["web_name", "horizon_points"]], on="web_name", how="inner")
        candidates = candidates.rename(columns={"horizon_points": "predicted_points"})
        squad = select_squad(candidates, rules)
        squad["player_id"] = squad["web_name"]
        pool["player_id"] = pool["web_name"]

        models = train_as_of(table, target_season, as_of_gw, columns)
        horizon_points = build_horizon_points(table, target_season, as_of_gw, horizon, models, columns)

        squad_h = squad.merge(horizon_points[["web_name", "horizon_points"]], on="web_name", how="left")
        squad_h["horizon_points"] = squad_h["horizon_points"].fillna(0)
        squad_h["predicted_points"] = squad_h["horizon_points"] / horizon
        pool_h = pool.merge(horizon_points[["web_name", "horizon_points"]], on="web_name", how="left")
        pool_h["horizon_points"] = pool_h["horizon_points"].fillna(0)

        fixtures = pd.read_parquet(PROCESSED_DIR / "all_seasons_fixtures.parquet")
        teams = pd.read_parquet(PROCESSED_DIR / "all_seasons_teams.parquet")
        season_plan, chip_suggestions = _build_season_plan_and_chip_suggestions(
            squad_h, fixtures[fixtures["season"] == target_season], teams[teams["season"] == target_season],
            as_of_gw, team_id=None, chips_used=[],
        )

        bank = rules.budget_million - squad["now_cost_million"].sum()
        free_transfers = rules.free_transfers_max_banked
        recommendation = recommend_transfers(squad_h, pool_h, bank, free_transfers, rules, core_player_ids=season_plan.core_player_ids)

        starting_xi, _bench = select_starting_xi(squad_h, rules)
        vol_table = add_minutes_volatility(table[table["season"] == target_season], group_keys=["season", "element"])
        latest_vol = vol_table[vol_table["GW"] == as_of_gw - 1][["name", "minutes_volatility"]]
        starting_xi = starting_xi.merge(latest_vol, left_on="web_name", right_on="name", how="left")
        starting_xi["chance_of_playing_next_round"] = 100
        ranked = captaincy_candidates(starting_xi)

        involved = list(recommendation["dropped"]["web_name"]) + list(recommendation["bought"]["web_name"])
        actual_transfer_points = actual_points_following(table, target_season, involved, as_of_gw, horizon)
        actual_captain_points = actual_points_following(table, target_season, ranked["web_name"].tolist(), as_of_gw, 1)

        result = {
            "target_season": target_season,
            "as_of_gw": as_of_gw,
            "first_gw": first_gw,
            "horizon": horizon,
            "squad": squad,
            "recommendation": recommendation,
            "captaincy_ranked": ranked,
            "actual_transfer_points": actual_transfer_points,
            "actual_captain_points": actual_captain_points,
            "season_plan": season_plan,
            "chip_suggestions": chip_suggestions,
        }
        _finish_run(run_id, "success", f"replayed {target_season} GW{as_of_gw}")
        return result
    except Exception as exc:
        _finish_run(run_id, "failed", str(exc))
        raise


# Prints a human-readable replay report, including what actually happened for sanity-checking.
def print_replay_report(result: dict) -> None:
    target_season, as_of_gw, horizon = result["target_season"], result["as_of_gw"], result["horizon"]
    squad, recommendation = result["squad"], result["recommendation"]

    print(f"Replaying {target_season}, standing right before GW{as_of_gw}...\n")
    print(f"Squad held since GW{result['first_gw']}, {len(squad)} players, £{squad['now_cost_million'].sum():.1f}m.\n")

    print("Transfer recommendation:")
    for line in recommendation["reasoning"]:
        print(f"  - {line}")

    if recommendation["num_transfers"] > 0:
        involved = list(recommendation["dropped"]["web_name"]) + list(recommendation["bought"]["web_name"])
        print(f"\nWhat actually happened over the following {horizon} gameweeks:")
        for name in involved:
            print(f"  {name}: {result['actual_transfer_points'].get(name, 0):.0f} actual points")

    print(f"\nCaptaincy for GW{as_of_gw}:")
    ranked = result["captaincy_ranked"]
    print(ranked[["web_name", "predicted_points", "risk_score"]].head(3).to_string(index=False))

    print(f"\nWhat actually happened in GW{as_of_gw}:")
    for name in ranked["web_name"].head(3):
        print(f"  {name}: {result['actual_captain_points'].get(name, 0):.0f} actual points")

    print_season_plan_and_chip_suggestions(result["season_plan"], result["chip_suggestions"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--replay":
        replay_season = sys.argv[2] if len(sys.argv) > 2 else "2025-26"
        replay_gw = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        print_replay_report(run_replay(replay_season, replay_gw))
    else:
        print_live_report(run_live())
