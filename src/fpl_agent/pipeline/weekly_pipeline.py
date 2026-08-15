"""Single entrypoint wiring refresh -> news enrichment -> predict -> optimize -> cache, for live runs and historical replay.

Consolidates what was previously split across refresh_data.py, build_initial_squad.py, sync_team.py, and
scripts/replay_gameweek.py, so later phases (season_planner, the dashboard) have one real integration point
instead of several scripts glued together ad hoc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from fpl_agent.features.rolling_stats import add_minutes_volatility
from fpl_agent.ingestion import news_rss, news_scrape
from fpl_agent.models.evaluate import (
    HORIZON,
    MIN_SQUAD_FORMATION_GW,
    actual_points_following,
    build_horizon_points,
    target_season_player_pool,
    train_as_of,
)
from fpl_agent.models.predict import (
    REGISTRY_PATH,
    SETTINGS_PATH,
    completed_gameweeks_this_season,
    predict_points,
    should_use_cold_start,
)
from fpl_agent.models.train import feature_columns
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
from fpl_agent.utils.env import get_team_id

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _start_run(run_type: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Inserts a 'running' row into run_history and returns its id, so the dashboard can show run status/history later."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("INSERT INTO run_history (run_type, status) VALUES (?, 'running')", (run_type,))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _finish_run(run_id: int, status: str, details: str = "", db_path: Path = DEFAULT_DB_PATH) -> None:
    """Marks a run_history row finished with its outcome."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE run_history SET status = ?, details = ?, finished_at = datetime('now') WHERE id = ?",
            (status, details, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _load_horizon_gws() -> int:
    """Reads the configured near-term prediction horizon (in gameweeks) from settings.yaml."""
    with SETTINGS_PATH.open(encoding="utf-8") as f:
        return int(yaml.safe_load(f)["model"]["near_term_horizon_gws"])


def refresh_and_enrich_news() -> tuple[pd.DataFrame, dict]:
    """Pulls the latest player/fixture data, then layers RSS and scrape news on top; news failures don't abort the run."""
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


def predict_horizon_points(players: pd.DataFrame, bootstrap: dict, horizon_gws: int) -> tuple[pd.DataFrame, bool]:
    """Attaches `horizon_points`: the router's single-GW predicted_points held flat across the horizon.

    This matches the flat-rate assumption backtest_season.py's naive baseline uses. A true per-GW rolling horizon
    (like models.evaluate.build_horizon_points computes for historical replay) needs live per-player match
    histories via fpl_api.get_element_summary, which isn't wired into build_current_features yet - see the
    NotImplementedError below, which fires loudly instead of silently mispredicting once that in-season branch
    would otherwise be reached.
    """
    with SETTINGS_PATH.open(encoding="utf-8") as f:
        threshold = int(yaml.safe_load(f)["model"]["cold_start_gw_threshold"])
    completed = completed_gameweeks_this_season(bootstrap)
    if not should_use_cold_start(completed, threshold) and REGISTRY_PATH.exists():
        raise NotImplementedError(
            f"{completed} gameweeks completed this season (>= cold-start threshold {threshold}): the live "
            "in-season trained-model path needs a player_histories builder (fpl_api.get_element_summary per "
            "player, fed through build_current_features) that hasn't been wired up yet. Build that before "
            "relying on live recommendations past this point."
        )
    predictions, used_cold_start = predict_points(players, bootstrap)
    predictions = predictions.copy()
    predictions["horizon_points"] = predictions["predicted_points"] * horizon_gws
    return predictions, used_cold_start


def run_live(team_id: int | None = None) -> dict:
    """Runs the full live pipeline: refresh, news enrichment, team sync, and either an initial-squad or transfer/captaincy recommendation."""
    run_id = _start_run("weekly_pipeline_live")
    try:
        rules = load_rules()
        horizon_gws = _load_horizon_gws()
        players, bootstrap = refresh_and_enrich_news()
        resolved_team_id = team_id or get_team_id()
        snapshot = sync_team(resolved_team_id)

        if snapshot is None:
            squad, starting_xi, bench = build_initial_squad()
            _finish_run(run_id, "success", "pre-deadline: no live squad yet, returned initial squad recommendation")
            return {"mode": "initial_squad", "squad": squad, "starting_xi": starting_xi, "bench": bench}

        predictions, used_cold_start = predict_horizon_points(players, bootstrap, horizon_gws)
        picks_ids = {pick["player_id"] for pick in snapshot["picks"]}
        current_squad = predictions[predictions["player_id"].isin(picks_ids)].reset_index(drop=True)

        recommendation = recommend_transfers(
            current_squad, predictions, snapshot["bank"], snapshot["free_transfers"], rules
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
        }
        _finish_run(run_id, "success", f"GW{snapshot['gameweek']} live recommendation generated")
        return result
    except Exception as exc:
        _finish_run(run_id, "failed", str(exc))
        raise


def print_live_report(result: dict) -> None:
    """Prints a human-readable live recommendation report."""
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


def run_replay(target_season: str, as_of_gw: int, horizon: int = HORIZON) -> dict:
    """Replays a past gameweek: holds a squad formed from early-season predictions, then recommends a transfer and
    captain as of as_of_gw, alongside what actually happened - a human-checkable sanity test of the whole chain."""
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
        pool_h = pool.merge(horizon_points[["web_name", "horizon_points"]], on="web_name", how="left")
        pool_h["horizon_points"] = pool_h["horizon_points"].fillna(0)

        bank = rules.budget_million - squad["now_cost_million"].sum()
        free_transfers = rules.free_transfers_max_banked
        recommendation = recommend_transfers(squad_h, pool_h, bank, free_transfers, rules)

        xi_input = squad_h.assign(predicted_points=squad_h["horizon_points"] / horizon)
        starting_xi, _bench = select_starting_xi(xi_input, rules)
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
        }
        _finish_run(run_id, "success", f"replayed {target_season} GW{as_of_gw}")
        return result
    except Exception as exc:
        _finish_run(run_id, "failed", str(exc))
        raise


def print_replay_report(result: dict) -> None:
    """Prints a human-readable replay report, including what actually happened for sanity-checking."""
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--replay":
        replay_season = sys.argv[2] if len(sys.argv) > 2 else "2025-26"
        replay_gw = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        print_replay_report(run_replay(replay_season, replay_gw))
    else:
        print_live_report(run_live())
