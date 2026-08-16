# Routes point prediction to cold-start priors pre-season, or the trained LightGBM models once enough data exists.

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

from fpl_agent.features.build_features import build_current_features
from fpl_agent.ingestion import fpl_api
from fpl_agent.models.cold_start import build_opening_difficulty, predict_cold_start_points
from fpl_agent.storage.repository import PROCESSED_DIR, infer_current_gameweek

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"
PLAYER_HISTORY_FRESHNESS_HOURS = 20.0  # reuse a same-day fetch instead of re-pulling all ~600 players on every manual run


# Counts how many of this season's gameweeks have finished, to decide whether cold-start or the trained model applies.
def completed_gameweeks_this_season(bootstrap: dict) -> int:
    return sum(1 for event in bootstrap.get("events", []) if event.get("finished"))


# True until enough current-season gameweeks exist for the trained model's rolling features to be meaningful.
def should_use_cold_start(completed_gameweeks: int, threshold: int) -> bool:
    return completed_gameweeks < threshold


# Loads each position's trained model and its expected feature list from the registry.
def load_position_models(registry_path: Path = REGISTRY_PATH) -> dict[str, tuple[lgb.Booster, list[str]]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    models = {}
    for position, info in registry.items():
        model_path = registry_path.parent / "artifacts" / info["model_file"]
        models[position] = (lgb.Booster(model_file=str(model_path)), info["features"])
    return models


# Runs each position's model over its rows and attaches a `predicted_points` column.
def predict_with_trained_models(features: pd.DataFrame, models: dict[str, tuple[lgb.Booster, list[str]]]) -> pd.DataFrame:
    features = features.copy()
    features["predicted_points"] = 0.0
    for position, (model, columns) in models.items():
        mask = features["position"] == position
        if mask.any():
            features.loc[mask, "predicted_points"] = model.predict(features.loc[mask, columns])
    return features


# Returns (players_with_predicted_points, used_cold_start), routing per the configured cold-start GW threshold.
def predict_points(
    players: pd.DataFrame,
    bootstrap: dict,
    player_histories: dict[int, pd.DataFrame] | None = None,
    fixtures_current: pd.DataFrame | None = None,
    teams_current: pd.DataFrame | None = None,
    target_gameweek: int | None = None,
    registry_path: Path = REGISTRY_PATH,
) -> tuple[pd.DataFrame, bool]:
    with SETTINGS_PATH.open(encoding="utf-8") as f:
        threshold = yaml.safe_load(f)["model"]["cold_start_gw_threshold"]
    completed = completed_gameweeks_this_season(bootstrap)

    if should_use_cold_start(completed, threshold) or not registry_path.exists():
        opening_difficulty = None
        if fixtures_current is not None and teams_current is not None:
            opening_difficulty = build_opening_difficulty(fixtures_current, teams_current)
        return predict_cold_start_points(players, opening_difficulty=opening_difficulty), True

    if player_histories is None or fixtures_current is None or teams_current is None or target_gameweek is None:
        raise ValueError("player_histories, fixtures_current, teams_current, and target_gameweek are required past cold-start.")

    models = load_position_models(registry_path)
    features = build_current_features(player_histories, players, fixtures_current, teams_current, target_gameweek)
    return predict_with_trained_models(features, models), False


# Fetches each current player's this-season match history, reusing same-day cached pulls to avoid ~600 calls per run.
def fetch_player_histories(players: pd.DataFrame, max_age_hours: float = PLAYER_HISTORY_FRESHNESS_HOURS) -> dict[int, pd.DataFrame]:
    histories = {}
    for player_id in players["player_id"]:
        summary = fpl_api.get_element_summary(int(player_id), max_age_hours=max_age_hours)
        history = pd.DataFrame(summary.get("history", []))
        if not history.empty:
            history = history.rename(columns={"round": "GW"})
        histories[int(player_id)] = history
    return histories


# Loads the current season's fixtures/teams if available, for opening-fixture-difficulty or live rolling context.
def load_fixtures_and_teams_current(bootstrap: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    fixtures_path = PROCESSED_DIR / "fixtures_current.parquet"
    fixtures_current = pd.read_parquet(fixtures_path) if fixtures_path.exists() else None
    teams_current = pd.DataFrame(bootstrap["teams"]) if "teams" in bootstrap else None
    return fixtures_current, teams_current


# Attaches `horizon_points`, summing the routed predictor's output across horizon_gws gameweeks.
def predict_horizon_points(players: pd.DataFrame, bootstrap: dict, horizon_gws: int) -> tuple[pd.DataFrame, bool]:
    with SETTINGS_PATH.open(encoding="utf-8") as f:
        threshold = int(yaml.safe_load(f)["model"]["cold_start_gw_threshold"])
    completed = completed_gameweeks_this_season(bootstrap)

    if should_use_cold_start(completed, threshold) or not REGISTRY_PATH.exists():
        fixtures_current, teams_current = load_fixtures_and_teams_current(bootstrap)
        predictions, used_cold_start = predict_points(players, bootstrap, fixtures_current=fixtures_current, teams_current=teams_current)
        predictions = predictions.copy()
        predictions["horizon_points"] = predictions["predicted_points"] * horizon_gws
        return predictions, used_cold_start

    next_gw = infer_current_gameweek(bootstrap)
    if next_gw is None:
        raise ValueError("Cannot determine the current/next gameweek from bootstrap-static (is the season over?).")

    fixtures_current, teams_current = load_fixtures_and_teams_current(bootstrap)
    if fixtures_current is None or teams_current is None:
        raise ValueError("The trained-model path needs fixtures_current.parquet and bootstrap['teams'] - run the refresh first.")
    player_histories = fetch_player_histories(players)

    horizon_frames = []
    for gw in range(next_gw, next_gw + horizon_gws):
        gw_predictions, _used_cold_start = predict_points(
            players, bootstrap, player_histories=player_histories, fixtures_current=fixtures_current,
            teams_current=teams_current, target_gameweek=gw,
        )
        gw_predictions = gw_predictions.copy()
        gw_predictions.loc[gw_predictions["fixture_count"] == 0, "predicted_points"] = 0.0
        horizon_frames.append(gw_predictions[["player_id", "predicted_points"]])

    horizon_sum = pd.concat(horizon_frames, ignore_index=True).groupby("player_id", as_index=False)["predicted_points"].sum()
    horizon_sum = horizon_sum.rename(columns={"predicted_points": "horizon_points"})

    predictions = players.merge(horizon_sum, on="player_id", how="left")
    predictions["horizon_points"] = predictions["horizon_points"].fillna(0.0)
    predictions["predicted_points"] = predictions["horizon_points"] / horizon_gws
    return predictions, False
