"""Routes point prediction to cold-start priors pre-season, or the trained LightGBM models once enough data exists."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

from fpl_agent.features.build_features import build_current_features
from fpl_agent.models.cold_start import predict_cold_start_points

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"


def completed_gameweeks_this_season(bootstrap: dict) -> int:
    """Counts how many of this season's gameweeks have finished, to decide whether cold-start or the trained model applies."""
    return sum(1 for event in bootstrap.get("events", []) if event.get("finished"))


def should_use_cold_start(completed_gameweeks: int, threshold: int) -> bool:
    """True until enough current-season gameweeks exist for the trained model's rolling features to be meaningful."""
    return completed_gameweeks < threshold


def load_position_models(registry_path: Path = REGISTRY_PATH) -> dict[str, tuple[lgb.Booster, list[str]]]:
    """Loads each position's trained model and its expected feature list from the registry."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    models = {}
    for position, info in registry.items():
        model_path = registry_path.parent / "artifacts" / info["model_file"]
        models[position] = (lgb.Booster(model_file=str(model_path)), info["features"])
    return models


def predict_with_trained_models(features: pd.DataFrame, models: dict[str, tuple[lgb.Booster, list[str]]]) -> pd.DataFrame:
    """Runs each position's model over its rows and attaches a `predicted_points` column."""
    features = features.copy()
    features["predicted_points"] = 0.0
    for position, (model, columns) in models.items():
        mask = features["position"] == position
        if mask.any():
            features.loc[mask, "predicted_points"] = model.predict(features.loc[mask, columns])
    return features


def predict_points(
    players: pd.DataFrame,
    bootstrap: dict,
    player_histories: dict[int, pd.DataFrame] | None = None,
    fixtures_current: pd.DataFrame | None = None,
    teams_current: pd.DataFrame | None = None,
    target_gameweek: int | None = None,
    registry_path: Path = REGISTRY_PATH,
) -> tuple[pd.DataFrame, bool]:
    """Returns (players_with_predicted_points, used_cold_start), routing per the configured cold-start GW threshold."""
    with SETTINGS_PATH.open(encoding="utf-8") as f:
        threshold = yaml.safe_load(f)["model"]["cold_start_gw_threshold"]
    completed = completed_gameweeks_this_season(bootstrap)

    if should_use_cold_start(completed, threshold) or not registry_path.exists():
        return predict_cold_start_points(players), True

    if player_histories is None or fixtures_current is None or teams_current is None or target_gameweek is None:
        raise ValueError("player_histories, fixtures_current, teams_current, and target_gameweek are required past cold-start.")

    models = load_position_models(registry_path)
    features = build_current_features(player_histories, players, fixtures_current, teams_current, target_gameweek)
    return predict_with_trained_models(features, models), False
