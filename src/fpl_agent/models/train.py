"""Trains 4 per-position LightGBM models on strictly time-ordered historical gameweek data."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from fpl_agent.features.rolling_stats import WINDOWS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"

POSITIONS = ("GKP", "DEF", "MID", "FWD")
TARGET_COLUMN = "total_points"
VALIDATION_SEASON = "2025-26"

LGB_PARAMS = {"objective": "regression", "metric": "mae", "verbosity": -1, "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 30}


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Lists the model's input features: rolling-window columns plus known-in-advance fixture/price context."""
    roll_cols = [c for c in table.columns if any(c.endswith(f"_roll{w}") for w in WINDOWS)]
    context_cols = [c for c in ("difficulty", "opponent_attack_strength", "opponent_defence_strength", "fixture_count", "value", "selected") if c in table.columns]
    return roll_cols + context_cols


def time_ordered_split(table: pd.DataFrame, validation_season: str = VALIDATION_SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits into train (every season before validation_season) and validation (validation_season itself)."""
    train = table[table["season"] < validation_season]
    valid = table[table["season"] == validation_season]
    return train, valid


def train_position_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, columns: list[str]) -> tuple[lgb.Booster, dict]:
    """Trains one LightGBM model for a single position, early-stopping on the held-out validation season."""
    train_set = lgb.Dataset(train_df[columns], label=train_df[TARGET_COLUMN])
    valid_set = lgb.Dataset(valid_df[columns], label=valid_df[TARGET_COLUMN], reference=train_set)
    model = lgb.train(
        LGB_PARAMS, train_set, num_boost_round=500, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    metadata = {
        "best_iteration": model.best_iteration,
        "validation_mae": model.best_score["valid_0"]["l1"],
        "n_train": len(train_df),
        "n_valid": len(valid_df),
    }
    return model, metadata


def train_all_positions(
    table: pd.DataFrame, validation_season: str = VALIDATION_SEASON, artifacts_dir: Path = ARTIFACTS_DIR,
    registry_path: Path = REGISTRY_PATH,
) -> dict:
    """Trains one LightGBM model per position, saves each to artifacts_dir, and writes the model registry."""
    columns = feature_columns(table)
    train_df, valid_df = time_ordered_split(table, validation_season)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    registry = {}
    for position in POSITIONS:
        pos_train = train_df[train_df["position"] == position].dropna(subset=[TARGET_COLUMN])
        pos_valid = valid_df[valid_df["position"] == position].dropna(subset=[TARGET_COLUMN])
        model, metadata = train_position_model(pos_train, pos_valid, columns)
        model_path = artifacts_dir / f"{position}.txt"
        model.save_model(str(model_path))
        registry[position] = {"model_file": model_path.name, "features": columns, **metadata}

    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


if __name__ == "__main__":
    from fpl_agent.features.build_features import build_training_table

    gw_rows = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "all_seasons_gw.parquet")
    fixtures = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "all_seasons_fixtures.parquet")
    teams = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "all_seasons_teams.parquet")
    training_table = build_training_table(gw_rows, fixtures, teams)

    result = train_all_positions(training_table)
    for position, info in result.items():
        print(f"{position}: MAE={info['validation_mae']:.3f}  n_train={info['n_train']}  n_valid={info['n_valid']}")
