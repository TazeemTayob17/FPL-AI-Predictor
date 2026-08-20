# Thin orchestration entrypoint for the decomposed model: builds the training table (reusing build_training_table exactly as train.py does), trains every position's component set, and saves each component's LightGBM booster to disk with a registry describing the file layout - the decomposed-model analog of train.py.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fpl_agent.models.component_models import train_all_component_models
from fpl_agent.models.train import PROJECT_ROOT, VALIDATION_SEASON, feature_columns, time_ordered_split

ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / "decomposed"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry_decomposed.json"

APPEARANCE_KEYS = ("any_minutes", "sixty_plus_minutes", "minutes_when_played")


def _save_booster(model, path: Path) -> str | None:
    if model is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path.name


# Saves every trained booster to its own file and writes a JSON registry describing the layout, mirroring train.py's registry.json pattern.
def save_component_models(
    models: dict, artifacts_dir: Path = ARTIFACTS_DIR, registry_path: Path = REGISTRY_PATH, columns: list[str] | None = None
) -> dict:
    registry: dict = {"features": columns or [], "positions": {}}
    for position, components in models.items():
        entry: dict = {"appearance": {}, "rates": {}, "defensive_contribution": None}
        for key in APPEARANCE_KEYS:
            entry["appearance"][key] = _save_booster(components["appearance"][key], artifacts_dir / f"{position}_{key}.txt")
        for output_column, model in components["rates"].items():
            entry["rates"][output_column] = _save_booster(model, artifacts_dir / f"{position}_{output_column}.txt")
        entry["defensive_contribution"] = _save_booster(
            components["defensive_contribution"], artifacts_dir / f"{position}_defensive_contribution.txt"
        )
        registry["positions"][position] = entry

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


if __name__ == "__main__":
    from fpl_agent.features.build_features import build_training_table
    from fpl_agent.storage.repository import PROCESSED_DIR

    gw_rows = pd.read_parquet(PROCESSED_DIR / "all_seasons_gw.parquet")
    fixtures = pd.read_parquet(PROCESSED_DIR / "all_seasons_fixtures.parquet")
    teams = pd.read_parquet(PROCESSED_DIR / "all_seasons_teams.parquet")
    training_table = build_training_table(gw_rows, fixtures, teams)

    columns = feature_columns(training_table)
    train_df, valid_df = time_ordered_split(training_table, VALIDATION_SEASON)
    models = train_all_component_models(train_df, valid_df, columns)
    save_component_models(models, columns=columns)
    print(f"Trained decomposed component models for {len(models)} positions -> {REGISTRY_PATH}")
