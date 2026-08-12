"""Naive point predictor: last completed season's total points per player, no ML yet (Phase 2 baseline)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from fpl_agent.ingestion.vaastav_sync import EXTERNAL_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def previous_season_label(season_label: str) -> str:
    """Converts a "YYYY-YY" season label (e.g. "2026-27") to the prior season's label ("2025-26")."""
    start_year = int(season_label[:4])
    prev_start = start_year - 1
    return f"{prev_start}-{str(prev_start + 1)[-2:]}"


def load_last_season_points(season_label: str | None = None, external_dir: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Loads the prior season's synced cleaned_players.csv, one row per player name with their season total."""
    if season_label is None:
        with SETTINGS_PATH.open(encoding="utf-8") as f:
            season_label = yaml.safe_load(f)["season"]["label"]
    csv_path = external_dir / previous_season_label(season_label) / "cleaned_players.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["player_name", "last_season_points"])

    frame = pd.read_csv(csv_path)
    frame["player_name"] = (frame["first_name"] + " " + frame["second_name"]).str.strip().str.lower()
    deduped = frame.groupby("player_name", as_index=False)["total_points"].max()
    return deduped.rename(columns={"total_points": "last_season_points"})


def predict_naive_points(players: pd.DataFrame, last_season: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attaches a `predicted_points` column: last season's total points matched by name, else 0.0."""
    if last_season is None:
        last_season = load_last_season_points()

    join_key = players["web_name_full"].str.strip().str.lower()
    merged = players.assign(_join_key=join_key).merge(
        last_season.rename(columns={"player_name": "_join_key"}), on="_join_key", how="left"
    )
    merged["predicted_points"] = merged["last_season_points"].fillna(0.0)
    return merged.drop(columns=["_join_key", "last_season_points"])
