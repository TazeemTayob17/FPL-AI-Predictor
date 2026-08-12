"""Downloads historical season player stats from the vaastav/Fantasy-Premier-League GitHub repo for model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def sync_seasons(seasons: list[str] = SEASONS) -> pd.DataFrame:
    """Downloads each season's player summary CSV and merges them into one parquet file."""
    frames = [frame for season in seasons if (frame := _download_season(season)) is not None]
    merged = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(PROCESSED_DIR / "all_seasons.parquet", index=False)
    return merged


def _download_season(season: str) -> pd.DataFrame | None:
    """Fetches and caches one season's cleaned_players.csv, tagging every row with its season."""
    response = requests.get(f"{RAW_BASE_URL}/{season}/cleaned_players.csv", timeout=30)
    if response.status_code != 200:
        return None
    season_dir = EXTERNAL_DIR / season
    season_dir.mkdir(parents=True, exist_ok=True)
    csv_path = season_dir / "cleaned_players.csv"
    csv_path.write_bytes(response.content)
    frame = pd.read_csv(csv_path)
    frame["season"] = season
    return frame


if __name__ == "__main__":
    result = sync_seasons()
    print(f"Synced {len(result)} rows across {result['season'].nunique()} seasons")
