"""Downloads historical season player, gameweek, fixture, and team stats from the vaastav/Fantasy-Premier-League GitHub repo."""

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
    frames = [frame for season in seasons if (frame := _download_csv(season, "cleaned_players.csv")) is not None]
    return _concat_and_save(frames, "all_seasons.parquet")


def sync_gw_data(seasons: list[str] = SEASONS) -> pd.DataFrame:
    """Downloads each season's per-gameweek player rows and merges them into one parquet file."""
    frames = [frame for season in seasons if (frame := _download_csv(season, "gws/merged_gw.csv")) is not None]
    return _concat_and_save(frames, "all_seasons_gw.parquet")


def sync_fixtures_data(seasons: list[str] = SEASONS) -> pd.DataFrame:
    """Downloads each season's fixture results and difficulty ratings and merges them into one parquet file."""
    frames = [frame for season in seasons if (frame := _download_csv(season, "fixtures.csv")) is not None]
    return _concat_and_save(frames, "all_seasons_fixtures.parquet")


def sync_teams_data(seasons: list[str] = SEASONS) -> pd.DataFrame:
    """Downloads each season's team id/name/strength table and merges them into one parquet file."""
    frames = [frame for season in seasons if (frame := _download_csv(season, "teams.csv")) is not None]
    return _concat_and_save(frames, "all_seasons_teams.parquet")


def _download_csv(season: str, relative_path: str) -> pd.DataFrame | None:
    """Fetches and caches one season's CSV at the given repo-relative path, tagging every row with its season."""
    response = requests.get(f"{RAW_BASE_URL}/{season}/{relative_path}", timeout=30)
    if response.status_code != 200:
        return None
    local_path = EXTERNAL_DIR / season / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(response.content)
    frame = pd.read_csv(local_path)
    frame["season"] = season
    return frame


def _concat_and_save(frames: list[pd.DataFrame], filename: str) -> pd.DataFrame:
    """Concatenates per-season frames (mismatched columns become NaN) and writes the result to parquet."""
    merged = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(PROCESSED_DIR / filename, index=False)
    return merged


if __name__ == "__main__":
    for label, sync_fn in [
        ("player totals", sync_seasons),
        ("gameweek rows", sync_gw_data),
        ("fixtures", sync_fixtures_data),
        ("teams", sync_teams_data),
    ]:
        result = sync_fn()
        print(f"Synced {len(result)} {label} rows across {result['season'].nunique()} seasons")
