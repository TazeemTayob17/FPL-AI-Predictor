"""Normalizes raw FPL API payloads into the SQLite snapshot tables and processed parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_agent.storage.db import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PLAYER_COLUMNS = [
    "player_id", "web_name", "web_name_full", "team_name", "team_short", "position",
    "now_cost_million", "form", "total_points", "selected_by_percent",
    "status", "news", "chance_of_playing_this_round", "chance_of_playing_next_round",
]


def save_bootstrap(bootstrap: dict) -> pd.DataFrame:
    """Builds the current-players table from bootstrap-static and writes it to parquet + SQLite."""
    players = _build_players_frame(bootstrap)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    players.to_parquet(PROCESSED_DIR / "players_current.parquet", index=False)
    gameweek = _infer_current_gameweek(bootstrap)
    _write_snapshots(players, gameweek)
    _write_player_status(players)
    return players


def save_fixtures(fixtures: list) -> pd.DataFrame:
    """Writes the fixture list to parquet for the features layer to consume later."""
    frame = pd.DataFrame(fixtures)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PROCESSED_DIR / "fixtures_current.parquet", index=False)
    return frame


def _build_players_frame(bootstrap: dict) -> pd.DataFrame:
    """Joins raw player rows with team and position names into one readable table."""
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]].rename(
        columns={"id": "team", "name": "team_name", "short_name": "team_short"}
    )
    positions = pd.DataFrame(bootstrap["element_types"])[["id", "singular_name_short"]].rename(
        columns={"id": "element_type", "singular_name_short": "position"}
    )
    players = elements.merge(teams, on="team", how="left").merge(positions, on="element_type", how="left")
    players["now_cost_million"] = players["now_cost"] / 10
    players["form"] = pd.to_numeric(players["form"], errors="coerce")
    players["selected_by_percent"] = pd.to_numeric(players["selected_by_percent"], errors="coerce")
    players["web_name_full"] = players["first_name"] + " " + players["second_name"]
    players = players.rename(columns={"id": "player_id"})
    return players[PLAYER_COLUMNS]


def _infer_current_gameweek(bootstrap: dict) -> int | None:
    """Finds the current (or next, pre-season) gameweek number from bootstrap-static's events list."""
    events = bootstrap.get("events", [])
    for event in events:
        if event.get("is_current"):
            return event["id"]
    for event in events:
        if event.get("is_next"):
            return event["id"]
    return None


def _write_snapshots(players: pd.DataFrame, gameweek: int | None) -> None:
    """Inserts one snapshot row per player into SQLite, for tracking value/ownership trends over time."""
    conn = get_connection()
    try:
        rows = [
            (
                gameweek,
                int(row.player_id),
                int(row.now_cost_million * 10),
                _float_or_none(row.selected_by_percent),
                _float_or_none(row.form),
                int(row.total_points),
            )
            for row in players.itertuples()
        ]
        conn.executemany(
            """
            INSERT INTO snapshots (gameweek, player_id, now_cost, selected_by_percent, form, total_points)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _write_player_status(players: pd.DataFrame) -> None:
    """Upserts the automated player_status table from bootstrap-static's own injury/availability fields."""
    conn = get_connection()
    try:
        rows = [
            (
                int(row.player_id),
                row.status,
                row.news,
                _int_or_none(row.chance_of_playing_this_round),
                _int_or_none(row.chance_of_playing_next_round),
                "fpl_api",
            )
            for row in players.itertuples()
        ]
        conn.executemany(
            """
            INSERT INTO player_status
                (player_id, status, news, chance_of_playing_this_round, chance_of_playing_next_round, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                status=excluded.status,
                news=excluded.news,
                chance_of_playing_this_round=excluded.chance_of_playing_this_round,
                chance_of_playing_next_round=excluded.chance_of_playing_next_round,
                source=excluded.source,
                updated_at=datetime('now')
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _float_or_none(value: object) -> float | None:
    """Converts a pandas value to a plain float, or None if it's NaN."""
    return None if pd.isna(value) else float(value)


def _int_or_none(value: object) -> int | None:
    """Converts a pandas value to a plain int, or None if it's NaN."""
    return None if pd.isna(value) else int(value)
