"""Normalizes raw FPL API payloads into the SQLite snapshot tables and processed parquet files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fpl_agent.optimizer.constraints import SquadRules
from fpl_agent.storage.db import DEFAULT_DB_PATH, get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

UNLIMITED_TRANSFER_CHIPS = {"wildcard", "freehit"}

PLAYER_COLUMNS = [
    "player_id", "web_name", "web_name_full", "team_id", "team_name", "team_short", "position",
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
    players = players.rename(columns={"id": "player_id", "team": "team_id"})
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


def compute_free_transfers(history: dict, rules: SquadRules) -> int:
    """Replays each gameweek's transfers to compute free transfers banked, skipping GW1 and wildcard/free-hit weeks."""
    chip_gameweeks = {
        chip["event"] for chip in history.get("chips", []) if chip.get("name") in UNLIMITED_TRANSFER_CHIPS
    }
    free_transfers = rules.free_transfers_per_week
    for gw_entry in history.get("current", []):
        gameweek = gw_entry.get("event")
        if gameweek == 1:
            continue
        if gameweek in chip_gameweeks:
            free_transfers = min(rules.free_transfers_max_banked, free_transfers + rules.free_transfers_per_week)
            continue
        transfers_made = gw_entry.get("event_transfers", 0) or 0
        used = min(free_transfers, transfers_made)
        free_transfers = min(rules.free_transfers_max_banked, free_transfers - used + rules.free_transfers_per_week)
    return free_transfers


def save_team_snapshot(
    entry: dict, history: dict, picks: dict | None, rules: SquadRules, db_path: Path = DEFAULT_DB_PATH
) -> dict | None:
    """Normalizes live entry/history/picks data into the team_snapshot table; returns None if picks aren't live yet."""
    if picks is None:
        return None

    current_history = history.get("current", [])
    latest = current_history[-1] if current_history else {}
    bank = _tenths_to_million(latest.get("bank"))
    squad_value = _tenths_to_million(latest.get("value"))
    free_transfers = compute_free_transfers(history, rules)
    chips_used = [{"chip": chip["name"], "gameweek": chip["event"]} for chip in history.get("chips", [])]
    pick_rows = [
        {
            "player_id": pick["element"],
            "is_captain": pick["is_captain"],
            "is_vice_captain": pick["is_vice_captain"],
            "multiplier": pick["multiplier"],
            "position": pick["position"],
        }
        for pick in picks.get("picks", [])
    ]
    gameweek = picks.get("entry_history", {}).get("event")

    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO team_snapshot (gameweek, bank, squad_value, free_transfers, chips_used_json, picks_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (gameweek, bank, squad_value, free_transfers, json.dumps(chips_used), json.dumps(pick_rows)),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "gameweek": gameweek,
        "bank": bank,
        "squad_value": squad_value,
        "free_transfers": free_transfers,
        "chips_used": chips_used,
        "picks": pick_rows,
    }


def _tenths_to_million(value: object) -> float | None:
    """Converts a tenths-of-a-million integer (the API's bank/value unit) to millions, or None if unset."""
    return None if value is None else value / 10


def _float_or_none(value: object) -> float | None:
    """Converts a pandas value to a plain float, or None if it's NaN."""
    return None if pd.isna(value) else float(value)


def _int_or_none(value: object) -> int | None:
    """Converts a pandas value to a plain int, or None if it's NaN."""
    return None if pd.isna(value) else int(value)
