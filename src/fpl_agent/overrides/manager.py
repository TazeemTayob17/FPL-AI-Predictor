"""Manual override CRUD: user-set flags (injury doubt, do-not-sell) that always take precedence over automated data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_agent.storage.db import DEFAULT_DB_PATH, get_connection

NUMERIC_OVERRIDE_FIELDS = {"chance_of_playing_this_round", "chance_of_playing_next_round"}


def create_override(
    player_id: int, field: str, value: str, reason: str | None = None, gw_scope: int | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Creates a new active manual override and returns its id."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO overrides (player_id, field, value, reason, gw_scope) VALUES (?, ?, ?, ?, ?)",
            (player_id, field, value, reason, gw_scope),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def deactivate_override(override_id: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Soft-deletes an override (active=0) without erasing it, preserving the audit trail."""
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE overrides SET active = 0, updated_at = datetime('now') WHERE id = ?", (override_id,))
        conn.commit()
    finally:
        conn.close()


def list_overrides(active_only: bool = True, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Lists overrides, most recent first, optionally filtered to only currently-active ones."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM overrides" + (" WHERE active = 1" if active_only else "") + " ORDER BY created_at DESC"
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def active_overrides_for_gameweek(current_gw: int | None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Lists active overrides that apply to the given gameweek - a null gw_scope means it always applies."""
    overrides = list_overrides(active_only=True, db_path=db_path)
    if overrides.empty:
        return overrides
    applies = overrides["gw_scope"].isna() | (overrides["gw_scope"] == current_gw)
    return overrides[applies]


def apply_overrides(player_status: pd.DataFrame, current_gw: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Layers active manual overrides onto automated player_status, badging affected rows as manual."""
    result = player_status.copy()
    result["is_manual_override"] = False
    overrides = active_overrides_for_gameweek(current_gw, db_path=db_path)
    if overrides.empty:
        return result

    for row in overrides.itertuples():
        mask = result["player_id"] == row.player_id
        if not mask.any():
            continue
        value = int(row.value) if row.field in NUMERIC_OVERRIDE_FIELDS else row.value
        result.loc[mask, row.field] = value
        result.loc[mask, "is_manual_override"] = True
    return result
