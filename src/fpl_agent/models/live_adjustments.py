# Adjusts model predictions for current-moment context the trained model can't learn from history alone: injury/fitness doubt and set-piece duties. Both are rule-based, not trained, since neither is available in the historical per-gameweek data - they can't be backtest-validated the way the rest of the pipeline is, only reasoned about.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_agent.overrides.manager import apply_overrides
from fpl_agent.storage.db import DEFAULT_DB_PATH, get_connection

PREDICTION_COLUMNS = ("predicted_points", "horizon_points")

# order==1 is the primary taker, order==2 the backup; anyone else (or unlisted) gets no boost.
SET_PIECE_BOOST = {
    "penalties_order": {1: 0.15, 2: 0.03},
    "corners_and_indirect_freekicks_order": {1: 0.05, 2: 0.01},
    "direct_freekicks_order": {1: 0.05, 2: 0.01},
}


# Loads player_status with active manual overrides layered on top - the same effective status Player Explorer shows.
def load_effective_player_status(current_gw: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    try:
        player_status = pd.read_sql_query("SELECT * FROM player_status", conn)
    finally:
        conn.close()
    return apply_overrides(player_status, current_gw=current_gw, db_path=db_path)


# Scales predictions down by each player's chance of playing next round - a 100% or unset chance leaves points unchanged.
def apply_availability_scaling(predictions: pd.DataFrame, player_status: pd.DataFrame) -> pd.DataFrame:
    chance = player_status.set_index("player_id")["chance_of_playing_next_round"]
    factor = (predictions["player_id"].map(chance).fillna(100) / 100).clip(0, 1)
    result = predictions.copy()
    for column in PREDICTION_COLUMNS:
        if column in result.columns:
            result[column] = result[column] * factor
    return result


# Boosts predictions for confirmed penalty/corner/free-kick takers, since dead-ball duties are a real scoring edge the trained model has no historical signal for.
def apply_set_piece_boost(predictions: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    order_columns = [c for c in SET_PIECE_BOOST if c in players.columns]
    if not order_columns:
        return predictions
    orders = players.set_index("player_id")[order_columns]

    multiplier = pd.Series(1.0, index=predictions.index)
    for field, boosts in SET_PIECE_BOOST.items():
        if field not in order_columns:
            continue
        player_order = predictions["player_id"].map(orders[field])
        for rank, boost in boosts.items():
            multiplier = multiplier + (player_order == rank).fillna(False) * boost

    result = predictions.copy()
    for column in PREDICTION_COLUMNS:
        if column in result.columns:
            result[column] = result[column] * multiplier
    return result


# Applies both live-context adjustments in sequence - the single entry point predict.py calls after computing raw model predictions.
def apply_live_adjustments(predictions: pd.DataFrame, players: pd.DataFrame, current_gw: int | None = None) -> pd.DataFrame:
    player_status = load_effective_player_status(current_gw)
    adjusted = apply_availability_scaling(predictions, player_status)
    return apply_set_piece_boost(adjusted, players)


# Applies one visitor's personal availability corrections on top of the shared predictions - independent of the shared overrides table, never persisted, never visible to any other visitor. A visitor's correction always takes effect even if the shared data hasn't caught up yet, since it's applied as its own scaling pass rather than merged into the shared table.
def apply_session_overrides(predictions: pd.DataFrame, session_overrides: dict[int, dict]) -> pd.DataFrame:
    if not session_overrides:
        return predictions
    rows = [{"player_id": player_id, **fields} for player_id, fields in session_overrides.items()]
    session_status = pd.DataFrame(rows)
    if "chance_of_playing_next_round" in session_status.columns:
        session_status["chance_of_playing_next_round"] = pd.to_numeric(session_status["chance_of_playing_next_round"], errors="coerce")
    return apply_availability_scaling(predictions, session_status)
