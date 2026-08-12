"""Checks free-transfer accounting and live entry/history/picks normalization into team_snapshot."""

import json

from fpl_agent.optimizer.constraints import SquadRules
from fpl_agent.storage.db import get_connection, init_db
from fpl_agent.storage.repository import compute_free_transfers, save_team_snapshot

RULES = SquadRules(
    budget_million=100.0, squad_size=15,
    squad_composition={"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
    starting_xi_size=11, formation_min={"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
    max_players_per_club=3, captain_multiplier=2, triple_captain_multiplier=3,
    free_transfers_per_week=1, free_transfers_max_banked=5, points_hit_per_extra_transfer=-4,
)


def _history(current, chips=None):
    """Builds a minimal history payload matching the FPL API's entry/history/ shape."""
    return {"current": current, "past": [], "chips": chips or []}


def test_compute_free_transfers_before_gw2_is_the_base_rate():
    """GW1 has no free-transfer accounting - only the base rate (1) applies before GW2 exists."""
    history = _history([{"event": 1, "event_transfers": 0}])
    assert compute_free_transfers(history, RULES) == 1


def test_compute_free_transfers_accumulates_when_untouched():
    """Three gameweeks with zero transfers made should each bank one more, up to 4 by GW4."""
    history = _history(
        [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 0},
            {"event": 3, "event_transfers": 0},
            {"event": 4, "event_transfers": 0},
        ]
    )
    assert compute_free_transfers(history, RULES) == 4


def test_compute_free_transfers_taking_a_hit_does_not_go_negative():
    """Making 3 transfers on 1 available free transfer takes a hit but leaves the bank at 1, not -2."""
    history = _history([{"event": 1, "event_transfers": 0}, {"event": 2, "event_transfers": 3}])
    assert compute_free_transfers(history, RULES) == 1


def test_compute_free_transfers_caps_at_max_banked():
    """Six untouched gameweeks would reach 6 uncapped, but the config caps banking at 5."""
    history = _history([{"event": e, "event_transfers": 0} for e in range(1, 8)])
    assert compute_free_transfers(history, RULES) == 5


def test_compute_free_transfers_wildcard_week_is_not_consumed():
    """A wildcard gameweek's 15 transfers must not eat into the free-transfer count at all."""
    history = _history(
        [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 0},
            {"event": 3, "event_transfers": 15},
        ],
        chips=[{"name": "wildcard", "event": 3}],
    )
    assert compute_free_transfers(history, RULES) == 3


def test_save_team_snapshot_returns_none_when_picks_not_live_yet():
    """Pre-deadline, picks is None - the function must no-op, not raise or write a garbage row."""
    assert save_team_snapshot(entry={}, history=_history([]), picks=None, rules=RULES) is None


def test_save_team_snapshot_normalizes_and_persists_a_live_squad(tmp_path):
    """A realistic entry/history/picks payload should normalize into the expected snapshot shape and land in SQLite."""
    db_path = tmp_path / "test.db"
    init_db(db_path)

    history = _history(
        [{"event": 1, "event_transfers": 0}, {"event": 2, "bank": 15, "value": 1005, "event_transfers": 1}],
        chips=[{"name": "bboost", "event": 5}],
    )
    picks = {
        "entry_history": {"event": 2},
        "picks": [
            {"element": 101, "is_captain": True, "is_vice_captain": False, "multiplier": 2, "position": 1},
            {"element": 202, "is_captain": False, "is_vice_captain": True, "multiplier": 1, "position": 2},
        ],
    }

    result = save_team_snapshot(entry={"id": 4062058}, history=history, picks=picks, rules=RULES, db_path=db_path)

    assert result["gameweek"] == 2
    assert result["bank"] == 1.5
    assert result["squad_value"] == 100.5
    assert result["free_transfers"] == 1
    assert result["chips_used"] == [{"chip": "bboost", "gameweek": 5}]
    assert result["picks"][0] == {
        "player_id": 101, "is_captain": True, "is_vice_captain": False, "multiplier": 2, "position": 1,
    }

    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM team_snapshot").fetchone()
    finally:
        conn.close()
    assert row["gameweek"] == 2
    assert row["free_transfers"] == 1
    assert json.loads(row["picks_json"])[1]["player_id"] == 202
