"""Checks manual override CRUD, soft-delete audit trail, and precedence over automated player_status."""

import pandas as pd

from fpl_agent.overrides.manager import (
    active_overrides_for_gameweek,
    apply_overrides,
    create_override,
    deactivate_override,
    list_overrides,
)
from fpl_agent.storage.db import init_db

PLAYER_STATUS = pd.DataFrame(
    [
        {"player_id": 1, "status": "a", "chance_of_playing_next_round": 100},
        {"player_id": 2, "status": "a", "chance_of_playing_next_round": 100},
    ]
)


def test_create_and_list_override(tmp_path):
    """A newly created override must appear in the active list with the fields it was given."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    create_override(1, "status", "d", reason="knock in training", db_path=db_path)
    active = list_overrides(db_path=db_path)
    assert len(active) == 1
    assert active.iloc[0]["field"] == "status"
    assert active.iloc[0]["reason"] == "knock in training"


def test_deactivate_preserves_audit_trail(tmp_path):
    """Deactivating an override must remove it from the active list without deleting the row (audit trail)."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    override_id = create_override(1, "status", "d", db_path=db_path)
    deactivate_override(override_id, db_path=db_path)
    assert list_overrides(active_only=True, db_path=db_path).empty
    assert len(list_overrides(active_only=False, db_path=db_path)) == 1


def test_apply_overrides_badges_and_replaces_the_value(tmp_path):
    """A player with an active override must show the override's value and be flagged is_manual_override."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    create_override(1, "status", "i", db_path=db_path)
    result = apply_overrides(PLAYER_STATUS, db_path=db_path)
    row = result[result["player_id"] == 1].iloc[0]
    assert row["status"] == "i"
    assert bool(row["is_manual_override"]) is True


def test_apply_overrides_leaves_other_players_untouched(tmp_path):
    """A player with no override must keep their automated status and not be flagged manual."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    create_override(1, "status", "i", db_path=db_path)
    result = apply_overrides(PLAYER_STATUS, db_path=db_path)
    row = result[result["player_id"] == 2].iloc[0]
    assert row["status"] == "a"
    assert bool(row["is_manual_override"]) is False


def test_numeric_override_field_is_cast_to_int(tmp_path):
    """chance_of_playing_next_round overrides must come back as an int, not the raw stored string."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    create_override(1, "chance_of_playing_next_round", "25", db_path=db_path)
    result = apply_overrides(PLAYER_STATUS, db_path=db_path)
    row = result[result["player_id"] == 1].iloc[0]
    assert row["chance_of_playing_next_round"] == 25
    assert int(row["chance_of_playing_next_round"]) == 25


def test_gw_scoped_override_only_applies_to_its_gameweek(tmp_path):
    """An override scoped to GW5 must not apply when checking GW6, but must apply when checking GW5."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    create_override(1, "status", "i", gw_scope=5, db_path=db_path)
    assert active_overrides_for_gameweek(6, db_path=db_path).empty
    assert not active_overrides_for_gameweek(5, db_path=db_path).empty


def test_unscoped_override_applies_to_every_gameweek(tmp_path):
    """An override with no gw_scope (None) must apply regardless of which gameweek is being checked."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    create_override(1, "status", "i", db_path=db_path)
    assert not active_overrides_for_gameweek(1, db_path=db_path).empty
    assert not active_overrides_for_gameweek(38, db_path=db_path).empty
