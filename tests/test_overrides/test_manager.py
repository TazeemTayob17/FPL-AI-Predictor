# Checks apply_overrides layers the shared (global) overrides table and a visitor's session_overrides correctly, with the session layer winning without touching the shared table.

import pandas as pd
import pytest

from fpl_agent.overrides.manager import apply_overrides, create_override
from fpl_agent.storage.db import init_db

PLAYER_STATUS = pd.DataFrame(
    [
        {"player_id": 1, "status": "a", "chance_of_playing_next_round": None},
        {"player_id": 2, "status": "a", "chance_of_playing_next_round": None},
    ]
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "fpl.db"
    init_db(path)
    return path


# With no overrides of any kind, player_status must pass through unchanged and unbadged.
def test_apply_overrides_is_a_no_op_with_nothing_active(db_path):
    result = apply_overrides(PLAYER_STATUS, db_path=db_path)
    assert not result["is_manual_override"].any()
    assert result.loc[result["player_id"] == 1, "status"].iloc[0] == "a"


# A shared (global) override still applies exactly as before this change.
def test_apply_overrides_applies_the_shared_table(db_path):
    create_override(1, "status", "i", db_path=db_path)
    result = apply_overrides(PLAYER_STATUS, db_path=db_path)
    row = result[result["player_id"] == 1].iloc[0]
    assert row["status"] == "i"
    assert row["is_manual_override"]


# A visitor's session override must apply even when the shared table has nothing active for that player - it's independent, not merged into the shared table.
def test_apply_overrides_applies_session_overrides_independently_of_the_shared_table(db_path):
    result = apply_overrides(PLAYER_STATUS, db_path=db_path, session_overrides={2: {"status": "d"}})
    row = result[result["player_id"] == 2].iloc[0]
    assert row["status"] == "d"
    assert row["is_manual_override"]
    # untouched for the other player
    assert result.loc[result["player_id"] == 1, "is_manual_override"].iloc[0] == False  # noqa: E712


# When both a shared override and a session override target the same player/field, the session override (this visitor's own correction) must win.
def test_session_override_wins_over_the_shared_table_for_the_same_field(db_path):
    create_override(1, "status", "i", db_path=db_path)
    result = apply_overrides(PLAYER_STATUS, db_path=db_path, session_overrides={1: {"status": "a"}})
    assert result.loc[result["player_id"] == 1, "status"].iloc[0] == "a"


# Numeric override fields (chance_of_playing_next_round) must be coerced to int from a session override too, matching the shared-table behavior.
def test_session_override_coerces_numeric_fields(db_path):
    result = apply_overrides(PLAYER_STATUS, db_path=db_path, session_overrides={1: {"chance_of_playing_next_round": "75"}})
    value = result.loc[result["player_id"] == 1, "chance_of_playing_next_round"].iloc[0]
    assert value == 75
    assert isinstance(value, int)
