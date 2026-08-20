# Checks that per-visitor state (team ID, preferences, overrides) is genuinely session-scoped and never leaks between visitors via shared files.

import pytest
import streamlit as st

from fpl_agent.ui.components import session as session_module
from fpl_agent.ui.components.session import (
    clear_session_override,
    get_session_differential_aggressiveness,
    get_session_mini_league_ids,
    get_session_overrides,
    get_session_team_id,
    set_session_differential_aggressiveness,
    set_session_mini_league_ids,
    set_session_override,
    set_session_team_id,
)


# Clears Streamlit's (bare-mode) session state and query params before each test, so tests don't leak into each other.
@pytest.fixture(autouse=True)
def _clean_session_state():
    st.session_state.clear()
    st.query_params.clear()
    yield
    st.session_state.clear()
    st.query_params.clear()


# A team ID present in the URL must be picked up even before anything is in session state - makes a shared link work for a brand-new visitor.
def test_get_session_team_id_reads_from_query_params():
    st.query_params["team_id"] = "12345"
    assert get_session_team_id() == 12345
    assert st.session_state[session_module.TEAM_ID_KEY] == 12345


# With no query param, the session-stored value (e.g. set earlier via the Settings form) must still be returned.
def test_get_session_team_id_falls_back_to_session_state():
    set_session_team_id(999)
    st.query_params.clear()
    assert get_session_team_id() == 999


# No query param and no session value yet must return None, not raise or default to some other visitor's team.
def test_get_session_team_id_returns_none_when_unset():
    assert get_session_team_id() is None


# A malformed query param must not crash the page - falls back to whatever's in session state instead.
def test_get_session_team_id_ignores_a_non_numeric_query_param():
    st.session_state[session_module.TEAM_ID_KEY] = 42
    st.query_params["team_id"] = "not-a-number"
    assert get_session_team_id() == 42


# set_session_team_id must write to both session state and the URL, so a page reload and a shared link both keep working.
def test_set_session_team_id_writes_session_and_query_params():
    set_session_team_id(777)
    assert st.session_state[session_module.TEAM_ID_KEY] == 777
    assert st.query_params["team_id"] == "777"


# Preferences default to the shared settings.yaml value on first read this session, then stay session-local after that.
def test_session_preferences_default_from_settings_then_stay_session_local(monkeypatch):
    monkeypatch.setattr(session_module, "load_differential_aggressiveness", lambda: "balanced")
    monkeypatch.setattr(session_module, "load_mini_league_ids", lambda: [111])

    assert get_session_differential_aggressiveness() == "balanced"
    assert get_session_mini_league_ids() == [111]

    set_session_differential_aggressiveness("differential")
    set_session_mini_league_ids([222, 333])
    assert get_session_differential_aggressiveness() == "differential"
    assert get_session_mini_league_ids() == [222, 333]


# Overrides are a plain per-session dict - setting one for a player must not affect any other player, and clearing must remove exactly that field.
def test_session_overrides_set_and_clear():
    set_session_override(10, "chance_of_playing_next_round", "50")
    set_session_override(10, "status", "d")
    set_session_override(20, "do_not_sell", "true")

    overrides = get_session_overrides()
    assert overrides[10] == {"chance_of_playing_next_round": "50", "status": "d"}
    assert overrides[20] == {"do_not_sell": "true"}

    clear_session_override(10, "status")
    assert get_session_overrides()[10] == {"chance_of_playing_next_round": "50"}

    clear_session_override(20, "do_not_sell")
    assert 20 not in get_session_overrides()  # the player's entry is dropped entirely once its last field is cleared


# Two independent "sessions" (simulated by clearing state between them) must never see each other's team ID or overrides - the core multi-visitor isolation guarantee.
def test_two_sessions_never_see_each_others_state():
    set_session_team_id(1)
    set_session_override(5, "do_not_sell", "true")
    assert get_session_team_id() == 1
    assert 5 in get_session_overrides()

    st.session_state.clear()
    st.query_params.clear()

    assert get_session_team_id() is None
    assert get_session_overrides() == {}
