# Session-scoped state for a multi-visitor deployment: each visitor's FPL team ID, strategy preferences, and personal overrides live in their own browser session (st.session_state), never written to shared files - so concurrent visitors never see or affect each other's data.

from __future__ import annotations

import streamlit as st

from fpl_agent.utils.settings import load_differential_aggressiveness, load_mini_league_ids

TEAM_ID_KEY = "session_team_id"
DIFFERENTIAL_AGGRESSIVENESS_KEY = "session_differential_aggressiveness"
MINI_LEAGUE_IDS_KEY = "session_mini_league_ids"
OVERRIDES_KEY = "session_overrides"


# This visitor's FPL team ID - read from the URL (?team_id=...) first so a link is bookmarkable/shareable, else from this browser session, else None if neither is set yet.
def get_session_team_id() -> int | None:
    query_value = st.query_params.get("team_id")
    if query_value:
        try:
            team_id = int(query_value)
        except ValueError:
            team_id = None
        if team_id is not None:
            st.session_state[TEAM_ID_KEY] = team_id
            return team_id
    return st.session_state.get(TEAM_ID_KEY)


# Stores this visitor's FPL team ID in both the session and the URL, so refreshing or sharing the link keeps it.
def set_session_team_id(team_id: int) -> None:
    st.session_state[TEAM_ID_KEY] = team_id
    st.query_params["team_id"] = str(team_id)


# Renders a prompt for a team ID if this visitor hasn't set one yet; returns the team ID once known (None while still prompting).
def ensure_session_team_id() -> int | None:
    team_id = get_session_team_id()
    if team_id is not None:
        return team_id

    st.info("Enter your FPL team ID to get recommendations for your own squad (the number after /entry/ in your team's URL).")
    entered = st.number_input("Your FPL team ID", min_value=0, step=1, value=0, key="team_id_prompt")
    if entered and st.button("Use this team"):
        set_session_team_id(int(entered))
        st.rerun()
    return None


# This visitor's differential-aggressiveness preference - session-scoped, defaulting to the shared settings.yaml value the first time it's read this session.
def get_session_differential_aggressiveness() -> str:
    if DIFFERENTIAL_AGGRESSIVENESS_KEY not in st.session_state:
        st.session_state[DIFFERENTIAL_AGGRESSIVENESS_KEY] = load_differential_aggressiveness()
    return st.session_state[DIFFERENTIAL_AGGRESSIVENESS_KEY]


def set_session_differential_aggressiveness(value: str) -> None:
    st.session_state[DIFFERENTIAL_AGGRESSIVENESS_KEY] = value


# This visitor's mini-league IDs - session-scoped, defaulting to the shared settings.yaml value the first time it's read this session.
def get_session_mini_league_ids() -> list[int]:
    if MINI_LEAGUE_IDS_KEY not in st.session_state:
        st.session_state[MINI_LEAGUE_IDS_KEY] = load_mini_league_ids()
    return st.session_state[MINI_LEAGUE_IDS_KEY]


def set_session_mini_league_ids(league_ids: list[int]) -> None:
    st.session_state[MINI_LEAGUE_IDS_KEY] = league_ids


# This visitor's personal overrides (do-not-sell, injury-doubt corrections) - {player_id: {field: value}}, never persisted to the shared overrides table and never visible to any other visitor.
def get_session_overrides() -> dict[int, dict]:
    return st.session_state.setdefault(OVERRIDES_KEY, {})


def set_session_override(player_id: int, field: str, value: str) -> None:
    overrides = get_session_overrides()
    overrides.setdefault(player_id, {})[field] = value


def clear_session_override(player_id: int, field: str) -> None:
    overrides = get_session_overrides()
    if player_id not in overrides:
        return
    overrides[player_id].pop(field, None)
    if not overrides[player_id]:
        del overrides[player_id]
