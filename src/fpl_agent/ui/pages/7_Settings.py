# Streamlit page: this visitor's team ID, differential-aggressiveness toggle, and mini-league IDs - all session-scoped, none of it shared with or visible to any other visitor.

import streamlit as st

from fpl_agent.ui.components.session import (
    get_session_differential_aggressiveness,
    get_session_mini_league_ids,
    get_session_team_id,
    set_session_differential_aggressiveness,
    set_session_mini_league_ids,
    set_session_team_id,
)
from fpl_agent.ui.components.theme import inject_fpl_css
from fpl_agent.utils.settings import DIFFERENTIAL_AGGRESSIVENESS_CHOICES

inject_fpl_css()
st.title("Settings")

st.subheader("Team ID")
current_team_id = get_session_team_id()
if current_team_id is None:
    st.warning("No team ID set yet.")

with st.form("team_id_form"):
    new_team_id = st.number_input(
        "Your FPL team ID (the number after /entry/ in your team URL)",
        min_value=0, step=1, value=current_team_id or 0,
    )
    if st.form_submit_button("Save team ID") and new_team_id:
        set_session_team_id(int(new_team_id))
        st.success("Saved for this session.")
        st.rerun()

st.subheader("Differential aggressiveness")
st.caption("Manual override on top of the automatic mini-league-driven risk posture, not a replacement for it.")
current_aggressiveness = get_session_differential_aggressiveness()
new_aggressiveness = st.selectbox(
    "How aggressively to lean into differentials vs. template picks",
    DIFFERENTIAL_AGGRESSIVENESS_CHOICES,
    index=DIFFERENTIAL_AGGRESSIVENESS_CHOICES.index(current_aggressiveness),
)
if new_aggressiveness != current_aggressiveness and st.button("Save differential aggressiveness"):
    set_session_differential_aggressiveness(new_aggressiveness)
    st.success("Saved for this session.")
    st.rerun()

st.subheader("Mini-league IDs")
st.caption("The league ID is the number in the URL when viewing your mini-league standings on the FPL site.")
current_ids = get_session_mini_league_ids()
ids_text = st.text_input("Comma-separated league IDs", value=", ".join(str(i) for i in current_ids))
if st.button("Save mini-league IDs"):
    try:
        parsed_ids = [int(x.strip()) for x in ids_text.split(",") if x.strip()]
    except ValueError:
        st.error("League IDs must be numbers, comma-separated.")
    else:
        set_session_mini_league_ids(parsed_ids)
        st.success("Saved for this session.")
        st.rerun()
