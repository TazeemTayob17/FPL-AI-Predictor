# Streamlit page: team ID, differential-aggressiveness toggle, and mini-league IDs.

import streamlit as st

from fpl_agent.ui.components.theme import inject_fpl_css
from fpl_agent.utils.env import get_team_id, set_team_id
from fpl_agent.utils.settings import (
    DIFFERENTIAL_AGGRESSIVENESS_CHOICES,
    load_differential_aggressiveness,
    load_mini_league_ids,
    save_differential_aggressiveness,
    save_mini_league_ids,
)

inject_fpl_css()
st.title("Settings")

st.subheader("Team ID")
try:
    current_team_id = get_team_id()
except RuntimeError:
    current_team_id = None
    st.warning("FPL_TEAM_ID isn't set yet.")

with st.form("team_id_form"):
    new_team_id = st.number_input(
        "Your FPL team ID (the number after /entry/ in your team URL)",
        min_value=0, step=1, value=current_team_id or 0,
    )
    if st.form_submit_button("Save team ID") and new_team_id:
        set_team_id(int(new_team_id))
        st.success("Saved. Click Refresh now on any page to pick it up.")

st.subheader("Differential aggressiveness")
st.caption("Manual override on top of the automatic mini-league-driven risk posture, not a replacement for it.")
current_aggressiveness = load_differential_aggressiveness()
new_aggressiveness = st.selectbox(
    "How aggressively to lean into differentials vs. template picks",
    DIFFERENTIAL_AGGRESSIVENESS_CHOICES,
    index=DIFFERENTIAL_AGGRESSIVENESS_CHOICES.index(current_aggressiveness),
)
if new_aggressiveness != current_aggressiveness and st.button("Save differential aggressiveness"):
    save_differential_aggressiveness(new_aggressiveness)
    st.success("Saved.")
    st.rerun()

st.subheader("Mini-league IDs")
st.caption("The league ID is the number in the URL when viewing your mini-league standings on the FPL site.")
current_ids = load_mini_league_ids()
ids_text = st.text_input("Comma-separated league IDs", value=", ".join(str(i) for i in current_ids))
if st.button("Save mini-league IDs"):
    try:
        parsed_ids = [int(x.strip()) for x in ids_text.split(",") if x.strip()]
    except ValueError:
        st.error("League IDs must be numbers, comma-separated.")
    else:
        save_mini_league_ids(parsed_ids)
        st.success("Saved.")
        st.rerun()
