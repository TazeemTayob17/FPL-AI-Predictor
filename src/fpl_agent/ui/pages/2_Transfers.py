# Streamlit page: this week's transfer recommendation, with full reasoning and hit economics.

import streamlit as st

from fpl_agent.ui.components.staleness import render_staleness_and_refresh
from fpl_agent.ui.components.theme import inject_fpl_css
from fpl_agent.utils.env import get_team_id

DISPLAY_COLUMNS = ["web_name", "team_short", "position", "now_cost_million", "horizon_points"]

st.set_page_config(layout="wide")
inject_fpl_css()
st.title("Transfers")

try:
    team_id = get_team_id()
except RuntimeError:
    team_id = None

result = render_staleness_and_refresh(team_id)

if result is None:
    st.info("No recommendation cached yet - go to the home page and click Refresh now.")
elif result["mode"] == "initial_squad":
    st.info("Your live squad isn't synced yet (before the first gameweek deadline) - transfer recommendations start once it is.")
else:
    recommendation = result["recommendation"]

    st.metric("Transfers recommended", recommendation["num_transfers"])
    st.metric("Hits taken", recommendation["hits"])

    st.subheader("Reasoning")
    for line in recommendation["reasoning"]:
        st.write(f"- {line}")

    if recommendation["num_transfers"] > 0:
        columns = [c for c in DISPLAY_COLUMNS if c in recommendation["dropped"].columns]
        st.subheader("Out")
        st.dataframe(recommendation["dropped"][columns], use_container_width=True)
        st.subheader("In")
        st.dataframe(recommendation["bought"][columns], use_container_width=True)
