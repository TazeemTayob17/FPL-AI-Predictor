# Streamlit entrypoint: landing page showing a quick summary, staleness indicator, and refresh button.

import streamlit as st

from fpl_agent.ui.components.staleness import render_staleness_and_refresh
from fpl_agent.ui.components.theme import inject_fpl_css
from fpl_agent.utils.env import get_team_id

st.set_page_config(page_title="FPL Agent", page_icon="⚽")
inject_fpl_css()
st.title("FPL Agent")

try:
    team_id = get_team_id()
except RuntimeError as exc:
    team_id = None
    st.warning(f"{exc} You can still browse, but Refresh now will fail until it's set.")

result = render_staleness_and_refresh(team_id)

if result is None:
    st.info("No recommendation cached yet - click Refresh now above to compute your first one.")
elif result["mode"] == "initial_squad":
    squad = result["squad"]
    st.subheader("Pre-season: recommended squad ready")
    st.write(f"{len(squad)} players, £{squad['now_cost_million'].sum():.1f}m - see Squad Planner for the full breakdown.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gameweek", result["gameweek"])
    col2.metric("Bank", f"£{result['bank']:.1f}m")
    col3.metric("Free transfers", result["free_transfers"])
    col4.metric("Transfers suggested", result["recommendation"]["num_transfers"])
    st.write(f"Captain: **{result['captain']['web_name']}** - see Captaincy for the full ranking.")
    st.write(f"Mini-league posture: **{result['season_plan'].risk_posture['posture']}** - see Chip Strategy for details.")

st.write("Use the sidebar to open a specific page: Squad Planner, Transfers, Captaincy, Chip Strategy, Player Explorer, Overrides, Settings.")
