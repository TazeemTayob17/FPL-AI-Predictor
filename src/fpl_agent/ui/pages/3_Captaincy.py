# Streamlit page: captain/vice-captain choice, with the full starting XI ranked by predicted points.

import streamlit as st

from fpl_agent.optimizer.captaincy import choose_captaincy
from fpl_agent.ui.components.staleness import render_staleness_and_refresh
from fpl_agent.ui.components.theme import inject_fpl_css
from fpl_agent.utils.env import get_team_id

DISPLAY_COLUMNS = ["web_name", "team_short", "position", "predicted_points"]

inject_fpl_css()
st.title("Captaincy")

try:
    team_id = get_team_id()
except RuntimeError:
    team_id = None

result = render_staleness_and_refresh(team_id)

if result is None:
    st.info("No recommendation cached yet - go to the home page and click Refresh now.")
else:
    starting_xi = result["starting_xi"]
    captain, vice_captain = (result["captain"], result["vice_captain"]) if result["mode"] == "live" else choose_captaincy(starting_xi)

    col1, col2 = st.columns(2)
    col1.metric("Captain", captain["web_name"], f"{captain['predicted_points']:.1f} pts")
    col2.metric("Vice-captain", vice_captain["web_name"], f"{vice_captain['predicted_points']:.1f} pts")

    st.caption("Vice-captain gets the 2x multiplier only if the captain doesn't play any minutes.")

    st.subheader("Starting XI, ranked by predicted points")
    columns = [c for c in DISPLAY_COLUMNS if c in starting_xi.columns]
    st.dataframe(starting_xi[columns].sort_values("predicted_points", ascending=False), use_container_width=True)
