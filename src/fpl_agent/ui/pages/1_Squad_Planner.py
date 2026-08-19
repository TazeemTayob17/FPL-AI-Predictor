# Streamlit page: current 15-man squad, starting XI, and bench, with squad value and budget.

import pandas as pd
import streamlit as st

from fpl_agent.optimizer.constraints import load_rules
from fpl_agent.ui.components.staleness import render_staleness_and_refresh
from fpl_agent.ui.components.theme import inject_fpl_css, render_pitch
from fpl_agent.utils.env import get_team_id

inject_fpl_css()
st.title("Squad Planner")

try:
    team_id = get_team_id()
except RuntimeError:
    team_id = None

result = render_staleness_and_refresh(team_id)

if result is None:
    st.info("No recommendation cached yet - go to the home page and click Refresh now.")
else:
    rules = load_rules()

    if result["mode"] == "initial_squad":
        squad, starting_xi, bench = result["squad"], result["starting_xi"], result["bench"]
    else:
        starting_xi, bench = result["starting_xi"], result["bench"]
        squad = pd.concat([starting_xi, bench], ignore_index=True)

    st.metric("Squad value", f"£{squad['now_cost_million'].sum():.1f}m / £{rules.budget_million:.1f}m")

    captain_name = result.get("captain", {}).get("web_name") if result["mode"] == "live" else None
    vice_captain_name = result.get("vice_captain", {}).get("web_name") if result["mode"] == "live" else None
    render_pitch(starting_xi, bench, captain_name, vice_captain_name)
