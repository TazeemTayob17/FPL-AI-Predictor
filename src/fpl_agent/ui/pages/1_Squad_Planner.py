# Streamlit page: current 15-man squad, starting XI, and bench, with squad value and budget.

import pandas as pd
import streamlit as st

from fpl_agent.optimizer.captaincy import choose_captaincy
from fpl_agent.optimizer.constraints import load_rules
from fpl_agent.ui.components.theme import inject_fpl_css, render_pitch
from fpl_agent.ui.components.visitor import render_visitor_recommendation

inject_fpl_css()
st.title("Squad Planner")

result = render_visitor_recommendation()

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

    if result["mode"] == "live":
        captain, vice_captain = result["captain"], result["vice_captain"]
    else:
        captain, vice_captain = choose_captaincy(starting_xi)
    render_pitch(starting_xi, bench, captain["web_name"], vice_captain["web_name"])
