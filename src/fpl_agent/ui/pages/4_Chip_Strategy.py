# Streamlit page: mini-league risk posture, wildcard signal, and rule-aware chip suggestions - never auto-played.

import pandas as pd
import streamlit as st

from fpl_agent.ui.components.theme import inject_fpl_css
from fpl_agent.ui.components.visitor import render_visitor_recommendation

inject_fpl_css()
st.title("Chip Strategy")

result = render_visitor_recommendation()

if result is None:
    st.info("No recommendation cached yet - go to the home page and click Refresh now.")
elif result["mode"] == "initial_squad":
    st.info("Chip suggestions start once your live squad is synced (before the first gameweek deadline).")
else:
    plan = result["season_plan"]
    chip_report = result["chip_suggestions"]

    st.subheader("Mini-league posture")
    st.write(f"**{plan.risk_posture['posture'].title()}** - {plan.risk_posture['reasoning']}")

    if plan.wildcard_signal.get("suggest_wildcard"):
        st.warning(f"Wildcard signal: forward fixtures notably worse than average (difficulty gap {plan.wildcard_signal['gap']:.2f}).")

    if chip_report.get("urgency_warning"):
        st.error(chip_report["urgency_warning"])

    st.subheader("Chip suggestions")
    suggestions = pd.DataFrame(chip_report["suggestions"])
    upcoming = suggestions[suggestions["chip"].notna()]
    if upcoming.empty:
        st.write("No chip window meets the suggestion threshold in the scanned range.")
    else:
        st.dataframe(upcoming[["GW", "chip_label", "reasoning"]], use_container_width=True, hide_index=True)

    st.caption(f"Available chips this half: {', '.join(sorted(chip_report['available_chips'])) or 'none left'}")
