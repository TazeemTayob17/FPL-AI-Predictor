# Streamlit page: browse the full player pool with predicted points, ownership%, injury status, and template/differential tags.

import pandas as pd
import streamlit as st
import yaml

from fpl_agent.overrides.manager import apply_overrides
from fpl_agent.storage.db import get_connection
from fpl_agent.ui.components.staleness import render_staleness_and_refresh
from fpl_agent.ui.components.theme import inject_fpl_css
from fpl_agent.utils.env import get_team_id
from fpl_agent.utils.settings import SETTINGS_PATH

TEMPLATE_OWNERSHIP_THRESHOLD = 10.0  # % owned at/above which a player counts as "template" rather than "differential"

inject_fpl_css()
st.title("Player Explorer")

try:
    team_id = get_team_id()
except RuntimeError:
    team_id = None

result = render_staleness_and_refresh(team_id)

if result is None or "all_players" not in result:
    st.info("No recommendation cached yet - click Refresh now above to compute predictions for the full player pool.")
else:
    predictions = result["all_players"]

    settings = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))
    if result["used_cold_start"]:
        threshold = settings["model"]["cold_start_gw_threshold"]
        st.info(f"Using pre-season priors (switches to the trained model after GW {threshold}).")
    else:
        st.success("Using the trained prediction model.")
    st.caption(f"Differential aggressiveness: {settings['strategy']['differential_aggressiveness']} (change in Settings).")

    conn = get_connection()
    try:
        player_status = pd.read_sql_query("SELECT * FROM player_status", conn)
    finally:
        conn.close()
    effective_status = apply_overrides(player_status)

    stale_status_cols = [c for c in ("status", "news", "chance_of_playing_this_round", "chance_of_playing_next_round") if c in predictions.columns]
    display = predictions.drop(columns=stale_status_cols).merge(
        effective_status[["player_id", "status", "chance_of_playing_next_round", "news", "is_manual_override"]],
        on="player_id", how="left",
    )
    display["status"] = display.apply(
        lambda r: f"{r['status']} (manual)" if r["is_manual_override"] else r["status"], axis=1
    )
    display["template_or_differential"] = display["selected_by_percent"].apply(
        lambda pct: "Template" if pd.notna(pct) and pct >= TEMPLATE_OWNERSHIP_THRESHOLD else "Differential"
    )

    columns = [
        "web_name", "team_short", "position", "now_cost_million", "predicted_points",
        "selected_by_percent", "template_or_differential", "status", "chance_of_playing_next_round", "news",
    ]
    st.dataframe(display[columns].sort_values("predicted_points", ascending=False), use_container_width=True)
