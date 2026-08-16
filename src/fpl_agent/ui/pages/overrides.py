# Streamlit page: manual injury/do-not-sell override CRUD - always takes precedence over automated player_status.

import pandas as pd
import streamlit as st

from fpl_agent.overrides.manager import create_override, deactivate_override, list_overrides
from fpl_agent.storage.repository import PROCESSED_DIR

st.title("Overrides")
st.caption("Manual overrides always take precedence over automated status and are never silently cleared by a refresh.")

players_path = PROCESSED_DIR / "players_current.parquet"
if not players_path.exists():
    st.warning("No player data yet - run the refresh pipeline first.")
else:
    players = pd.read_parquet(players_path)
    player_labels = players.sort_values("web_name")["web_name"] + " (" + players["team_short"] + ")"

    with st.form("add_override"):
        player_choice = st.selectbox("Player", player_labels)
        field = st.selectbox("Field", ["status", "chance_of_playing_next_round", "do_not_sell"])
        value = st.text_input("Value")
        reason = st.text_input("Reason (optional)")
        gw_scope = st.number_input("Applies to this gameweek only (blank = until removed)", min_value=0, value=0, step=1)
        submitted = st.form_submit_button("Add override")
        if submitted and value:
            player_id = int(players[player_labels == player_choice]["player_id"].iloc[0])
            create_override(player_id, field, value, reason or None, gw_scope or None)
            st.rerun()

    st.subheader("Active overrides")
    active = list_overrides()
    if active.empty:
        st.write("No active overrides.")
    else:
        active_named = active.merge(players[["player_id", "web_name"]], on="player_id", how="left")
        st.dataframe(active_named[["id", "web_name", "field", "value", "reason", "gw_scope", "created_at"]], use_container_width=True)
        remove_id = st.selectbox("Remove override id", [None, *active["id"].tolist()])
        if remove_id is not None and st.button("Remove selected override"):
            deactivate_override(remove_id)
            st.rerun()
