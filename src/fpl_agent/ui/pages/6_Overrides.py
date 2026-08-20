# Streamlit page: this visitor's personal overrides (injury doubt, do-not-sell) - session-scoped, visible only to this visitor, never persisted once the session ends and never affecting anyone else's recommendations.

import pandas as pd
import streamlit as st

from fpl_agent.storage.repository import PROCESSED_DIR
from fpl_agent.ui.components.session import clear_session_override, get_session_overrides, set_session_override
from fpl_agent.ui.components.theme import inject_fpl_css

inject_fpl_css()
st.title("Overrides")
st.caption("Your personal corrections - visible only to you this session, never shared with other visitors and cleared once you close the tab.")

players_path = PROCESSED_DIR / "players_current.parquet"
if not players_path.exists():
    st.warning("No player data yet - the app owner needs to run a refresh first.")
else:
    players = pd.read_parquet(players_path)
    player_labels = players.sort_values("web_name")["web_name"] + " (" + players["team_short"] + ")"

    with st.form("add_override"):
        player_choice = st.selectbox("Player", player_labels)
        field = st.selectbox("Field", ["status", "chance_of_playing_next_round", "do_not_sell"])
        value = st.text_input("Value")
        submitted = st.form_submit_button("Add override")
        if submitted and value:
            player_id = int(players[player_labels == player_choice]["player_id"].iloc[0])
            set_session_override(player_id, field, value)
            st.rerun()

    st.subheader("Your active overrides")
    overrides = get_session_overrides()
    if not overrides:
        st.write("No active overrides.")
    else:
        rows = []
        for player_id, fields in overrides.items():
            name_match = players.loc[players["player_id"] == player_id, "web_name"]
            web_name = name_match.iloc[0] if not name_match.empty else str(player_id)
            for field, value in fields.items():
                rows.append({"player_id": player_id, "web_name": web_name, "field": field, "value": value})

        st.dataframe(pd.DataFrame(rows)[["web_name", "field", "value"]], use_container_width=True)

        labels = [f"{r['web_name']} - {r['field']}" for r in rows]
        remove_choice = st.selectbox("Remove override", [None, *labels])
        if remove_choice is not None and st.button("Remove selected override"):
            chosen = rows[labels.index(remove_choice)]
            clear_session_override(chosen["player_id"], chosen["field"])
            st.rerun()
