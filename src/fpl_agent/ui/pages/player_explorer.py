"""Streamlit page: browse the current player pool with predicted points, injury status, and manual overrides."""

import pandas as pd
import streamlit as st
import yaml

from fpl_agent.ingestion.cache import load_latest_json
from fpl_agent.models.predict import SETTINGS_PATH, completed_gameweeks_this_season, predict_points
from fpl_agent.overrides.manager import apply_overrides, create_override, deactivate_override, list_overrides
from fpl_agent.storage.db import get_connection
from fpl_agent.storage.repository import PROCESSED_DIR

st.title("Player Explorer")

players_path = PROCESSED_DIR / "players_current.parquet"
bootstrap = load_latest_json("bootstrap")

if not players_path.exists() or bootstrap is None:
    st.warning("No data yet - run the refresh pipeline first: python -m fpl_agent.pipeline.refresh_data")
else:
    players = pd.read_parquet(players_path)
    predictions, used_cold_start = predict_points(players, bootstrap)

    if used_cold_start:
        threshold = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))["model"]["cold_start_gw_threshold"]
        completed = completed_gameweeks_this_season(bootstrap)
        st.info(f"Using pre-season priors until GW {threshold} ({completed} gameweek(s) completed so far).")
    else:
        st.success("Using the trained prediction model.")

    conn = get_connection()
    try:
        player_status = pd.read_sql_query("SELECT * FROM player_status", conn)
    finally:
        conn.close()
    effective_status = apply_overrides(player_status)

    stale_status_cols = ["status", "news", "chance_of_playing_this_round", "chance_of_playing_next_round"]
    display = predictions.drop(columns=stale_status_cols).merge(
        effective_status[["player_id", "status", "chance_of_playing_next_round", "news", "is_manual_override"]],
        on="player_id", how="left",
    )
    display["status"] = display.apply(
        lambda r: f"{r['status']} (manual)" if r["is_manual_override"] else r["status"], axis=1
    )
    st.dataframe(display.sort_values("predicted_points", ascending=False), use_container_width=True)

    st.header("Manual overrides")
    st.caption("Manual overrides always take precedence over automated status and are never silently cleared by a refresh.")

    with st.form("add_override"):
        player_choice = st.selectbox("Player", players.sort_values("web_name")["web_name"] + " (" + players["team_short"] + ")")
        field = st.selectbox("Field", ["status", "chance_of_playing_next_round", "do_not_sell"])
        value = st.text_input("Value")
        reason = st.text_input("Reason (optional)")
        gw_scope = st.number_input("Applies to this gameweek only (blank = until removed)", min_value=0, value=0, step=1)
        submitted = st.form_submit_button("Add override")
        if submitted and value:
            player_id = int(players[players["web_name"] + " (" + players["team_short"] + ")" == player_choice]["player_id"].iloc[0])
            create_override(player_id, field, value, reason or None, gw_scope or None)
            st.rerun()

    active = list_overrides()
    if not active.empty:
        active_named = active.merge(players[["player_id", "web_name"]], on="player_id", how="left")
        st.dataframe(active_named[["id", "web_name", "field", "value", "reason", "gw_scope", "created_at"]], use_container_width=True)
        remove_id = st.selectbox("Remove override id", [None, *active["id"].tolist()])
        if remove_id is not None and st.button("Remove selected override"):
            deactivate_override(remove_id)
            st.rerun()
    else:
        st.write("No active overrides.")
