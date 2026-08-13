"""Streamlit page: browse the current player pool with predicted points and which predictor is active."""

import pandas as pd
import streamlit as st
import yaml

from fpl_agent.ingestion.cache import load_latest_json
from fpl_agent.models.predict import SETTINGS_PATH, completed_gameweeks_this_season, predict_points
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

    st.dataframe(predictions.sort_values("predicted_points", ascending=False), use_container_width=True)
