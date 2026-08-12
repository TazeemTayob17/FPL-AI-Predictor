"""Streamlit page: browse the current player pool pulled from local storage."""

import pandas as pd
import streamlit as st

from fpl_agent.storage.repository import PROCESSED_DIR

st.title("Player Explorer")

players_path = PROCESSED_DIR / "players_current.parquet"
if not players_path.exists():
    st.warning("No data yet - run the refresh pipeline first: python -m fpl_agent.pipeline.refresh_data")
else:
    players = pd.read_parquet(players_path)
    st.dataframe(players, use_container_width=True)
