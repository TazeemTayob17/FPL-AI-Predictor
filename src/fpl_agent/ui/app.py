# Streamlit entrypoint: defines the sidebar navigation (titles + icons) and dispatches to the selected page.

import streamlit as st

st.set_page_config(page_title="FPL Agent", page_icon="⚽", layout="wide")

pages = [
    st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_Squad_Planner.py", title="Squad Planner", icon="🧩"),
    st.Page("pages/2_Transfers.py", title="Transfers", icon="🔄"),
    st.Page("pages/3_Captaincy.py", title="Captaincy", icon="🎯"),
    st.Page("pages/4_Chip_Strategy.py", title="Chip Strategy", icon="🃏"),
    st.Page("pages/5_Player_Explorer.py", title="Player Explorer", icon="🔍"),
    st.Page("pages/6_Overrides.py", title="Overrides", icon="🛠️"),
    st.Page("pages/7_Settings.py", title="Settings", icon="⚙️"),
]

st.navigation(pages).run()
