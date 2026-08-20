# Renders the "your squad, live" flow for a public multi-visitor deployment: shows how fresh the shared prediction data is, prompts for a team ID if this visitor hasn't set one yet, then builds their own personal recommendation against the shared data - the multi-visitor analog of staleness.py's render_staleness_and_refresh. Every call re-syncs this visitor's squad live (cheap: sync_team's 3 lightweight API calls, no model inference), so a page reload always reflects their current real squad.

from __future__ import annotations

import streamlit as st

from fpl_agent.pipeline.recommendation_cache import load_shared_predictions
from fpl_agent.pipeline.weekly_pipeline import build_visitor_recommendation
from fpl_agent.ui.components.session import ensure_session_team_id, get_session_mini_league_ids, get_session_overrides
from fpl_agent.ui.components.staleness import format_time_ago


# Renders the shared-data staleness note + a manual refresh button, prompts for a team ID if needed, and returns this visitor's live recommendation (or None while still waiting on either).
def render_visitor_recommendation() -> dict | None:
    shared, cached_at = load_shared_predictions()
    if shared is None:
        st.warning("No shared prediction data yet - the app owner needs to run a refresh before any recommendations are available.")
        return None

    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(f"Shared prediction data last refreshed: {format_time_ago(cached_at)}")
    with col2:
        if st.button("Refresh my squad"):
            st.rerun()

    team_id = ensure_session_team_id()
    if team_id is None:
        return None

    try:
        return build_visitor_recommendation(
            team_id, shared, mini_league_ids=get_session_mini_league_ids(), session_overrides=get_session_overrides(),
        )
    except Exception as exc:
        st.error(f"Couldn't build your recommendation: {exc}")
        return None
