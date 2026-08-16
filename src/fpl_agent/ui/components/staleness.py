# Shared "data last refreshed" indicator + manual refresh button, reused across every dashboard page.

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from fpl_agent.pipeline.recommendation_cache import load_recommendation, refresh_and_cache_recommendation


# Formats how long ago a UTC timestamp was, in a short human-readable form.
def format_time_ago(cached_at: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    seconds = (now - cached_at).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


# Renders the staleness indicator + refresh button, returns the cached recommendation dict (or None if never refreshed).
def render_staleness_and_refresh(team_id: int | None = None) -> dict | None:
    result, cached_at = load_recommendation()

    col1, col2 = st.columns([4, 1])
    with col1:
        if cached_at is None:
            st.caption("Data last refreshed: never - click Refresh now to compute your first recommendation.")
        else:
            st.caption(f"Data last refreshed: {format_time_ago(cached_at)}")
    with col2:
        if st.button("Refresh now"):
            try:
                with st.spinner("Refreshing..."):
                    refresh_and_cache_recommendation(team_id)
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")
                return result
            st.rerun()

    return result
