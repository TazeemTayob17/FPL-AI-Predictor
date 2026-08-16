# CLI: fetches fresh data, computes the weekly recommendation, and caches it for the dashboard - run manually, daily.

from __future__ import annotations

from fpl_agent.pipeline.recommendation_cache import refresh_and_cache_recommendation
from fpl_agent.pipeline.weekly_pipeline import print_live_report
from fpl_agent.utils.env import get_team_id

if __name__ == "__main__":
    result = refresh_and_cache_recommendation(get_team_id())
    print_live_report(result)
