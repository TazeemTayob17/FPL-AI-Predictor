# CLI: fetches fresh data once, computes the weekly recommendation, and caches both the single-user report and the shared prediction-pool cache the multi-visitor deployment reads from - run manually, daily. Commit and push data/processed/shared_predictions.pkl (and the other files listed in .gitignore's deployment-snapshot section) afterward to update a deployed Cloud instance.

from __future__ import annotations

from fpl_agent.pipeline.recommendation_cache import CACHE_PATH, refresh_and_cache_shared_predictions, save_recommendation
from fpl_agent.pipeline.weekly_pipeline import build_visitor_recommendation, print_live_report
from fpl_agent.utils.env import get_team_id

if __name__ == "__main__":
    shared = refresh_and_cache_shared_predictions()
    result = build_visitor_recommendation(get_team_id(), shared)
    save_recommendation(result, cache_path=CACHE_PATH)
    print_live_report(result)
