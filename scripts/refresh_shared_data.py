# CI entrypoint: refreshes the shared prediction-pool cache that the deployed multi-visitor app reads from - run on a schedule via GitHub Actions (see .github/workflows/refresh_shared_data.yml), not manually. For the local single-user refresh + report, use scripts/update_now.py instead.

from __future__ import annotations

from fpl_agent.pipeline.recommendation_cache import refresh_and_cache_shared_predictions
from fpl_agent.storage.db import init_db

if __name__ == "__main__":
    init_db()  # a CI runner starts with no database at all every run - this is what app.py also does on startup
    shared = refresh_and_cache_shared_predictions()
    print(f"Shared predictions refreshed: used_cold_start={shared['used_cold_start']}, {len(shared['predictions'])} players")
