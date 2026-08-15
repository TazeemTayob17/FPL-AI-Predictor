"""Entrypoint that pulls the latest FPL data and stores it locally; run manually now, via Task Scheduler later."""

from __future__ import annotations

import pandas as pd

from fpl_agent.ingestion import fpl_api
from fpl_agent.storage import repository


def run_refresh() -> tuple[pd.DataFrame, dict]:
    """Fetches bootstrap-static and fixtures, persists both, and returns the current players frame + raw bootstrap payload."""
    bootstrap = fpl_api.get_bootstrap_static()
    fixtures = fpl_api.get_fixtures()
    players = repository.save_bootstrap(bootstrap)
    repository.save_fixtures(fixtures)
    return players, bootstrap


if __name__ == "__main__":
    run_refresh()
    print("Refresh complete.")
