"""Entrypoint that pulls the latest FPL data and stores it locally; run manually now, via Task Scheduler later."""

from __future__ import annotations

from fpl_agent.ingestion import fpl_api
from fpl_agent.storage import repository


def run_refresh() -> None:
    """Fetches bootstrap-static and fixtures, then persists both to parquet and SQLite."""
    bootstrap = fpl_api.get_bootstrap_static()
    fixtures = fpl_api.get_fixtures()
    repository.save_bootstrap(bootstrap)
    repository.save_fixtures(fixtures)


if __name__ == "__main__":
    run_refresh()
    print("Refresh complete.")
