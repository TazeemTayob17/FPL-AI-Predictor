"""Checks that scrape/RSS enrichment only ever writes to player_status, and only overrides when genuinely more cautious."""

import pandas as pd

from fpl_agent.overrides.manager import list_overrides
from fpl_agent.storage.db import get_connection, init_db
from fpl_agent.storage.repository import enrich_player_status_from_rss, enrich_player_status_from_scrape


def _seed_baseline(db_path):
    """Writes one fpl_api baseline row so enrichment has something to update."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO player_status (player_id, status, chance_of_playing_next_round, source) VALUES (1, 'a', 100, 'fpl_api')"
        )
        conn.commit()
    finally:
        conn.close()


def _read_status(db_path):
    """Reads back the player_status row for player 1."""
    conn = get_connection(db_path)
    try:
        return dict(conn.execute("SELECT * FROM player_status WHERE player_id = 1").fetchone())
    finally:
        conn.close()


def test_scrape_enrichment_overrides_when_more_cautious(tmp_path):
    """A scraped chance of 25 must override the baseline's 100, since it's more cautious (lower)."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    _seed_baseline(db_path)
    scrape_rows = pd.DataFrame([{"player_id": 1, "status": "d", "chance_of_playing_next_round": 25, "news": "Hamstring knock"}])
    enrich_player_status_from_scrape(scrape_rows, db_path=db_path)
    row = _read_status(db_path)
    assert row["chance_of_playing_next_round"] == 25
    assert row["source"] == "scrape"


def test_scrape_enrichment_does_not_downgrade_a_more_cautious_baseline(tmp_path):
    """If the baseline is already more cautious than the scrape result, the scrape must not override it."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    conn.execute("INSERT INTO player_status (player_id, status, chance_of_playing_next_round, source) VALUES (1, 'i', 0, 'fpl_api')")
    conn.commit()
    conn.close()
    scrape_rows = pd.DataFrame([{"player_id": 1, "status": "d", "chance_of_playing_next_round": 75, "news": "Minor knock"}])
    enrich_player_status_from_scrape(scrape_rows, db_path=db_path)
    row = _read_status(db_path)
    assert row["chance_of_playing_next_round"] == 0
    assert row["source"] == "fpl_api"


def test_rss_enrichment_updates_news_without_touching_status(tmp_path):
    """RSS enrichment should only ever change the news text, never the structured status/chance fields."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    _seed_baseline(db_path)
    rss_rows = pd.DataFrame([{"player_id": 1, "news": "Reported to be a doubt for the weekend", "news_added": "2026-08-14"}])
    enrich_player_status_from_rss(rss_rows, db_path=db_path)
    row = _read_status(db_path)
    assert row["news"] == "Reported to be a doubt for the weekend"
    assert row["status"] == "a"
    assert row["chance_of_playing_next_round"] == 100


def test_enrichment_never_writes_to_the_overrides_table(tmp_path):
    """Automated enrichment must never touch the overrides table - that table is exclusively user-controlled."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    _seed_baseline(db_path)
    scrape_rows = pd.DataFrame([{"player_id": 1, "status": "i", "chance_of_playing_next_round": 0, "news": "Injured"}])
    rss_rows = pd.DataFrame([{"player_id": 1, "news": "Injured", "news_added": "2026-08-14"}])
    enrich_player_status_from_scrape(scrape_rows, db_path=db_path)
    enrich_player_status_from_rss(rss_rows, db_path=db_path)
    assert list_overrides(active_only=False, db_path=db_path).empty
