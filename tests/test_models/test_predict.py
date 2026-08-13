"""Checks the cold-start-vs-trained-model routing threshold logic, with no real model files or network calls."""

from fpl_agent.models.predict import completed_gameweeks_this_season, should_use_cold_start

BOOTSTRAP_PRE_SEASON = {"events": [{"id": 1, "finished": False}, {"id": 2, "finished": False}]}
BOOTSTRAP_MID_SEASON = {"events": [{"id": 1, "finished": True}, {"id": 2, "finished": True}, {"id": 3, "finished": False}]}


def test_completed_gameweeks_counts_only_finished_events():
    """Pre-season, with no events finished yet, the count must be 0."""
    assert completed_gameweeks_this_season(BOOTSTRAP_PRE_SEASON) == 0


def test_completed_gameweeks_reflects_finished_events():
    """Two finished events and one still upcoming must count as 2, not 3."""
    assert completed_gameweeks_this_season(BOOTSTRAP_MID_SEASON) == 2


def test_should_use_cold_start_below_threshold():
    """Fewer completed gameweeks than the configured threshold must route to cold-start."""
    assert should_use_cold_start(completed_gameweeks=2, threshold=4) is True


def test_should_use_cold_start_at_or_above_threshold():
    """Once enough gameweeks have completed, routing must switch away from cold-start."""
    assert should_use_cold_start(completed_gameweeks=4, threshold=4) is False
    assert should_use_cold_start(completed_gameweeks=5, threshold=4) is False
