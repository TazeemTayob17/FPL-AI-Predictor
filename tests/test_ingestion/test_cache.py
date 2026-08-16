# Checks the raw-JSON cache's freshness-aware read path, used to avoid re-fetching unchanged data on every manual run.

import os
import time

import pytest

from fpl_agent.ingestion import cache


@pytest.fixture(autouse=True)
def _isolate_raw_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "RAW_DIR", tmp_path)


# No cached file at all must be treated as not fresh, not raise.
def test_load_latest_json_if_fresh_returns_none_when_nothing_cached():
    assert cache.load_latest_json_if_fresh("some/subdir", max_age_hours=24) is None


# A file just written is well within any reasonable freshness window and must be returned.
def test_load_latest_json_if_fresh_returns_recent_file():
    cache.save_json("some/subdir", {"a": 1})

    assert cache.load_latest_json_if_fresh("some/subdir", max_age_hours=24) == {"a": 1}


# A file older than the freshness window must be treated as stale, not returned.
def test_load_latest_json_if_fresh_rejects_a_stale_file():
    path = cache.save_json("some/subdir", {"a": 1})
    old_time = time.time() - 48 * 3600
    os.utime(path, (old_time, old_time))

    assert cache.load_latest_json_if_fresh("some/subdir", max_age_hours=24) is None
