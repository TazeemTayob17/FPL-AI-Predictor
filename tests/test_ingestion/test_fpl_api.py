# Checks get_element_summary's optional freshness-aware cache reuse, which avoids ~600 live calls on every manual run.

import pytest

from fpl_agent.ingestion import cache, fpl_api


@pytest.fixture(autouse=True)
def _isolate_raw_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "RAW_DIR", tmp_path)


# A fresh cached pull must be returned without any network call.
def test_get_element_summary_reuses_a_fresh_cached_pull(monkeypatch):
    cache.save_json("element_summary/1", {"history": ["cached"]})
    monkeypatch.setattr(fpl_api, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call the live API")))

    assert fpl_api.get_element_summary(1, max_age_hours=24) == {"history": ["cached"]}


# With no max_age_hours given, the live API must always be called, matching the prior uncached behavior.
def test_get_element_summary_without_max_age_hours_always_fetches_live(monkeypatch):
    cache.save_json("element_summary/1", {"history": ["cached"]})
    monkeypatch.setattr(fpl_api, "_get", lambda url, params=None: {"history": ["live"]})

    assert fpl_api.get_element_summary(1) == {"history": ["live"]}
