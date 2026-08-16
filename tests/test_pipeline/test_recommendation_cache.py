# Checks the recommendation cache round-trips correctly and refresh_and_cache_recommendation wires run_live to it.

from datetime import datetime, timedelta, timezone

from fpl_agent.pipeline import recommendation_cache


# No cache file yet must return (None, None), not raise.
def test_load_recommendation_returns_none_when_nothing_cached(tmp_path):
    result, cached_at = recommendation_cache.load_recommendation(cache_path=tmp_path / "missing.pkl")
    assert result is None
    assert cached_at is None


# A saved result must load back identical, with a cached_at timestamp close to now.
def test_save_and_load_recommendation_round_trips(tmp_path):
    cache_path = tmp_path / "rec.pkl"
    saved = {"mode": "live", "gameweek": 5}

    recommendation_cache.save_recommendation(saved, cache_path=cache_path)
    result, cached_at = recommendation_cache.load_recommendation(cache_path=cache_path)

    assert result == saved
    assert datetime.now(timezone.utc) - cached_at < timedelta(seconds=5)


# refresh_and_cache_recommendation must call run_live and persist exactly what it returned.
def test_refresh_and_cache_recommendation_saves_run_lives_result(monkeypatch, tmp_path):
    cache_path = tmp_path / "rec.pkl"
    fake_result = {"mode": "initial_squad"}
    monkeypatch.setattr(recommendation_cache, "run_live", lambda team_id=None: fake_result)
    monkeypatch.setattr(recommendation_cache, "CACHE_PATH", cache_path)

    result = recommendation_cache.refresh_and_cache_recommendation()

    assert result == fake_result
    loaded, _cached_at = recommendation_cache.load_recommendation(cache_path=cache_path)
    assert loaded == fake_result
