# Caches weekly_pipeline.run_live()'s result to disk so dashboard pages never call live APIs on open.

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

from fpl_agent.pipeline.weekly_pipeline import refresh_shared_predictions, run_live
from fpl_agent.storage.repository import PROCESSED_DIR

CACHE_PATH = PROCESSED_DIR / "latest_recommendation.pkl"
SHARED_PREDICTIONS_CACHE_PATH = PROCESSED_DIR / "shared_predictions.pkl"


# Runs the full live pipeline and caches the result with a timestamp, for the dashboard to read without a live call.
def refresh_and_cache_recommendation(team_id: int | None = None) -> dict:
    result = run_live(team_id)
    save_recommendation(result, cache_path=CACHE_PATH)
    return result


# Pickles a run_live() result dict to disk alongside a UTC timestamp.
def save_recommendation(result: dict, cache_path: Path = CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"result": result, "cached_at": datetime.now(timezone.utc)}
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)


# Loads the cached recommendation and its timestamp, or (None, None) if nothing has been cached yet.
def load_recommendation(cache_path: Path = CACHE_PATH) -> tuple[dict | None, datetime | None]:
    if not cache_path.exists():
        return None, None
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    return payload["result"], payload["cached_at"]


# Runs the shared, expensive half of a live run (refresh + full player-pool predictions) and caches it with a timestamp - the multi-visitor equivalent of refresh_and_cache_recommendation, meant to be run by the app owner (locally, then git-pushed for a Cloud deployment to pick up) rather than triggered per-visitor.
def refresh_and_cache_shared_predictions(cache_path: Path = SHARED_PREDICTIONS_CACHE_PATH) -> dict:
    shared = refresh_shared_predictions()
    save_recommendation(shared, cache_path=cache_path)
    return shared


# Loads the cached shared predictions and their timestamp, or (None, None) if the owner hasn't refreshed yet.
def load_shared_predictions(cache_path: Path = SHARED_PREDICTIONS_CACHE_PATH) -> tuple[dict | None, datetime | None]:
    if not cache_path.exists():
        return None, None
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    return payload["result"], payload["cached_at"]
