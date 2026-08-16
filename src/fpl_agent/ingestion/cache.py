# Saves every raw API pull as its own timestamped file so nothing is ever overwritten.

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# Writes payload as a new timestamped JSON file under data/raw/<subdir>/ and returns its path.
def save_json(subdir: str, payload: object) -> Path:
    target_dir = RAW_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
    path = target_dir / f"{timestamp}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# Reads the most recently saved raw JSON file under data/raw/<subdir>/, or None if none exist.
def load_latest_json(subdir: str) -> dict | list | None:
    files = sorted((RAW_DIR / subdir).glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


# Reads the most recently saved raw JSON file under data/raw/<subdir>/ if it's younger than max_age_hours, else None.
def load_latest_json_if_fresh(subdir: str, max_age_hours: float) -> dict | list | None:
    files = sorted((RAW_DIR / subdir).glob("*.json"))
    if not files:
        return None
    latest = files[-1]
    if (time.time() - latest.stat().st_mtime) / 3600 > max_age_hours:
        return None
    return json.loads(latest.read_text(encoding="utf-8"))
