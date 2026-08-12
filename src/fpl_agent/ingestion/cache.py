"""Saves every raw API pull as its own timestamped file so nothing is ever overwritten."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def save_json(subdir: str, payload: object) -> Path:
    """Writes payload as a new timestamped JSON file under data/raw/<subdir>/ and returns its path."""
    target_dir = RAW_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
    path = target_dir / f"{timestamp}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
