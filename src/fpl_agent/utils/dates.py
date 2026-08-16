# Gameweek deadline lookups and chip-window (first-half/second-half) arithmetic.

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


# Returns the UTC deadline datetime for a given gameweek, or None if that gameweek isn't in bootstrap-static.
def gameweek_deadline(bootstrap: dict, gameweek: int) -> datetime | None:
    for event in bootstrap.get("events", []):
        if event.get("id") == gameweek:
            deadline = event.get("deadline_time")
            return datetime.fromisoformat(deadline.replace("Z", "+00:00")) if deadline else None
    return None


# Loads the chips section of settings.yaml (deadline gameweek/time, chips-per-gameweek limit).
def load_chip_config(settings_path: Path = SETTINGS_PATH) -> dict:
    with settings_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["chips"]


# Returns "first_half" or "second_half" depending on which side of the GW19-style cutoff a gameweek falls.
def current_chip_half(gameweek: int, first_half_deadline_gw: int) -> str:
    return "first_half" if gameweek <= first_half_deadline_gw else "second_half"


# Gameweeks left before first-half chips are forfeited; 0 or negative once the cutoff gameweek has passed.
def gws_remaining_in_first_half(current_gw: int, first_half_deadline_gw: int) -> int:
    return first_half_deadline_gw - current_gw
