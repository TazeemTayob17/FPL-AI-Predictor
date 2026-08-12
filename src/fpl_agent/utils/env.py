"""Loads settings from .env - currently just the manager's public FPL team ID."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"


def get_team_id() -> int:
    """Reads FPL_TEAM_ID from .env; raises a clear error if the file or value is missing."""
    load_dotenv(ENV_PATH)
    team_id = os.environ.get("FPL_TEAM_ID")
    if not team_id:
        raise RuntimeError(
            "FPL_TEAM_ID is not set - copy .env.example to .env and fill in your team ID "
            "(the number after /entry/ in your FPL team URL)."
        )
    return int(team_id)
