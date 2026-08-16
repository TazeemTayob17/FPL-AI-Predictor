# Loads/writes settings from .env - currently just the manager's public FPL team ID.

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"


# Reads FPL_TEAM_ID from .env; raises a clear error if the file or value is missing.
def get_team_id() -> int:
    load_dotenv(ENV_PATH)
    team_id = os.environ.get("FPL_TEAM_ID")
    if not team_id:
        raise RuntimeError(
            "FPL_TEAM_ID is not set - copy .env.example to .env and fill in your team ID "
            "(the number after /entry/ in your FPL team URL)."
        )
    return int(team_id)


# Writes FPL_TEAM_ID to .env via surgical text replacement, so any other lines/comments survive untouched.
def set_team_id(team_id: int, env_path: Path = ENV_PATH) -> None:
    if not env_path.exists():
        env_path.write_text(f"FPL_TEAM_ID={team_id}\n", encoding="utf-8")
        return
    text = env_path.read_text(encoding="utf-8")
    if re.search(r"^FPL_TEAM_ID=", text, re.MULTILINE):
        updated = re.sub(r"^FPL_TEAM_ID=.*$", f"FPL_TEAM_ID={team_id}", text, flags=re.MULTILINE)
    else:
        updated = text.rstrip("\n") + f"\nFPL_TEAM_ID={team_id}\n"
    env_path.write_text(updated, encoding="utf-8")
