# Read/write access to the settings.yaml fields the dashboard's Settings page can edit, without disturbing comments.

from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

DIFFERENTIAL_AGGRESSIVENESS_CHOICES = ("template", "balanced", "differential")


# Reads the current differential-aggressiveness toggle from settings.yaml.
def load_differential_aggressiveness(settings_path: Path = SETTINGS_PATH) -> str:
    with settings_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["strategy"]["differential_aggressiveness"]


# Writes a new differential-aggressiveness value via surgical text replacement, preserving every comment.
def save_differential_aggressiveness(value: str, settings_path: Path = SETTINGS_PATH) -> None:
    text = settings_path.read_text(encoding="utf-8")
    updated = re.sub(r'(differential_aggressiveness:\s*)"[^"]*"', rf'\1"{value}"', text)
    settings_path.write_text(updated, encoding="utf-8")


# Reads the configured mini-league IDs from settings.yaml.
def load_mini_league_ids(settings_path: Path = SETTINGS_PATH) -> list[int]:
    with settings_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["team"].get("mini_league_ids") or []


# Writes a new list of mini-league IDs via surgical text replacement, preserving every comment.
def save_mini_league_ids(league_ids: list[int], settings_path: Path = SETTINGS_PATH) -> None:
    text = settings_path.read_text(encoding="utf-8")
    new_value = "[" + ", ".join(str(i) for i in league_ids) + "]"
    updated = re.sub(r"(mini_league_ids:\s*)\[[^\]]*\]", rf"\1{new_value}", text)
    settings_path.write_text(updated, encoding="utf-8")
