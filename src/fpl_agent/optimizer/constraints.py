"""Loads the confirmed season squad/formation rules from config/settings.yaml as typed constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

POSITIONS = ("GKP", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class SquadRules:
    """Typed view over the `rules` section of config/settings.yaml."""

    budget_million: float
    squad_size: int
    squad_composition: dict[str, int]
    starting_xi_size: int
    formation_min: dict[str, int]
    max_players_per_club: int
    captain_multiplier: int
    triple_captain_multiplier: int
    free_transfers_per_week: int = 1
    free_transfers_max_banked: int = 5
    points_hit_per_extra_transfer: int = -4


def load_rules(settings_path: Path = SETTINGS_PATH) -> SquadRules:
    """Reads config/settings.yaml and returns the current season's rules as a SquadRules object."""
    with settings_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    rules = raw["rules"]
    return SquadRules(
        budget_million=float(rules["budget_million"]),
        squad_size=int(rules["squad_size"]),
        squad_composition=dict(rules["squad_composition"]),
        starting_xi_size=int(rules["starting_xi_size"]),
        formation_min=dict(rules["formation_min"]),
        max_players_per_club=int(rules["max_players_per_club"]),
        captain_multiplier=int(rules["captain_multiplier"]),
        triple_captain_multiplier=int(rules["triple_captain_multiplier"]),
        free_transfers_per_week=int(rules["free_transfers_per_week"]),
        free_transfers_max_banked=int(rules["free_transfers_max_banked"]),
        points_hit_per_extra_transfer=int(rules["points_hit_per_extra_transfer"]),
    )
