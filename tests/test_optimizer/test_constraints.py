"""Checks that the confirmed 2026/27 rules load correctly from config/settings.yaml."""

from fpl_agent.optimizer.constraints import load_rules


def test_load_rules_matches_confirmed_season_config():
    """Checks the loaded rules match the confirmed 2026/27 values documented in the implementation plan."""
    rules = load_rules()
    assert rules.budget_million == 100.0
    assert rules.squad_size == 15
    assert rules.squad_composition == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert rules.starting_xi_size == 11
    assert rules.formation_min == {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
    assert rules.max_players_per_club == 3
    assert rules.captain_multiplier == 2
    assert rules.triple_captain_multiplier == 3
