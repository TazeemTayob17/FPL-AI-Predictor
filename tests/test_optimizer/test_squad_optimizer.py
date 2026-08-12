"""Checks the CP-SAT squad/starting-XI optimizer against a hand-calculable optimum and infeasible inputs."""

import pandas as pd
import pytest

from fpl_agent.optimizer.constraints import SquadRules
from fpl_agent.optimizer.squad_optimizer import InfeasibleSquadError, select_squad, select_starting_xi

# Small ruleset (6-man squad, 4-man XI) so the optimum below is verifiable by hand, distinct from the
# real 15-man/100m season rules exercised separately in test_constraints.py.
SMALL_RULES = SquadRules(
    budget_million=34.0,
    squad_size=6,
    squad_composition={"GKP": 1, "DEF": 2, "MID": 2, "FWD": 1},
    starting_xi_size=4,
    formation_min={"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},
    max_players_per_club=2,
    captain_multiplier=2,
    triple_captain_multiplier=3,
)


def _player(name, club, position, cost, points):
    """Builds one synthetic player row for the optimizer test pools."""
    return {
        "web_name": name, "team_name": club, "position": position,
        "now_cost_million": cost, "predicted_points": points,
    }


# 10 candidates, one clearly-better and one clearly-worse option per required slot, plus a budget
# tight enough (34) to force picking fwd_cheap (7) over the higher-scoring fwd_expensive (12, would
# push the cheapest valid combo from 34 to 39) - this is what proves the budget constraint actually bites.
OPTIMUM_POOL = pd.DataFrame(
    [
        _player("gk_good", "GKC", "GKP", 5, 40),
        _player("gk_bad", "GKC2", "GKP", 5, 20),
        _player("def1", "DC1", "DEF", 5, 45),
        _player("def2", "DC2", "DEF", 5, 42),
        _player("def3", "DC3", "DEF", 5, 20),
        _player("mid1", "MC1", "MID", 6, 55),
        _player("mid2", "MC2", "MID", 6, 50),
        _player("mid3", "MC3", "MID", 6, 10),
        _player("fwd_expensive", "FC1", "FWD", 12, 90),
        _player("fwd_cheap", "FC2", "FWD", 7, 60),
    ]
)


def test_select_squad_hand_calculable_optimum():
    """Optimal 34m squad is gk_good+def1+def2+mid1+mid2+fwd_cheap (cost 34, 292 pts) - verified by hand."""
    squad = select_squad(OPTIMUM_POOL, SMALL_RULES)
    assert sorted(squad["web_name"]) == ["def1", "def2", "fwd_cheap", "gk_good", "mid1", "mid2"]
    assert squad["now_cost_million"].sum() == 34
    assert squad["predicted_points"].sum() == 292


def test_select_squad_infeasible_when_over_budget():
    """Even the cheapest valid composition (34m, all-cheapest-per-slot) can't fit a 5m budget - must raise, not silently fail."""
    tiny_budget_rules = SquadRules(**{**SMALL_RULES.__dict__, "budget_million": 5.0})
    with pytest.raises(InfeasibleSquadError):
        select_squad(OPTIMUM_POOL, tiny_budget_rules)


def test_select_squad_infeasible_when_club_cap_blocks_composition():
    """Composition needs 2 DEF, but all 3 DEF candidates share one club and the cap only allows 1 from it."""
    one_club_defenders = pd.DataFrame(
        [
            _player("gk_good", "GKC", "GKP", 5, 40),
            _player("def1", "SAME", "DEF", 5, 45),
            _player("def2", "SAME", "DEF", 5, 42),
            _player("def3", "SAME", "DEF", 5, 20),
            _player("mid1", "MC1", "MID", 6, 55),
            _player("mid2", "MC2", "MID", 6, 50),
            _player("fwd_cheap", "FC2", "FWD", 7, 60),
        ]
    )
    capped_rules = SquadRules(**{**SMALL_RULES.__dict__, "max_players_per_club": 1})
    with pytest.raises(InfeasibleSquadError):
        select_squad(one_club_defenders, capped_rules)


# 8-player squad (2 GKP, 3 DEF, 2 MID, 1 FWD) with formation_min summing exactly to the 5-man XI size,
# so the only real choice is *which* 2 of 3 DEF and 1 of 2 MID start - fully hand-calculable.
XI_RULES = SquadRules(
    budget_million=100.0, squad_size=8,
    squad_composition={"GKP": 2, "DEF": 3, "MID": 2, "FWD": 1},
    starting_xi_size=5,
    formation_min={"GKP": 1, "DEF": 2, "MID": 1, "FWD": 1},
    max_players_per_club=3, captain_multiplier=2, triple_captain_multiplier=3,
)

XI_SQUAD = pd.DataFrame(
    [
        _player("gk1", "C1", "GKP", 5, 50),
        _player("gk2", "C2", "GKP", 5, 45),
        _player("def1", "C3", "DEF", 5, 40),
        _player("def2", "C4", "DEF", 5, 38),
        _player("def3", "C5", "DEF", 5, 10),
        _player("mid1", "C6", "MID", 6, 42),
        _player("mid2", "C7", "MID", 6, 39),
        _player("fwd1", "C8", "FWD", 7, 55),
    ]
)


def test_select_starting_xi_picks_exactly_one_goalkeeper():
    """Squad has 2 GKP but only 1 may start, regardless of formation_min saying ">=1" - a hardcoded football rule."""
    starting_xi, bench = select_starting_xi(XI_SQUAD, XI_RULES)
    assert (starting_xi["position"] == "GKP").sum() == 1
    assert set(starting_xi["web_name"]) == {"gk1", "def1", "def2", "mid1", "fwd1"}
    assert set(bench["web_name"]) == {"gk2", "def3", "mid2"}


def test_select_starting_xi_infeasible_when_formation_cannot_be_met():
    """Formation requires >=3 DEF to start, but the squad only contains one defender - must raise, not silently fail."""
    thin_defense = pd.DataFrame(
        [
            _player("gk1", "C1", "GKP", 5, 50),
            _player("def1", "C3", "DEF", 5, 40),
            _player("mid1", "C6", "MID", 6, 42),
            _player("mid2", "C7", "MID", 6, 39),
            _player("fwd1", "C8", "FWD", 7, 55),
        ]
    )
    strict_rules = SquadRules(**{**XI_RULES.__dict__, "starting_xi_size": 5, "formation_min": {"GKP": 1, "DEF": 3, "MID": 1, "FWD": 1}})
    with pytest.raises(InfeasibleSquadError):
        select_starting_xi(thin_defense, strict_rules)
