"""Checks the CP-SAT squad/starting-XI optimizer against a hand-calculable optimum and infeasible inputs."""

import pandas as pd
import pytest

from fpl_agent.optimizer.constraints import SquadRules
from fpl_agent.optimizer.squad_optimizer import InfeasibleSquadError, select_squad, select_squad_and_starting_xi, select_starting_xi

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


JOINT_RULES = SquadRules(
    budget_million=15.0, squad_size=3,
    squad_composition={"GKP": 2, "MID": 1},
    starting_xi_size=2,
    formation_min={"GKP": 1, "DEF": 0, "MID": 1, "FWD": 0},
    max_players_per_club=3, captain_multiplier=2, triple_captain_multiplier=3,
)

JOINT_POOL = pd.DataFrame(
    [
        _player("gk_starter", "C1", "GKP", 6, 30),
        _player("gk_backup_cheap", "C2", "GKP", 3, 12),
        _player("gk_backup_expensive", "C3", "GKP", 5, 20),
        _player("mid_basic", "C4", "MID", 4, 20),
        _player("mid_great", "C5", "MID", 6, 26),
    ]
)


# Plain select_squad only sees raw point totals, so a pricier backup GK (which will never start) can outbid a better starting MID for budget.
def test_select_squad_alone_overspends_on_a_bench_only_player():
    squad = select_squad(JOINT_POOL, JOINT_RULES)
    assert sorted(squad["web_name"]) == ["gk_backup_expensive", "gk_starter", "mid_basic"]


# select_squad_and_starting_xi weights bench points down, so it should instead spend that budget on the starting MID and settle for a cheap bench GK.
def test_select_squad_and_starting_xi_prefers_a_cheap_bench_over_a_pricier_one():
    squad, starting_xi, bench = select_squad_and_starting_xi(JOINT_POOL, JOINT_RULES)
    assert sorted(squad["web_name"]) == ["gk_backup_cheap", "gk_starter", "mid_great"]
    assert set(starting_xi["web_name"]) == {"gk_starter", "mid_great"}
    assert set(bench["web_name"]) == {"gk_backup_cheap"}


def test_select_squad_and_starting_xi_raises_when_infeasible():
    tiny_budget_rules = SquadRules(**{**JOINT_RULES.__dict__, "budget_million": 5.0})
    with pytest.raises(InfeasibleSquadError):
        select_squad_and_starting_xi(JOINT_POOL, tiny_budget_rules)
