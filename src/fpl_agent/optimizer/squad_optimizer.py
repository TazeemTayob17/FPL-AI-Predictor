"""CP-SAT optimizer: selects a rules-compliant 15-man squad and starting XI maximizing predicted points."""

from __future__ import annotations

import pandas as pd
from ortools.sat.python import cp_model

from fpl_agent.optimizer.constraints import SquadRules, load_rules

POINTS_SCALE = 100   # preserves 2 decimal places of predicted_points in CP-SAT's integer objective
COST_SCALE = 10      # now_cost_million -> tenths of a million, matching the FPL API's native integer unit


class InfeasibleSquadError(Exception):
    """Raised when no valid selection exists under the given constraints (budget, composition, club cap, formation)."""


def select_squad(players: pd.DataFrame, rules: SquadRules | None = None) -> pd.DataFrame:
    """Solves for the 15-man squad maximizing total predicted_points under budget/composition/club-cap rules."""
    rules = rules or load_rules()
    players = players.reset_index(drop=True)
    n = len(players)

    model = cp_model.CpModel()
    picks = [model.new_bool_var(f"pick_{i}") for i in range(n)]

    model.add(sum(picks) == rules.squad_size)

    cost_tenths = (players["now_cost_million"] * COST_SCALE).round().astype(int)
    budget_tenths = int(round(rules.budget_million * COST_SCALE))
    model.add(sum(picks[i] * int(cost_tenths[i]) for i in range(n)) <= budget_tenths)

    for position, count in rules.squad_composition.items():
        idx = players.index[players["position"] == position].tolist()
        model.add(sum(picks[i] for i in idx) == count)

    for club in players["team_name"].unique():
        idx = players.index[players["team_name"] == club].tolist()
        model.add(sum(picks[i] for i in idx) <= rules.max_players_per_club)

    scaled_points = (players["predicted_points"] * POINTS_SCALE).round().astype(int)
    model.maximize(sum(picks[i] * int(scaled_points[i]) for i in range(n)))

    selected_idx = _solve(model, picks, n)
    if selected_idx is None:
        raise InfeasibleSquadError(
            "No 15-man squad satisfies the budget/composition/club-cap constraints for this player pool."
        )
    return players.loc[selected_idx].reset_index(drop=True)


def select_starting_xi(squad: pd.DataFrame, rules: SquadRules | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Solves for the 11-man starting lineup (valid formation) maximizing predicted points; the rest is the bench."""
    rules = rules or load_rules()
    squad = squad.reset_index(drop=True)
    n = len(squad)

    model = cp_model.CpModel()
    starts = [model.new_bool_var(f"start_{i}") for i in range(n)]

    model.add(sum(starts) == rules.starting_xi_size)

    gkp_idx = squad.index[squad["position"] == "GKP"].tolist()
    model.add(sum(starts[i] for i in gkp_idx) == 1)   # exactly one keeper plays; a football rule, not a config value

    for position in ("DEF", "MID", "FWD"):
        idx = squad.index[squad["position"] == position].tolist()
        model.add(sum(starts[i] for i in idx) >= rules.formation_min[position])

    scaled_points = (squad["predicted_points"] * POINTS_SCALE).round().astype(int)
    model.maximize(sum(starts[i] * int(scaled_points[i]) for i in range(n)))

    starting_idx = _solve(model, starts, n)
    if starting_idx is None:
        raise InfeasibleSquadError("No starting XI satisfies the formation constraints for this squad.")

    bench_idx = [i for i in range(n) if i not in set(starting_idx)]
    starting_xi = squad.loc[starting_idx].reset_index(drop=True)
    bench = squad.loc[bench_idx].reset_index(drop=True)
    return starting_xi, bench


def _solve(model: cp_model.CpModel, bool_vars: list, n: int) -> list[int] | None:
    """Runs the CP-SAT solver and returns the indices of true boolean vars, or None if infeasible."""
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [i for i in range(n) if solver.value(bool_vars[i]) == 1]
