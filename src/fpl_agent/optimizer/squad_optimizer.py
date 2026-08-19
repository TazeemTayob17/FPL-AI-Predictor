"""CP-SAT optimizer: selects a rules-compliant 15-man squad and starting XI maximizing predicted points."""

from __future__ import annotations

import pandas as pd
from ortools.sat.python import cp_model

from fpl_agent.optimizer.constraints import SquadRules, load_rules

POINTS_SCALE = 100
COST_SCALE = 10
BENCH_POINTS_WEIGHT_PERCENT = 5  # a benched player's points barely count in the objective, so budget concentrates on the starting XI instead of a "balanced" but bench-heavy 15


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
    model.add(sum(starts[i] for i in gkp_idx) == 1)

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


# Jointly picks a 15-man squad and its starting XI in one solve, weighting bench points down so budget concentrates on the XI instead of a "balanced" but bench-heavy squad - use this for building a fresh squad; for picking the best XI from an already-owned squad, use select_starting_xi instead.
def select_squad_and_starting_xi(
    players: pd.DataFrame, rules: SquadRules | None = None, bench_weight_percent: int = BENCH_POINTS_WEIGHT_PERCENT
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rules = rules or load_rules()
    players = players.reset_index(drop=True)
    n = len(players)

    model = cp_model.CpModel()
    picks = [model.new_bool_var(f"pick_{i}") for i in range(n)]
    starts = [model.new_bool_var(f"start_{i}") for i in range(n)]

    model.add(sum(picks) == rules.squad_size)
    model.add(sum(starts) == rules.starting_xi_size)
    for i in range(n):
        model.add(starts[i] <= picks[i])

    cost_tenths = (players["now_cost_million"] * COST_SCALE).round().astype(int)
    budget_tenths = int(round(rules.budget_million * COST_SCALE))
    model.add(sum(picks[i] * int(cost_tenths[i]) for i in range(n)) <= budget_tenths)

    for position, count in rules.squad_composition.items():
        idx = players.index[players["position"] == position].tolist()
        model.add(sum(picks[i] for i in idx) == count)

    for club in players["team_name"].unique():
        idx = players.index[players["team_name"] == club].tolist()
        model.add(sum(picks[i] for i in idx) <= rules.max_players_per_club)

    gkp_idx = players.index[players["position"] == "GKP"].tolist()
    model.add(sum(starts[i] for i in gkp_idx) == 1)
    for position in ("DEF", "MID", "FWD"):
        idx = players.index[players["position"] == position].tolist()
        model.add(sum(starts[i] for i in idx) >= rules.formation_min[position])

    scaled_points = (players["predicted_points"] * POINTS_SCALE).round().astype(int)
    objective_terms = []
    for i in range(n):
        objective_terms.append(starts[i] * int(scaled_points[i]))
        bench_points = int(scaled_points[i]) * bench_weight_percent // 100
        objective_terms.append((picks[i] - starts[i]) * bench_points)
    model.maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise InfeasibleSquadError(
            "No 15-man squad + starting XI satisfies the budget/composition/club-cap/formation constraints for this player pool."
        )

    squad_idx = [i for i in range(n) if solver.value(picks[i]) == 1]
    starting_idx_set = {i for i in range(n) if solver.value(starts[i]) == 1}
    squad = players.loc[squad_idx].reset_index(drop=True)
    starting_xi = players.loc[[i for i in squad_idx if i in starting_idx_set]].reset_index(drop=True)
    bench = players.loc[[i for i in squad_idx if i not in starting_idx_set]].reset_index(drop=True)
    return squad, starting_xi, bench


def _solve(model: cp_model.CpModel, bool_vars: list, n: int) -> list[int] | None:
    """Runs the CP-SAT solver and returns the indices of true boolean vars, or None if infeasible."""
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [i for i in range(n) if solver.value(bool_vars[i]) == 1]
