"""Weekly transfer optimizer: searches 0-2 transfer moves maximizing horizon points net of transfer-hit costs."""

from __future__ import annotations

import pandas as pd
from ortools.sat.python import cp_model

from fpl_agent.optimizer.constraints import SquadRules, load_rules
from fpl_agent.optimizer.squad_optimizer import COST_SCALE, POINTS_SCALE, InfeasibleSquadError

MAX_TRANSFERS_SEARCHED = 2


def recommend_transfers(
    current_squad: pd.DataFrame,
    player_pool: pd.DataFrame,
    bank: float,
    free_transfers: int,
    rules: SquadRules | None = None,
    max_transfers: int = MAX_TRANSFERS_SEARCHED,
) -> dict:
    """Recommends the best 0-2 transfer move and explains it, including why a hit was or wasn't worth taking."""
    rules = rules or load_rules()
    squad = current_squad.reset_index(drop=True)
    pool = player_pool[~player_pool["player_id"].isin(squad["player_id"])].reset_index(drop=True)
    hit_cost = abs(rules.points_hit_per_extra_transfer)

    best = _solve_transfer_move(squad, pool, bank, free_transfers, rules, max_transfers, min_transfers=0)
    if best is None:
        raise InfeasibleSquadError(
            "No transfer combination (including making none) satisfies the budget/composition/club-cap constraints."
        )

    dropped = squad.loc[[i for i in range(len(squad)) if i not in best["kept_idx"]]]
    bought = pool.loc[best["added_idx"]]
    best_net = best["horizon_points"] - best["hits"] * hit_cost

    reasoning = []
    if best["num_transfers"] == 0:
        reasoning.append("No transfer recommended - no available swap improves horizon points enough to be worth it.")
    else:
        dropped_by_position = dropped.sort_values("position")
        bought_by_position = bought.sort_values("position")
        for (_, sell), (_, buy) in zip(dropped_by_position.iterrows(), bought_by_position.iterrows()):
            reasoning.append(
                f"{buy['web_name']} in for {sell['web_name']}: "
                f"{buy['horizon_points']:.1f} vs {sell['horizon_points']:.1f} predicted points over the horizon."
            )
        if best["hits"] > 0:
            reasoning.append(f"{best['hits']} hit(s) taken (-{best['hits'] * hit_cost:.0f} pts) - the gain clearly exceeds the cost.")

    alternative = None
    if best["hits"] == 0 and free_transfers < max_transfers:
        alternative = _solve_transfer_move(squad, pool, bank, free_transfers, rules, max_transfers, min_transfers=free_transfers + 1)
    if alternative is not None:
        alt_net = alternative["horizon_points"] - alternative["hits"] * hit_cost
        if alt_net <= best_net:
            reasoning.append(
                f"A hit-taking alternative was considered ({alternative['hits']} hit(s), {alternative['horizon_points']:.1f} pts) "
                f"but nets {alt_net:.1f} vs {best_net:.1f} without it - hit not worth it."
            )

    return {
        "dropped": dropped,
        "bought": bought,
        "num_transfers": best["num_transfers"],
        "hits": best["hits"],
        "horizon_points": best["horizon_points"],
        "net_horizon_points": best_net,
        "reasoning": reasoning,
    }


def _solve_transfer_move(
    squad: pd.DataFrame, pool: pd.DataFrame, bank: float, free_transfers: int, rules: SquadRules,
    max_transfers: int, min_transfers: int,
) -> dict | None:
    """Solves one CP-SAT transfer search, bounding the number of transfers to [min_transfers, max_transfers]."""
    n_squad, n_pool = len(squad), len(pool)
    model = cp_model.CpModel()
    keep = [model.new_bool_var(f"keep_{i}") for i in range(n_squad)]
    add = [model.new_bool_var(f"add_{j}") for j in range(n_pool)]

    model.add(sum(keep) + sum(add) == rules.squad_size)

    for position, count in rules.squad_composition.items():
        squad_idx = squad.index[squad["position"] == position].tolist()
        pool_idx = pool.index[pool["position"] == position].tolist()
        model.add(sum(keep[i] for i in squad_idx) + sum(add[j] for j in pool_idx) == count)

    for club in pd.concat([squad["team_name"], pool["team_name"]]).unique():
        squad_idx = squad.index[squad["team_name"] == club].tolist()
        pool_idx = pool.index[pool["team_name"] == club].tolist()
        model.add(sum(keep[i] for i in squad_idx) + sum(add[j] for j in pool_idx) <= rules.max_players_per_club)

    squad_cost = (squad["now_cost_million"] * COST_SCALE).round().astype(int)
    pool_cost = (pool["now_cost_million"] * COST_SCALE).round().astype(int)
    final_cost = sum(keep[i] * int(squad_cost[i]) for i in range(n_squad)) + sum(add[j] * int(pool_cost[j]) for j in range(n_pool))
    available_tenths = int(round(bank * COST_SCALE)) + int(squad_cost.sum())
    model.add(final_cost <= available_tenths)

    num_transfers = sum(add)
    model.add(num_transfers <= max_transfers)
    model.add(num_transfers >= min_transfers)

    hits = model.new_int_var(0, max_transfers, "hits")
    model.add(hits >= num_transfers - free_transfers)

    squad_points = (squad["horizon_points"] * POINTS_SCALE).round().astype(int)
    pool_points = (pool["horizon_points"] * POINTS_SCALE).round().astype(int)
    total_points = sum(keep[i] * int(squad_points[i]) for i in range(n_squad)) + sum(add[j] * int(pool_points[j]) for j in range(n_pool))
    hit_cost_scaled = int(abs(rules.points_hit_per_extra_transfer) * POINTS_SCALE)
    model.maximize(total_points - hits * hit_cost_scaled)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    added_idx = [j for j in range(n_pool) if solver.value(add[j]) == 1]
    return {
        "kept_idx": [i for i in range(n_squad) if solver.value(keep[i]) == 1],
        "added_idx": added_idx,
        "num_transfers": len(added_idx),
        "hits": solver.value(hits),
        "horizon_points": solver.value(total_points) / POINTS_SCALE,
    }
