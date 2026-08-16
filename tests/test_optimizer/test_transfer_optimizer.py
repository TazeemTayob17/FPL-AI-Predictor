# Checks the transfer optimizer's 0/1/2-transfer search: correct swaps, hit economics, and constraint enforcement.

import pandas as pd
import pytest

from fpl_agent.optimizer.constraints import SquadRules
from fpl_agent.optimizer.squad_optimizer import InfeasibleSquadError
from fpl_agent.optimizer.transfer_optimizer import recommend_transfers

RULES = SquadRules(
    budget_million=100.0, squad_size=4,
    squad_composition={"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},
    starting_xi_size=4, formation_min={"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},
    max_players_per_club=2, captain_multiplier=2, triple_captain_multiplier=3,
    free_transfers_per_week=1, free_transfers_max_banked=5, points_hit_per_extra_transfer=-4,
)

SQUAD = pd.DataFrame(
    [
        {"player_id": 1, "web_name": "gk1", "team_name": "A", "position": "GKP", "now_cost_million": 5.0, "horizon_points": 20.0},
        {"player_id": 2, "web_name": "def1", "team_name": "B", "position": "DEF", "now_cost_million": 5.0, "horizon_points": 15.0},
        {"player_id": 3, "web_name": "mid1", "team_name": "C", "position": "MID", "now_cost_million": 5.0, "horizon_points": 15.0},
        {"player_id": 4, "web_name": "fwd1", "team_name": "D", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 15.0},
    ]
)


# Builds a candidate pool of the current squad plus extra replacement candidates.
def _pool_with(extra_rows):
    return pd.concat([SQUAD, pd.DataFrame(extra_rows)], ignore_index=True)


# With no better replacements in the pool, the optimizer must recommend making zero transfers.
def test_no_transfer_when_no_upgrade_available():
    result = recommend_transfers(SQUAD, SQUAD.copy(), bank=0.0, free_transfers=1, rules=RULES)
    assert result["num_transfers"] == 0
    assert result["hits"] == 0


# A big, affordable upgrade within the free transfer limit should be taken with zero hits.
def test_clear_free_upgrade_is_taken_without_a_hit():
    pool = _pool_with([{"player_id": 5, "web_name": "fwd_great", "team_name": "E", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 40.0}])
    result = recommend_transfers(SQUAD, pool, bank=0.0, free_transfers=1, rules=RULES)
    assert result["num_transfers"] == 1
    assert result["hits"] == 0
    assert result["bought"]["web_name"].iloc[0] == "fwd_great"
    assert result["dropped"]["web_name"].iloc[0] == "fwd1"


# Two big upgrades beyond the 1 free transfer should still be taken since the point gain dwarfs the -4 hit.
def test_hit_is_taken_when_gain_clearly_exceeds_cost():
    pool = _pool_with(
        [
            {"player_id": 5, "web_name": "fwd_great", "team_name": "E", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 40.0},
            {"player_id": 6, "web_name": "mid_great", "team_name": "F", "position": "MID", "now_cost_million": 5.0, "horizon_points": 40.0},
        ]
    )
    result = recommend_transfers(SQUAD, pool, bank=0.0, free_transfers=1, rules=RULES)
    assert result["num_transfers"] == 2
    assert result["hits"] == 1
    assert any("hit(s) taken" in line for line in result["reasoning"])


# A marginal second upgrade (+1-2 pts) is not worth a -4 hit, so only the free transfer should be taken.
def test_hit_is_rejected_when_gain_does_not_clearly_exceed_cost():
    pool = _pool_with(
        [
            {"player_id": 5, "web_name": "fwd_ok", "team_name": "E", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 17.0},
            {"player_id": 6, "web_name": "mid_ok", "team_name": "F", "position": "MID", "now_cost_million": 5.0, "horizon_points": 16.0},
        ]
    )
    result = recommend_transfers(SQUAD, pool, bank=0.0, free_transfers=1, rules=RULES)
    assert result["num_transfers"] == 1
    assert result["hits"] == 0
    assert any("hit not worth it" in line for line in result["reasoning"])


# A tempting upgrade that costs more than bank + squad value allows must not be recommended.
def test_unaffordable_upgrade_is_not_selected():
    pool = _pool_with([{"player_id": 5, "web_name": "fwd_expensive", "team_name": "E", "position": "FWD", "now_cost_million": 50.0, "horizon_points": 100.0}])
    result = recommend_transfers(SQUAD, pool, bank=0.0, free_transfers=1, rules=RULES)
    assert result["num_transfers"] == 0


# A great replacement is rejected if it would push its club over the max-players-per-club cap.
def test_club_cap_prevents_an_otherwise_tempting_swap():
    squad_two_from_club_e = pd.concat(
        [SQUAD.iloc[:3], pd.DataFrame([{"player_id": 4, "web_name": "fwd1", "team_name": "E", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 15.0}])],
        ignore_index=True,
    )
    squad_two_from_club_e.loc[1, "team_name"] = "E"
    pool = pd.concat(
        [squad_two_from_club_e, pd.DataFrame([{"player_id": 6, "web_name": "mid_great", "team_name": "E", "position": "MID", "now_cost_million": 5.0, "horizon_points": 40.0}])],
        ignore_index=True,
    )
    result = recommend_transfers(squad_two_from_club_e, pool, bank=0.0, free_transfers=1, rules=RULES)
    assert result["num_transfers"] == 0


# If the current squad itself can't satisfy composition (e.g. a duplicate position slot), the search must raise.
def test_infeasible_when_no_valid_squad_exists_even_with_zero_transfers():
    broken_squad = SQUAD.copy()
    broken_squad.loc[0, "position"] = "DEF"
    with pytest.raises(InfeasibleSquadError):
        recommend_transfers(broken_squad, broken_squad.copy(), bank=0.0, free_transfers=1, rules=RULES)


# A 1.0 pt upgrade normally goes through, but the 2.0 pt core-sell penalty makes it net negative here.
def test_core_player_alignment_penalty_blocks_a_marginal_sale():
    pool = _pool_with([{"player_id": 5, "web_name": "fwd_slightly_better", "team_name": "E", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 16.0}])
    result = recommend_transfers(SQUAD, pool, bank=0.0, free_transfers=1, rules=RULES, core_player_ids={4})
    assert result["num_transfers"] == 0


# The penalty is soft: a 25 pt upgrade still goes through despite selling a flagged "core" player.
def test_core_player_alignment_penalty_does_not_block_a_large_enough_gain():
    pool = _pool_with([{"player_id": 5, "web_name": "fwd_great", "team_name": "E", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 40.0}])
    result = recommend_transfers(SQUAD, pool, bank=0.0, free_transfers=1, rules=RULES, core_player_ids={4})
    assert result["num_transfers"] == 1
    assert result["bought"]["web_name"].iloc[0] == "fwd_great"
    assert any("core player" in line for line in result["reasoning"])


# Without core_player_ids, behavior must be identical to before this feature existed - no penalty applied anywhere.
def test_core_player_alignment_penalty_is_a_no_op_when_no_core_players_flagged():
    pool = _pool_with([{"player_id": 5, "web_name": "fwd_slightly_better", "team_name": "E", "position": "FWD", "now_cost_million": 5.0, "horizon_points": 16.0}])
    result = recommend_transfers(SQUAD, pool, bank=0.0, free_transfers=1, rules=RULES)
    assert result["num_transfers"] == 1
    assert result["bought"]["web_name"].iloc[0] == "fwd_slightly_better"
