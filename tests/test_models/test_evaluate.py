# Checks simulate_season_transfers's mechanics on a tiny synthetic table (the real ablation uses real data).

import pandas as pd
import pytest

from fpl_agent.models.evaluate import build_horizon_points, simulate_season_transfers
from fpl_agent.optimizer.constraints import SquadRules

RULES = SquadRules(
    budget_million=100.0, squad_size=4,
    squad_composition={"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},
    starting_xi_size=4, formation_min={"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},
    max_players_per_club=2, captain_multiplier=2, triple_captain_multiplier=3,
    free_transfers_per_week=1, free_transfers_max_banked=5, points_hit_per_extra_transfer=-4,
)

POSITIONS = ("GKP", "DEF", "MID", "FWD")


TEAM_IDS = {(position, variant): i for i, (position, variant) in enumerate(
    (p, v) for p in POSITIONS for v in ("strong", "weak")
)}


# Two prior seasons plus a 3-GW target season, 2 candidates per position each on their own team.
def _build_synthetic_table() -> pd.DataFrame:
    rows = []
    element = 1
    for season in ("s0", "s1"):
        for gw in (1, 2, 3):
            for position in POSITIONS:
                for variant, points in (("strong", 8), ("weak", 2)):
                    team_id = TEAM_IDS[(position, variant)]
                    rows.append(
                        {
                            "season": season, "GW": gw, "element": element, "position": position,
                            "name": f"{position}_{variant}", "team": f"team_{team_id}", "team_id": team_id, "value": 50,
                            "selected": 10.0, "total_points": points, "difficulty": 3, "fixture_count": 1,
                            "opponent_attack_strength": 1000, "opponent_defence_strength": 1000,
                        }
                    )
                    element += 1
    target_season = "s2"
    for gw in (1, 2):
        for position in POSITIONS:
            for variant, points in (("strong", 8), ("weak", 2)):
                team_id = TEAM_IDS[(position, variant)]
                rows.append(
                    {
                        "season": target_season, "GW": gw, "element": element, "position": position,
                        "name": f"{position}_{variant}", "team": f"team_{team_id}", "team_id": team_id, "value": 50,
                        "selected": 10.0, "total_points": points, "difficulty": 3, "fixture_count": 1,
                        "opponent_attack_strength": 1000, "opponent_defence_strength": 1000, "minutes": 90,
                    }
                )
                element += 1
    return pd.DataFrame(rows)


# The simulation must run end to end and produce one weekly_log row per gameweek matching cumulative_points.
def test_simulate_season_transfers_runs_and_produces_a_plausible_weekly_log():
    table = _build_synthetic_table()
    result = simulate_season_transfers(table, "s2", first_gw=1, last_gw=2, horizon=1, rules=RULES, use_season_planner=False)

    assert list(result["weekly_log"]["GW"]) == [1, 2]
    assert result["cumulative_points"] == pytest.approx(result["weekly_log"]["cumulative_points"].iloc[-1])
    assert len(result["final_squad"]) == RULES.squad_size


# The use_season_planner=True path (core-player classification feeding the transfer search) must also run cleanly.
def test_simulate_season_transfers_works_with_season_planner_guidance_too():
    table = _build_synthetic_table()
    result = simulate_season_transfers(table, "s2", first_gw=1, last_gw=2, horizon=1, rules=RULES, use_season_planner=True)

    assert list(result["weekly_log"]["GW"]) == [1, 2]
    assert len(result["final_squad"]) == RULES.squad_size


# A model that just sums its input columns, standing in for a trained LightGBM booster in these unit tests.
class _SumModel:
    def predict(self, X):
        return X.sum(axis=1)


# A blank gameweek exactly at as_of_gw must not zero the whole horizon - other weeks still use the player's last known form.
def test_build_horizon_points_carries_form_forward_across_a_blank_at_as_of_gw():
    table = pd.DataFrame(
        [
            {"season": "s", "GW": 1, "name": "A", "position": "MID", "form_roll3": 5.0, "fixture_count": 1},
            {"season": "s", "GW": 3, "name": "A", "position": "MID", "form_roll3": 5.0, "fixture_count": 1},
        ]
    )
    result = build_horizon_points(
        table, "s", as_of_gw=2, horizon=2, models={"MID": _SumModel()}, columns=["form_roll3", "fixture_count"]
    )

    assert result.loc[result["web_name"] == "A", "horizon_points"].iloc[0] > 0


# A gameweek the player genuinely blanks (no fixture) must still contribute 0, even once form carries forward.
def test_build_horizon_points_still_zeroes_the_blank_gameweek_itself():
    table = pd.DataFrame(
        [
            {"season": "s", "GW": 1, "name": "A", "position": "MID", "form_roll3": 5.0, "fixture_count": 1},
            {"season": "s", "GW": 3, "name": "A", "position": "MID", "form_roll3": 5.0, "fixture_count": 1},
        ]
    )
    result = build_horizon_points(
        table, "s", as_of_gw=2, horizon=1, models={"MID": _SumModel()}, columns=["form_roll3", "fixture_count"]
    )

    assert result.loc[result["web_name"] == "A", "horizon_points"].iloc[0] == 0
