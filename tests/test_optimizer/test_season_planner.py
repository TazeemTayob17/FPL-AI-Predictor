# Checks core/rotational classification, price momentum, chip-window scoring, wildcard signal, and mini-league risk posture.

import pandas as pd
import pytest

from fpl_agent.optimizer.season_planner import (
    build_season_plan,
    classify_core_vs_rotational,
    forward_chip_window_targets,
    mini_league_risk_posture,
    price_momentum,
    squad_value_trajectory,
    wildcard_fixture_swing_signal,
)

SQUAD = pd.DataFrame(
    [
        {"player_id": 1, "web_name": "def_expensive_reliable", "team_id": 1, "position": "DEF", "predicted_points": 5.0, "now_cost_million": 8.0, "chance_of_playing_next_round": 100, "minutes_volatility": 5.0},
        {"player_id": 2, "web_name": "def_cheap_reliable", "team_id": 2, "position": "DEF", "predicted_points": 5.0, "now_cost_million": 4.0, "chance_of_playing_next_round": 100, "minutes_volatility": 5.0},
        {"player_id": 3, "web_name": "fwd_expensive_doubtful", "team_id": 1, "position": "FWD", "predicted_points": 8.0, "now_cost_million": 12.0, "chance_of_playing_next_round": 50, "minutes_volatility": 5.0},
        {"player_id": 4, "web_name": "fwd_expensive_reliable", "team_id": 2, "position": "FWD", "predicted_points": 8.0, "now_cost_million": 12.0, "chance_of_playing_next_round": 100, "minutes_volatility": 5.0},
    ]
)


# Price gates "core", not points: budget-priced is rotational even with equal predicted_points to a premium teammate.
def test_classify_core_vs_rotational_requires_both_reliability_and_premium_price():
    result = classify_core_vs_rotational(SQUAD)
    by_name = result.set_index("web_name")["classification"]
    assert by_name["def_expensive_reliable"] == "core"        # top of DEF price tier (8.0 vs 4.0) and reliable
    assert by_name["def_cheap_reliable"] == "rotational"      # budget-priced DEF, despite equal predicted_points
    assert by_name["fwd_expensive_doubtful"] == "rotational"  # premium-priced but doubtful (chance=50)
    assert by_name["fwd_expensive_reliable"] == "core"        # premium-priced and reliable


# Large net transfers-in must flag a likely rise, large net transfers-out a likely fall, small net stays stable.
def test_price_momentum_flags_rise_and_fall_from_net_transfers():
    players = pd.DataFrame(
        [
            {"player_id": 1, "transfers_in_event": 100_000, "transfers_out_event": 1_000},
            {"player_id": 2, "transfers_in_event": 1_000, "transfers_out_event": 100_000},
            {"player_id": 3, "transfers_in_event": 5_000, "transfers_out_event": 4_000},
        ]
    )
    result = price_momentum(players).set_index("player_id")["price_trend"]
    assert result[1] == "likely_rise"
    assert result[2] == "likely_fall"
    assert result[3] == "stable"


# Player 3's price must not contribute to the squad's total value - only players 1 and 2 are in the squad.
def test_squad_value_trajectory_sums_only_squad_members_per_timestamp():
    history = pd.DataFrame(
        [
            {"pulled_at": "t1", "player_id": 1, "now_cost": 100},
            {"pulled_at": "t1", "player_id": 2, "now_cost": 50},
            {"pulled_at": "t1", "player_id": 3, "now_cost": 999},
            {"pulled_at": "t2", "player_id": 1, "now_cost": 101},
            {"pulled_at": "t2", "player_id": 2, "now_cost": 50},
        ]
    )
    result = squad_value_trajectory(history, squad_player_ids={1, 2})
    assert result.set_index("pulled_at")["total_value"].to_dict() == {"t1": 150, "t2": 151}


# No snapshot rows for this squad yet must return an empty frame, not crash.
def test_squad_value_trajectory_handles_no_matching_history():
    result = squad_value_trajectory(pd.DataFrame(columns=["pulled_at", "player_id", "now_cost"]), squad_player_ids={1})
    assert result.empty


CHIP_SQUAD = pd.DataFrame(
    [
        {"player_id": 1, "web_name": "team_a_player", "team_id": 1, "classification": "core", "predicted_points": 6.0},
        {"player_id": 2, "web_name": "team_b_player", "team_id": 2, "classification": "core", "predicted_points": 4.0},
    ]
)
TEAM_FIXTURE_COUNTS = pd.DataFrame(
    [
        {"GW": 1, "team_id": 1, "fixture_count": 2},  # team A has a double in GW1
        {"GW": 1, "team_id": 2, "fixture_count": 1},
        {"GW": 2, "team_id": 2, "fixture_count": 1},  # team A absent from GW2 -> blank
    ]
)


# GW1: team A's double counts toward bench_boost/triple_captain. GW2: team A's blank counts toward free_hit.
def test_forward_chip_window_targets_scores_doubles_and_blanks_correctly():
    result = forward_chip_window_targets(CHIP_SQUAD, TEAM_FIXTURE_COUNTS, gw_range=range(1, 3)).set_index("GW")
    assert result.loc[1, "bench_boost_score"] == 1
    assert result.loc[1, "triple_captain_candidate"] == "team_a_player"
    assert result.loc[1, "free_hit_score"] == 0
    assert result.loc[2, "free_hit_score"] == 1
    assert result.loc[2, "bench_boost_score"] == 0


TEAM_DIFFICULTY = pd.DataFrame(
    [
        {"GW": 1, "team_id": 1, "difficulty": 5},  # squad's team - hard fixtures
        {"GW": 1, "team_id": 2, "difficulty": 2},
        {"GW": 1, "team_id": 3, "difficulty": 2},
    ]
)


# Squad (team 1, difficulty 5) is much harder than the league average (2, 2, 5) -> wildcard suggested.
def test_wildcard_signal_fires_when_squad_fixtures_are_notably_worse_than_average():
    squad = pd.DataFrame([{"team_id": 1}])
    result = wildcard_fixture_swing_signal(squad, TEAM_DIFFICULTY, gw_range=range(1, 2))
    assert result["suggest_wildcard"] is True
    assert result["gap"] > 0


# A squad spread evenly across all three teams has the same average difficulty as the league -> no signal.
def test_wildcard_signal_does_not_fire_when_squad_fixtures_are_average():
    squad = pd.DataFrame([{"team_id": 1}, {"team_id": 2}, {"team_id": 3}])
    result = wildcard_fixture_swing_signal(squad, TEAM_DIFFICULTY, gw_range=range(1, 2))
    assert result["suggest_wildcard"] is False


STANDINGS = pd.DataFrame(
    [
        {"team_id": 111, "rank": 1, "points_behind_leader": 0},
        {"team_id": 222, "rank": 2, "points_behind_leader": 30},
        {"team_id": 333, "rank": 3, "points_behind_leader": 0},
    ]
)


# Even trailing, a huge number of GWs remaining means no need to force a posture change yet.
def test_risk_posture_is_balanced_with_plenty_of_the_season_left_regardless_of_rank():
    result = mini_league_risk_posture(STANDINGS, team_id=222, gws_remaining=20)
    assert result["posture"] == "balanced"


# Rank 1 with few GWs left -> protect (template-heavy).
def test_risk_posture_protects_a_late_season_lead():
    result = mini_league_risk_posture(STANDINGS, team_id=111, gws_remaining=5)
    assert result["posture"] == "protect"


# Behind on points with few GWs left -> chase (differential-heavy).
def test_risk_posture_chases_when_behind_late_in_the_season():
    result = mini_league_risk_posture(STANDINGS, team_id=222, gws_remaining=5)
    assert result["posture"] == "chase"
    assert result["points_behind_leader"] == 30


# Not leading but not behind either (tied on points) -> balanced, not chase.
def test_risk_posture_is_balanced_when_tied_late_in_the_season():
    result = mini_league_risk_posture(STANDINGS, team_id=333, gws_remaining=5)
    assert result["posture"] == "balanced"


# No mini-league configured (empty standings) must not crash, and must default to balanced.
def test_risk_posture_defaults_to_balanced_when_no_standings_available():
    result = mini_league_risk_posture(pd.DataFrame(columns=["team_id", "rank", "points_behind_leader"]), team_id=111, gws_remaining=5)
    assert result["posture"] == "balanced"
    assert result["rank"] is None


# The convenience core_player_ids property must match whichever players classify_core_vs_rotational tagged "core".
def test_build_season_plan_core_player_ids_matches_classification():
    plan = build_season_plan(SQUAD, TEAM_FIXTURE_COUNTS, TEAM_DIFFICULTY, gw_range=range(1, 3))
    assert plan.core_player_ids == {1, 4}
