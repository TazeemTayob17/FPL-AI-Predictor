# Checks the availability-scaling and set-piece-boost adjustments applied on top of raw model predictions.

import pandas as pd

from fpl_agent.models.live_adjustments import apply_availability_scaling, apply_session_overrides, apply_set_piece_boost

PREDICTIONS = pd.DataFrame(
    [
        {"player_id": 1, "web_name": "Nailed", "predicted_points": 6.0, "horizon_points": 30.0},
        {"player_id": 2, "web_name": "Doubtful", "predicted_points": 6.0, "horizon_points": 30.0},
        {"player_id": 3, "web_name": "Unlisted", "predicted_points": 6.0, "horizon_points": 30.0},
    ]
)


# A player with no chance-of-playing entry at all (the normal case - FPL only sets it when there's a doubt) must be left unscaled.
def test_apply_availability_scaling_leaves_a_fully_fit_player_unchanged():
    status = pd.DataFrame([{"player_id": 1, "chance_of_playing_next_round": None}])
    result = apply_availability_scaling(PREDICTIONS, status)
    assert result.loc[result["player_id"] == 1, "predicted_points"].iloc[0] == 6.0


# A 50% doubt must scale both predicted_points and horizon_points down proportionally, not just one of them.
def test_apply_availability_scaling_halves_points_for_a_fifty_percent_doubt():
    status = pd.DataFrame([{"player_id": 2, "chance_of_playing_next_round": 50}])
    result = apply_availability_scaling(PREDICTIONS, status)
    row = result[result["player_id"] == 2].iloc[0]
    assert row["predicted_points"] == 3.0
    assert row["horizon_points"] == 15.0


# A player with no player_status row at all (not just a null chance) must default to fully available, not crash or zero out.
def test_apply_availability_scaling_defaults_missing_status_row_to_fully_available():
    status = pd.DataFrame([{"player_id": 999, "chance_of_playing_next_round": 0}])
    result = apply_availability_scaling(PREDICTIONS, status)
    assert result.loc[result["player_id"] == 3, "predicted_points"].iloc[0] == 6.0


# The primary penalty taker gets the full boost; a player not on any set-piece list is untouched.
def test_apply_set_piece_boost_boosts_the_primary_penalty_taker():
    players = pd.DataFrame(
        [
            {"player_id": 1, "penalties_order": 1, "corners_and_indirect_freekicks_order": None, "direct_freekicks_order": None},
            {"player_id": 2, "penalties_order": None, "corners_and_indirect_freekicks_order": None, "direct_freekicks_order": None},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"player_id": 1, "predicted_points": 10.0},
            {"player_id": 2, "predicted_points": 10.0},
        ]
    )
    result = apply_set_piece_boost(predictions, players)
    assert result.loc[result["player_id"] == 1, "predicted_points"].iloc[0] == 11.5
    assert result.loc[result["player_id"] == 2, "predicted_points"].iloc[0] == 10.0


# A player on multiple set-piece duties (penalties + corners) stacks both boosts rather than only applying one.
def test_apply_set_piece_boost_stacks_multiple_duties():
    players = pd.DataFrame(
        [{"player_id": 1, "penalties_order": 1, "corners_and_indirect_freekicks_order": 1, "direct_freekicks_order": None}]
    )
    predictions = pd.DataFrame([{"player_id": 1, "predicted_points": 10.0}])
    result = apply_set_piece_boost(predictions, players)
    assert result.loc[0, "predicted_points"] == 12.0


# Missing set-piece order columns entirely (e.g. an older cached players frame) must leave predictions untouched, not crash.
def test_apply_set_piece_boost_is_a_no_op_when_order_columns_are_absent():
    players = pd.DataFrame([{"player_id": 1}])
    predictions = pd.DataFrame([{"player_id": 1, "predicted_points": 10.0}])
    result = apply_set_piece_boost(predictions, players)
    assert result.loc[0, "predicted_points"] == 10.0


# One visitor's personal correction must scale only that player, exactly like the shared-table version, and no session_overrides at all must be a pure no-op.
def test_apply_session_overrides_scales_only_the_overridden_player():
    result = apply_session_overrides(PREDICTIONS, {2: {"chance_of_playing_next_round": "50"}})
    assert result.loc[result["player_id"] == 2, "predicted_points"].iloc[0] == 3.0
    assert result.loc[result["player_id"] == 1, "predicted_points"].iloc[0] == 6.0


def test_apply_session_overrides_is_a_no_op_with_no_overrides():
    result = apply_session_overrides(PREDICTIONS, {})
    pd.testing.assert_frame_equal(result, PREDICTIONS)
