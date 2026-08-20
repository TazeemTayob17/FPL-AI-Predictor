# Guards the real FPL point-value constants against typos - these numbers are the ground truth every decomposed-model recombination depends on.

from fpl_agent.models import scoring_rules


def test_goal_points_match_2025_26_rules_including_the_higher_goalkeeper_value():
    assert scoring_rules.GOAL_POINTS == {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}


def test_clean_sheet_points_are_zero_for_forwards():
    assert scoring_rules.CLEAN_SHEET_POINTS["FWD"] == 0
    assert scoring_rules.CLEAN_SHEET_POINTS["GKP"] == scoring_rules.CLEAN_SHEET_POINTS["DEF"] == 4


def test_defensive_contribution_threshold_is_lower_for_defenders():
    assert scoring_rules.DEFENSIVE_CONTRIBUTION_THRESHOLD["DEF"] == 10
    assert scoring_rules.DEFENSIVE_CONTRIBUTION_THRESHOLD["MID"] == scoring_rules.DEFENSIVE_CONTRIBUTION_THRESHOLD["FWD"] == 12


def test_negative_point_values_are_actually_negative():
    assert scoring_rules.PENALTY_MISS_POINTS < 0
    assert scoring_rules.OWN_GOAL_POINTS < 0
    assert scoring_rules.YELLOW_CARD_POINTS < 0
    assert scoring_rules.RED_CARD_POINTS < scoring_rules.YELLOW_CARD_POINTS
    assert scoring_rules.GOALS_CONCEDED_POINTS_PER_2 < 0
