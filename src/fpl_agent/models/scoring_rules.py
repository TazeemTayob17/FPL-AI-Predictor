# Real FPL point-scoring constants (2025-26/2026-27 rules), used to recombine the decomposed model's component predictions into a single points estimate. No scoring rules existed anywhere else in this repo before this file.

from __future__ import annotations

GOAL_POINTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
SAVE_POINTS_PER_3 = 1
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_POINTS = -2
OWN_GOAL_POINTS = -2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
GOALS_CONCEDED_POINTS_PER_2 = -1  # GKP/DEF only

APPEARANCE_POINTS_ANY_MINUTES = 1
APPEARANCE_POINTS_SIXTY_PLUS_MINUTES = 1  # additional, on top of the "any minutes" point -> 2 total

# 2025-26 defensive-contribution rule: 2 pts for defenders hitting 10+ CBIT actions in a match, 12+ combined actions for mid/fwd. Goalkeepers aren't eligible.
DEFENSIVE_CONTRIBUTION_POINTS = 2
DEFENSIVE_CONTRIBUTION_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}

GOALS_CONCEDED_ELIGIBLE_POSITIONS = ("GKP", "DEF")
CLEAN_SHEET_ELIGIBLE_POSITIONS = ("GKP", "DEF", "MID")
