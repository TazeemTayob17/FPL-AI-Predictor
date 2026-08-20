# Recombines the decomposed model's component predictions (appearance probabilities, per-90 rates, defensive-contribution probability, team clean-sheet probability) into a single predicted_points estimate, using the real FPL scoring rules in scoring_rules.py.

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_agent.models.component_models import predict_component_models
from fpl_agent.models.scoring_rules import (
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    DEFENSIVE_CONTRIBUTION_POINTS,
    GOAL_POINTS,
    GOALS_CONCEDED_ELIGIBLE_POSITIONS,
    GOALS_CONCEDED_POINTS_PER_2,
    OWN_GOAL_POINTS,
    PENALTY_MISS_POINTS,
    PENALTY_SAVE_POINTS,
    RED_CARD_POINTS,
    SAVE_POINTS_PER_3,
    YELLOW_CARD_POINTS,
)

# Rare events read directly from rolling_stats.py's leakage-safe rolling features rather than a trained model - see component_models.py's module docstring for why.
RARE_EVENT_ROLL_COLUMNS = {
    "own_goals_roll10": OWN_GOAL_POINTS,
    "penalties_missed_roll10": PENALTY_MISS_POINTS,
    "yellow_cards_roll10": YELLOW_CARD_POINTS,
    "red_cards_roll10": RED_CARD_POINTS,
}
PENALTY_SAVED_ROLL_COLUMN = "penalties_saved_roll10"

LEAGUE_AVERAGE_GOALS_CONCEDED_PRIOR = 1.3  # rough league-average goals-per-game fallback when no team xG/goals history is available at all


# Poisson lambda for a team's expected goals conceded this fixture: the team's own rolling xG-against (falling back to actual-goals-based team_strength.py's rolling if xG is unavailable), scaled by the upcoming opponent's attacking strength relative to the rest of the fixture list. Not a full Dixon-Coles bivariate model - a deliberately simpler, cheaper adjustment, since the added complexity isn't justified before seeing whether this already beats the current single-target model.
def compute_expected_goals_conceded(features: pd.DataFrame) -> pd.Series:
    xga = features["team_xg_against_roll5"].copy() if "team_xg_against_roll5" in features.columns else pd.Series(np.nan, index=features.index)
    if "team_goals_conceded_roll5" in features.columns:
        xga = xga.fillna(features["team_goals_conceded_roll5"])
    xga = xga.fillna(LEAGUE_AVERAGE_GOALS_CONCEDED_PRIOR)

    if "opponent_attack_strength" in features.columns and features["opponent_attack_strength"].notna().any():
        mean_strength = features["opponent_attack_strength"].mean()
        relative = (features["opponent_attack_strength"] / mean_strength).fillna(1.0) if mean_strength else 1.0
        xga = xga * relative

    return xga.clip(lower=0.1)


REQUIRED_RECOMBINE_COLUMNS = [
    "p_any_minutes", "p_sixty_plus_minutes", "minutes_when_played",
    "goals_per90", "assists_per90", "saves_per90", "bonus_per90", "p_defensive_threshold",
    "team_expected_goals_conceded", *RARE_EVENT_ROLL_COLUMNS, PENALTY_SAVED_ROLL_COLUMN,
]


def _ensure_recombine_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in REQUIRED_RECOMBINE_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
        else:
            result[column] = result[column].fillna(0.0)
    return result


# Applies the real FPL scoring formula to a features table already carrying every component prediction (see REQUIRED_RECOMBINE_COLUMNS) - the decomposed model's equivalent of a trained booster's .predict().
def recombine_points(df: pd.DataFrame) -> pd.Series:
    df = _ensure_recombine_columns(df)
    expected_minutes_fraction = df["p_any_minutes"] * df["minutes_when_played"] / 90

    appearance_points = df["p_any_minutes"] + df["p_sixty_plus_minutes"]  # 1 pt for any minutes, +1 more for 60+

    goal_points = df["position"].map(GOAL_POINTS).fillna(0)
    clean_sheet_points = df["position"].map(CLEAN_SHEET_POINTS).fillna(0)
    p_clean_sheet = np.exp(-df["team_expected_goals_conceded"])

    points = (
        appearance_points
        + expected_minutes_fraction * df["goals_per90"] * goal_points
        + expected_minutes_fraction * df["assists_per90"] * ASSIST_POINTS
        + expected_minutes_fraction * (df["saves_per90"] / 3) * SAVE_POINTS_PER_3
        + p_clean_sheet * df["p_sixty_plus_minutes"] * clean_sheet_points
        + expected_minutes_fraction * df["bonus_per90"]
        + df["p_any_minutes"] * df["p_defensive_threshold"] * DEFENSIVE_CONTRIBUTION_POINTS
        + expected_minutes_fraction * df[PENALTY_SAVED_ROLL_COLUMN] * PENALTY_SAVE_POINTS
    )
    for column, point_value in RARE_EVENT_ROLL_COLUMNS.items():
        points = points + expected_minutes_fraction * df[column] * point_value  # point_value is already negative

    goals_conceded_penalty = pd.Series(0.0, index=df.index)
    eligible = df["position"].isin(GOALS_CONCEDED_ELIGIBLE_POSITIONS)
    goals_conceded_penalty.loc[eligible] = (
        0.5 * df.loc[eligible, "team_expected_goals_conceded"] * expected_minutes_fraction[eligible] * GOALS_CONCEDED_POINTS_PER_2
    )
    return points + goals_conceded_penalty  # GOALS_CONCEDED_POINTS_PER_2 is already negative


# End-to-end: runs every component model over `features`, computes team clean-sheet context, and recombines into predicted_points - the decomposed model's equivalent of predict.py's predict_with_trained_models.
def predict_decomposed_points(features: pd.DataFrame, models: dict, columns: list[str]) -> pd.DataFrame:
    result = predict_component_models(features, models, columns)
    result["team_expected_goals_conceded"] = compute_expected_goals_conceded(result)
    result["predicted_points"] = recombine_points(result)
    return result
