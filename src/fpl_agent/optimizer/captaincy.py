"""Picks captain and vice-captain: argmax of predicted_points over the chosen starting XI, no ML yet."""

from __future__ import annotations

import pandas as pd


def choose_captaincy(starting_xi: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns (captain, vice_captain) as the top two predicted_points rows in the starting XI."""
    ranked = starting_xi.sort_values("predicted_points", ascending=False)
    return ranked.iloc[0], ranked.iloc[1]


def compute_risk_score(chance_of_playing_next_round: pd.Series, minutes_volatility: pd.Series) -> pd.Series:
    """Blends injury doubt and minutes volatility into a 0-1 risk score, higher meaning riskier to captain."""
    availability_risk = 1 - (chance_of_playing_next_round.fillna(100) / 100)
    normalized_volatility = (minutes_volatility.fillna(0) / 45).clip(upper=1)
    return (0.7 * availability_risk + 0.3 * normalized_volatility).clip(0, 1)


def captaincy_candidates(starting_xi: pd.DataFrame) -> pd.DataFrame:
    """Ranks the starting XI by predicted_points, attaching a risk score wherever the underlying data supports it."""
    ranked = starting_xi.sort_values("predicted_points", ascending=False).copy()
    if {"chance_of_playing_next_round", "minutes_volatility"}.issubset(ranked.columns):
        ranked["risk_score"] = compute_risk_score(ranked["chance_of_playing_next_round"], ranked["minutes_volatility"])
    return ranked
