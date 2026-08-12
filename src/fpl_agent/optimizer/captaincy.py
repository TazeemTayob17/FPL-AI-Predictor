"""Picks captain and vice-captain: argmax of predicted_points over the chosen starting XI, no ML yet."""

from __future__ import annotations

import pandas as pd


def choose_captaincy(starting_xi: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns (captain, vice_captain) as the top two predicted_points rows in the starting XI."""
    ranked = starting_xi.sort_values("predicted_points", ascending=False)
    return ranked.iloc[0], ranked.iloc[1]
