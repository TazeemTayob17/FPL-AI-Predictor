"""Builds per-team fixture-difficulty, opponent-strength, and DGW/BGW features from an FPL fixtures table."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gameweek_fixture_counts(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Counts each team's fixtures per gameweek - the basis for DGW (2+) and BGW (0) detection."""
    home = fixtures[["event", "team_h"]].rename(columns={"team_h": "team"})
    away = fixtures[["event", "team_a"]].rename(columns={"team_a": "team"})
    counts = pd.concat([home, away], ignore_index=True).groupby(["event", "team"], as_index=False).size()
    return counts.rename(columns={"size": "fixture_count"})


def team_fixture_difficulty(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Builds one row per team per fixture: that team's difficulty rating and the opponent's attack/defence strength."""
    home = fixtures[["event", "team_h", "team_a", "team_h_difficulty"]].rename(
        columns={"team_h": "team", "team_a": "opponent_team", "team_h_difficulty": "difficulty"}
    )
    home["was_home"] = True
    away = fixtures[["event", "team_a", "team_h", "team_a_difficulty"]].rename(
        columns={"team_a": "team", "team_h": "opponent_team", "team_a_difficulty": "difficulty"}
    )
    away["was_home"] = False
    combined = pd.concat([home, away], ignore_index=True)

    strength_cols = ["id", "strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"]
    combined = combined.merge(
        teams[strength_cols].rename(columns={"id": "opponent_team"}), on="opponent_team", how="left"
    )
    combined["opponent_attack_strength"] = np.where(
        combined["was_home"], combined["strength_attack_away"], combined["strength_attack_home"]
    )
    combined["opponent_defence_strength"] = np.where(
        combined["was_home"], combined["strength_defence_away"], combined["strength_defence_home"]
    )
    return combined.drop(columns=["strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"])
