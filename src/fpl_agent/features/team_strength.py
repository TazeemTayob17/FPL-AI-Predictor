# Computes each team's rolling goals-scored/conceded/clean-sheet rate from real match results, carried across season boundaries via each club's stable code (not the season-scoped numeric team id, which gets reassigned every season).

from __future__ import annotations

import pandas as pd

from fpl_agent.features.rolling_stats import WINDOWS

TEAM_STAT_COLUMNS = ("team_goals_scored", "team_goals_conceded", "team_clean_sheet")
ROLL_COLUMNS = [f"{stat}_roll{window}" for stat in TEAM_STAT_COLUMNS for window in WINDOWS]


# One row per team per gameweek: goals scored/conceded and clean sheets, summed across fixtures on a double-gameweek exactly like player-level stats already are - a team with 2 fixtures in one GW must not produce 2 rows, which would silently double-count every one of its players in that gameweek downstream.
def build_team_match_results(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    finished = fixtures[fixtures["finished"] == True]  # noqa: E712 - pandas boolean column comparison
    home = finished[["season", "event", "team_h", "team_h_score", "team_a_score"]].rename(
        columns={"team_h": "team_id", "team_h_score": "team_goals_scored", "team_a_score": "team_goals_conceded"}
    )
    away = finished[["season", "event", "team_a", "team_a_score", "team_h_score"]].rename(
        columns={"team_a": "team_id", "team_a_score": "team_goals_scored", "team_h_score": "team_goals_conceded"}
    )
    combined = pd.concat([home, away], ignore_index=True)
    combined = combined.merge(teams[["season", "id", "code"]], left_on=["season", "team_id"], right_on=["season", "id"], how="left")
    combined["team_clean_sheet"] = (combined["team_goals_conceded"] == 0).astype(int)
    combined = combined.rename(columns={"event": "GW"})
    return combined.groupby(["season", "GW", "code"], as_index=False)[list(TEAM_STAT_COLUMNS)].sum()


# Rolling mean of each team's own scoring/conceding/clean-sheet rate over its last 3/5/10 matches, chained by club code so form carries into a new season instead of resetting to blank.
def add_team_strength_rolling(match_results: pd.DataFrame) -> pd.DataFrame:
    table = match_results.sort_values(["code", "season", "GW"]).reset_index(drop=True)
    group_cols = [table["code"]]
    for stat in TEAM_STAT_COLUMNS:
        shifted = table.groupby("code", sort=False)[stat].shift(1)
        for window in WINDOWS:
            rolled = shifted.groupby(group_cols).rolling(window, min_periods=1).mean()
            table[f"{stat}_roll{window}"] = rolled.reset_index(level=0, drop=True)
    return table


# Builds every team's rolling defensive/attacking form from real match results, spanning every season in `fixtures` as one continuous per-club sequence.
def build_team_strength_table(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    return add_team_strength_rolling(build_team_match_results(fixtures, teams))


# Each team's most recent rolling form regardless of season - the fallback used before that team has played a match in the season being predicted.
def latest_team_strength(team_strength_table: pd.DataFrame) -> pd.DataFrame:
    ordered = team_strength_table.sort_values(["code", "season", "GW"])
    return ordered.groupby("code", as_index=False).tail(1)[["code", *ROLL_COLUMNS]]
