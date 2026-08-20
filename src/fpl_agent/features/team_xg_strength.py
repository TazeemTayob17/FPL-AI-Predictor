# Computes each team's rolling expected-goals-for/against rate from player-level underlying xG stats, carried across season boundaries via each club's stable code. Lower-variance than team_strength.py's actual-goals version over short windows, since xG smooths out finishing luck - used for the decomposed model's Poisson clean-sheet estimate.

from __future__ import annotations

import pandas as pd

from fpl_agent.features.rolling_stats import WINDOWS

TEAM_XG_STAT_COLUMNS = ("team_xg_for", "team_xg_against")
ROLL_COLUMNS = [f"{stat}_roll{window}" for stat in TEAM_XG_STAT_COLUMNS for window in WINDOWS]


# One row per team per gameweek: total expected goals for, summed from that team's players' own individual expected_goals - additive by construction, so this reconstructs the team's real match xG.
def build_team_xg_for(gw_rows: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if "expected_goals" not in gw_rows.columns:
        return pd.DataFrame(columns=["season", "GW", "code", "team_xg_for"])
    with_code = gw_rows.merge(
        teams[["season", "name", "code"]], left_on=["season", "team"], right_on=["season", "name"], how="left"
    )
    grouped = with_code.dropna(subset=["code"]).groupby(["season", "GW", "code"], as_index=False)["expected_goals"].sum()
    return grouped.rename(columns={"expected_goals": "team_xg_for"})


# Pairs each team-gameweek with its opponent(s) that gameweek via fixtures, and takes the opponent's xG-for as this team's xG-against - what a defense actually faced is exactly what the attacking side mustered.
def build_team_xg_against(team_xg_for: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    finished = fixtures[fixtures["finished"] == True]  # noqa: E712 - pandas boolean column comparison
    home = finished[["season", "event", "team_h", "team_a"]].rename(columns={"team_h": "team_id", "team_a": "opponent_id", "event": "GW"})
    away = finished[["season", "event", "team_a", "team_h"]].rename(columns={"team_a": "team_id", "team_h": "opponent_id", "event": "GW"})
    pairings = pd.concat([home, away], ignore_index=True)

    pairings = pairings.merge(teams[["season", "id", "code"]], left_on=["season", "team_id"], right_on=["season", "id"], how="left")
    pairings = pairings.merge(
        teams[["season", "id", "code"]].rename(columns={"id": "opponent_id", "code": "opponent_code"}),
        on=["season", "opponent_id"], how="left",
    )
    pairings = pairings.merge(
        team_xg_for.rename(columns={"code": "opponent_code", "team_xg_for": "team_xg_against"}),
        on=["season", "GW", "opponent_code"], how="left",
    )
    return pairings.dropna(subset=["code"]).groupby(["season", "GW", "code"], as_index=False)["team_xg_against"].sum()


# One row per team per gameweek: both xG-for and xG-against, summed across fixtures on a double-gameweek exactly like team_strength.py's actual-goals version.
def build_team_xg_match_results(gw_rows: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    xg_for = build_team_xg_for(gw_rows, teams)
    if xg_for.empty:
        return pd.DataFrame(columns=["season", "GW", "code", *TEAM_XG_STAT_COLUMNS])
    xg_against = build_team_xg_against(xg_for, fixtures, teams)
    return xg_for.merge(xg_against, on=["season", "GW", "code"], how="outer").fillna(0.0)


# Rolling mean of each team's own xG-for/xG-against over its last 3/5/10 matches, chained by club code so form carries into a new season instead of resetting to blank.
def add_team_xg_strength_rolling(match_results: pd.DataFrame) -> pd.DataFrame:
    table = match_results.sort_values(["code", "season", "GW"]).reset_index(drop=True)
    group_cols = [table["code"]]
    for stat in TEAM_XG_STAT_COLUMNS:
        shifted = table.groupby("code", sort=False)[stat].shift(1)
        for window in WINDOWS:
            rolled = shifted.groupby(group_cols).rolling(window, min_periods=1).mean()
            table[f"{stat}_roll{window}"] = rolled.reset_index(level=0, drop=True)
    return table


# Builds every team's rolling xG-based attacking/defensive form, spanning every season as one continuous per-club sequence. Returns an empty (but correctly-columned) frame if expected_goals isn't present in gw_rows (older seasons before FPL exposed it).
def build_team_xg_strength_table(gw_rows: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    match_results = build_team_xg_match_results(gw_rows, fixtures, teams)
    if match_results.empty:
        return pd.DataFrame(columns=["code", "season", "GW", *TEAM_XG_STAT_COLUMNS, *ROLL_COLUMNS])
    return add_team_xg_strength_rolling(match_results)


# Each team's most recent rolling xG form regardless of season - the fallback used before that team has played a match in the season being predicted.
def latest_team_xg_strength(team_xg_strength_table: pd.DataFrame) -> pd.DataFrame:
    if team_xg_strength_table.empty:
        return pd.DataFrame(columns=["code", *ROLL_COLUMNS])
    ordered = team_xg_strength_table.sort_values(["code", "season", "GW"])
    return ordered.groupby("code", as_index=False).tail(1)[["code", *ROLL_COLUMNS]]
