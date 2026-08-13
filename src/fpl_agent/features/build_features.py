"""Assembles model-ready feature tables: rolling form + fixture context, for historical training and live prediction."""

from __future__ import annotations

import pandas as pd

from fpl_agent.features.fixtures_features import team_fixture_difficulty
from fpl_agent.features.rolling_stats import add_rolling_features

ADDITIVE_GW_STATS = (
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves",
    "bonus", "bps", "starts", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution", "influence", "creativity", "threat", "ict_index",
)

POSITION_ALIASES = {"GK": "GKP", "GKP": "GKP", "AM": "MID", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def collapse_to_gameweek(gw_rows: pd.DataFrame) -> pd.DataFrame:
    """Sums additive per-fixture stats into one row per player per gameweek, correctly handling double-gameweeks."""
    present_additive = [c for c in ADDITIVE_GW_STATS if c in gw_rows.columns]
    agg = {c: "sum" for c in present_additive}
    agg.update({"name": "first", "position": "first", "team": "first"})
    agg.update({c: "last" for c in ("value", "selected") if c in gw_rows.columns})
    collapsed = gw_rows.groupby(["season", "element", "GW"], as_index=False).agg(agg)
    collapsed["position"] = collapsed["position"].map(POSITION_ALIASES).fillna(collapsed["position"])
    return collapsed


def team_gameweek_context(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per team per gameweek: mean fixture difficulty, opponent strength, and DGW fixture count."""
    difficulty = team_fixture_difficulty(fixtures, teams)
    context = difficulty.groupby(["event", "team"], as_index=False).agg(
        difficulty=("difficulty", "mean"),
        opponent_attack_strength=("opponent_attack_strength", "mean"),
        opponent_defence_strength=("opponent_defence_strength", "mean"),
        fixture_count=("difficulty", "size"),
    )
    return context.rename(columns={"team": "team_id", "event": "GW"})


def build_training_table(gw_rows: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Builds one model-ready row per player per gameweek per season, with rolling features and fixture context."""
    frames = []
    for season in sorted(gw_rows["season"].unique()):
        season_gw = collapse_to_gameweek(gw_rows[gw_rows["season"] == season])
        season_teams = teams[teams["season"] == season]
        team_ids = season_teams.set_index("name")["id"]
        season_gw = season_gw.assign(team_id=season_gw["team"].map(team_ids))

        season_gw = add_rolling_features(season_gw, group_keys=["season", "element"])

        context = team_gameweek_context(fixtures[fixtures["season"] == season], season_teams)
        season_gw = season_gw.merge(context, on=["GW", "team_id"], how="left")
        frames.append(season_gw)
    return pd.concat(frames, ignore_index=True)


def build_current_features(
    player_histories: dict[int, pd.DataFrame], players: pd.DataFrame, fixtures_current: pd.DataFrame,
    teams_current: pd.DataFrame, target_gameweek: int,
) -> pd.DataFrame:
    """Builds one row per current player: their latest rolling form plus the target gameweek's fixture context."""
    per_player_rows = []
    for player_id, history in player_histories.items():
        if history.empty:
            continue
        collapsed = collapse_to_gameweek(history.assign(season="current", element=player_id))
        rolled = add_rolling_features(collapsed, group_keys=["season", "element"])
        per_player_rows.append(rolled.iloc[[-1]])

    if not per_player_rows:
        form_table = pd.DataFrame(columns=["element"])
    else:
        form_table = pd.concat(per_player_rows, ignore_index=True)

    context = team_gameweek_context(fixtures_current, teams_current)
    target_context = context[context["GW"] == target_gameweek].drop(columns="GW")

    result = players.merge(form_table, left_on="player_id", right_on="element", how="left")
    result = result.merge(target_context, on="team_id", how="left")
    result["fixture_count"] = result["fixture_count"].fillna(0).astype(int)
    return result
