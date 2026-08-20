# Assembles model-ready feature tables: rolling form + fixture context, for historical training and live prediction.

from __future__ import annotations

import pandas as pd

from fpl_agent.features.fixtures_features import team_fixture_difficulty
from fpl_agent.features.rolling_stats import ROLLING_STAT_COLUMNS, WINDOWS, add_rolling_features
from fpl_agent.features.team_strength import ROLL_COLUMNS as TEAM_STRENGTH_ROLL_COLUMNS
from fpl_agent.features.team_strength import build_team_strength_table, latest_team_strength
from fpl_agent.features.team_xg_strength import ROLL_COLUMNS as TEAM_XG_STRENGTH_ROLL_COLUMNS
from fpl_agent.features.team_xg_strength import build_team_xg_strength_table

ADDITIVE_GW_STATS = (
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves",
    "bonus", "bps", "starts", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution", "influence", "creativity", "threat", "ict_index",
)

POSITION_ALIASES = {"GK": "GKP", "GKP": "GKP", "AM": "MID", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

# The trained model always expects these columns to exist, even as all-NaN; guaranteed below.
EXPECTED_ROLL_COLUMNS = [f"{stat}_roll{window}" for stat in ROLLING_STAT_COLUMNS for window in WINDOWS]


# Sums additive per-fixture stats into one row per player per gameweek, correctly handling double-gameweeks.
def collapse_to_gameweek(gw_rows: pd.DataFrame) -> pd.DataFrame:
    present_additive = [c for c in ADDITIVE_GW_STATS if c in gw_rows.columns]
    agg = {c: "sum" for c in present_additive}
    agg.update({c: "first" for c in ("name", "position", "team") if c in gw_rows.columns})
    agg.update({c: "last" for c in ("value", "selected") if c in gw_rows.columns})
    collapsed = gw_rows.groupby(["season", "element", "GW"], as_index=False).agg(agg)
    if "position" in collapsed.columns:
        collapsed["position"] = collapsed["position"].map(POSITION_ALIASES).fillna(collapsed["position"])
    return collapsed


# One row per team per gameweek: mean fixture difficulty, opponent strength, DGW fixture count, and (if given) that team's own rolling defensive/attacking form (actual-goals-based and/or xG-based).
def team_gameweek_context(
    fixtures: pd.DataFrame, teams: pd.DataFrame,
    team_strength: pd.DataFrame | None = None, team_xg_strength: pd.DataFrame | None = None,
) -> pd.DataFrame:
    difficulty = team_fixture_difficulty(fixtures, teams)
    context = difficulty.groupby(["event", "team"], as_index=False).agg(
        difficulty=("difficulty", "mean"),
        opponent_attack_strength=("opponent_attack_strength", "mean"),
        opponent_defence_strength=("opponent_defence_strength", "mean"),
        fixture_count=("difficulty", "size"),
    )
    context = context.rename(columns={"team": "team_id", "event": "GW"})
    if team_strength is not None:
        context = context.merge(team_strength, on=["team_id", "GW"], how="left")
    if team_xg_strength is not None:
        context = context.merge(team_xg_strength, on=["team_id", "GW"], how="left")
    return context


# Builds one model-ready row per player per gameweek per season, with rolling features and fixture context.
def build_training_table(gw_rows: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    team_strength_table = build_team_strength_table(fixtures, teams)
    team_xg_strength_table = build_team_xg_strength_table(gw_rows, fixtures, teams)
    frames = []
    for season in sorted(gw_rows["season"].unique()):
        season_gw = collapse_to_gameweek(gw_rows[gw_rows["season"] == season])
        season_teams = teams[teams["season"] == season]
        team_ids = season_teams.set_index("name")["id"]
        season_gw = season_gw.assign(team_id=season_gw["team"].map(team_ids))

        season_gw = add_rolling_features(season_gw, group_keys=["season", "element"])

        season_team_strength = (
            team_strength_table[team_strength_table["season"] == season]
            .merge(season_teams[["code", "id"]], on="code", how="left")
            .rename(columns={"id": "team_id"})[["team_id", "GW", *TEAM_STRENGTH_ROLL_COLUMNS]]
        )
        season_team_xg_strength = None
        if not team_xg_strength_table.empty:
            season_team_xg_strength = (
                team_xg_strength_table[team_xg_strength_table["season"] == season]
                .merge(season_teams[["code", "id"]], on="code", how="left")
                .rename(columns={"id": "team_id"})[["team_id", "GW", *TEAM_XG_STRENGTH_ROLL_COLUMNS]]
            )
        context = team_gameweek_context(fixtures[fixtures["season"] == season], season_teams, season_team_strength, season_team_xg_strength)
        season_gw = season_gw.merge(context, on=["GW", "team_id"], how="left")
        frames.append(season_gw)
    return pd.concat(frames, ignore_index=True)


# Builds one row per current player: their latest rolling form plus the target gameweek's fixture context; historical_fixtures/historical_teams/previous_season_form (if given) carry team and player form over from last season until this season's own data takes over.
def build_current_features(
    player_histories: dict[int, pd.DataFrame], players: pd.DataFrame, fixtures_current: pd.DataFrame,
    teams_current: pd.DataFrame, target_gameweek: int,
    historical_fixtures: pd.DataFrame | None = None, historical_teams: pd.DataFrame | None = None,
    previous_season_form: pd.DataFrame | None = None,
) -> pd.DataFrame:
    per_player_rows = []
    players_with_current_history = set()
    for player_id, history in player_histories.items():
        if history.empty:
            continue
        players_with_current_history.add(player_id)
        collapsed = collapse_to_gameweek(history.assign(season="current", element=player_id))
        rolled = add_rolling_features(collapsed, group_keys=["season", "element"])
        per_player_rows.append(rolled.iloc[[-1]])

    if not per_player_rows:
        form_table = pd.DataFrame(columns=["element"])
    else:
        form_table = pd.concat(per_player_rows, ignore_index=True)

    context = team_gameweek_context(fixtures_current, teams_current)
    target_context = context[context["GW"] == target_gameweek].drop(columns="GW")

    if historical_fixtures is not None and historical_teams is not None:
        combined_fixtures = pd.concat([historical_fixtures, fixtures_current], ignore_index=True)
        combined_teams = pd.concat([historical_teams, teams_current], ignore_index=True)
        team_strength_now = latest_team_strength(build_team_strength_table(combined_fixtures, combined_teams))
        team_strength_now = team_strength_now.merge(
            teams_current[["code", "id"]], on="code", how="left"
        ).rename(columns={"id": "team_id"})[["team_id", *TEAM_STRENGTH_ROLL_COLUMNS]]
        target_context = target_context.merge(team_strength_now, on="team_id", how="left")

    result = players.merge(form_table, left_on="player_id", right_on="element", how="left")
    result = result.merge(target_context, on="team_id", how="left")
    result["fixture_count"] = result["fixture_count"].fillna(0).astype(int)

    if previous_season_form is not None and not previous_season_form.empty:
        needs_fallback = ~result["player_id"].isin(players_with_current_history)
        fallback = result.loc[needs_fallback, ["player_id"]].merge(previous_season_form, on="player_id", how="left")
        for column in EXPECTED_ROLL_COLUMNS:
            if column in fallback.columns:
                result.loc[needs_fallback, column] = fallback[column].values

    # "value" has a live equivalent (now_cost_million); "selected" doesn't, so it's left missing instead of guessed.
    if "value" not in result.columns:
        result["value"] = float("nan")
    result["value"] = result["value"].fillna(result["now_cost_million"] * 10)

    # float("nan"), not pd.NA: LightGBM requires numeric dtypes even for all-missing columns; pd.NA produces object dtype.
    for column in [*EXPECTED_ROLL_COLUMNS, *TEAM_STRENGTH_ROLL_COLUMNS, "selected"]:
        if column not in result.columns:
            result[column] = float("nan")
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result
