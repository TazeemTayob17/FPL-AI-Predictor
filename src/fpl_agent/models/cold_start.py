"""Pre-season point predictor: prior-season points-per-90 blended with regression-to-mean and a promoted-club prior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from fpl_agent.features.fixtures_features import team_fixture_difficulty
from fpl_agent.models.naive_predictor import previous_season_label
from fpl_agent.storage.repository import PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

SHRINKAGE_MINUTES = 450.0
TRANSFER_DISCOUNT = 0.85
WEAK_TEAM_TIER_SIZE = 5
DIFFICULTY_SWING_PER_STEP = 0.05
OPENING_DIFFICULTY_WINDOW_GWS = 3


def load_prior_season_player_stats(season_label: str) -> pd.DataFrame:
    """Aggregates last season's per-gameweek rows into one row per player: points, minutes, appearances, team."""
    prev_season = previous_season_label(season_label)
    gw_path = PROCESSED_DIR / "all_seasons_gw.parquet"
    columns = ["player_name", "team", "total_points", "minutes", "appearances"]
    if not gw_path.exists():
        return pd.DataFrame(columns=columns)
    gw_rows = pd.read_parquet(gw_path)
    season_rows = gw_rows[gw_rows["season"] == prev_season]
    if season_rows.empty:
        return pd.DataFrame(columns=columns)
    grouped = season_rows.groupby("name", as_index=False).agg(
        team=("team", "last"),
        total_points=("total_points", "sum"),
        minutes=("minutes", "sum"),
        appearances=("minutes", lambda m: int((m > 0).sum())),
    )
    grouped = grouped.rename(columns={"name": "player_name"})
    grouped["player_name"] = grouped["player_name"].str.strip().str.lower()
    return grouped


def position_average_pp90(prior_stats: pd.DataFrame, current_positions: pd.DataFrame) -> pd.DataFrame:
    """Computes the league-average points-per-90 for each position, from last season's regular starters."""
    with_position = prior_stats.merge(current_positions, on="player_name", how="inner")
    reliable = with_position[with_position["minutes"] >= SHRINKAGE_MINUTES]
    rate = reliable["total_points"] / (reliable["minutes"] / 90)
    return rate.groupby(reliable["position"]).mean().rename("position_avg_pp90").reset_index()


def blended_points_per_90(points: pd.Series, minutes: pd.Series, position_avg_pp90: pd.Series) -> pd.Series:
    """Blends a player's actual points-per-90 with their position's average, weighted by minutes played."""
    nineties = minutes / 90
    shrinkage_nineties = SHRINKAGE_MINUTES / 90
    return (points + shrinkage_nineties * position_avg_pp90) / (nineties + shrinkage_nineties)


def final_standings(fixtures: pd.DataFrame) -> pd.Series:
    """Computes each team's final league points from a season's full fixture results (3-1-0 scoring)."""
    home = fixtures[["team_h", "team_h_score", "team_a_score"]].rename(
        columns={"team_h": "team", "team_h_score": "gf", "team_a_score": "ga"}
    )
    away = fixtures[["team_a", "team_a_score", "team_h_score"]].rename(
        columns={"team_a": "team", "team_a_score": "gf", "team_h_score": "ga"}
    )
    results = pd.concat([home, away], ignore_index=True).dropna()
    results["points"] = (results["gf"] > results["ga"]).astype(int) * 3 + (results["gf"] == results["ga"]).astype(int)
    return results.groupby("team")["points"].sum().sort_values(ascending=False)


def weak_team_prior_points_per_gw(
    prior_stats: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, current_positions: pd.DataFrame
) -> pd.DataFrame:
    """Computes each position's average points-per-appearance among last season's bottom teams, for promoted clubs."""
    standings = final_standings(fixtures)
    weak_team_ids = standings.tail(WEAK_TEAM_TIER_SIZE).index
    weak_team_names = set(teams[teams["id"].isin(weak_team_ids)]["name"])
    weak_players = prior_stats[prior_stats["team"].isin(weak_team_names)].merge(
        current_positions, on="player_name", how="inner"
    )
    per_gw = weak_players["total_points"] / weak_players["appearances"].clip(lower=1)
    return per_gw.groupby(weak_players["position"]).mean().rename("weak_team_prior_pp_gw").reset_index()


def default_weak_team_prior(season_label: str, current_positions: pd.DataFrame, prior_stats: pd.DataFrame) -> pd.DataFrame:
    """Loads the previous season's completed fixtures/teams and computes the weak-team-tier prior for players with no prior_stats match.

    Returns an empty frame (matching predict_cold_start_points's old always-empty default) if that historical data
    isn't available yet - e.g. before vaastav_sync has ever run.
    """
    empty = pd.DataFrame(columns=["position", "weak_team_prior_pp_gw"])
    if prior_stats.empty:
        return empty
    fixtures_path = PROCESSED_DIR / "all_seasons_fixtures.parquet"
    teams_path = PROCESSED_DIR / "all_seasons_teams.parquet"
    if not fixtures_path.exists() or not teams_path.exists():
        return empty

    prev_season = previous_season_label(season_label)
    prev_fixtures = pd.read_parquet(fixtures_path).pipe(lambda f: f[f["season"] == prev_season])
    prev_teams = pd.read_parquet(teams_path).pipe(lambda t: t[t["season"] == prev_season])
    if prev_fixtures.empty or prev_teams.empty:
        return empty
    return weak_team_prior_points_per_gw(prior_stats, prev_fixtures, prev_teams, current_positions)


def build_opening_difficulty(
    fixtures_current: pd.DataFrame, teams_current: pd.DataFrame, window_gws: int = OPENING_DIFFICULTY_WINDOW_GWS
) -> pd.DataFrame:
    """Averages each team's fixture-difficulty rating across their first `window_gws` gameweeks of the new season."""
    difficulty = team_fixture_difficulty(fixtures_current, teams_current)
    opening = difficulty[difficulty["event"] <= window_gws]
    if opening.empty:
        return pd.DataFrame(columns=["team_id", "difficulty"])
    return opening.groupby("team", as_index=False)["difficulty"].mean().rename(columns={"team": "team_id"})


def predict_cold_start_points(
    players: pd.DataFrame,
    season_label: str | None = None,
    prior_stats: pd.DataFrame | None = None,
    weak_team_prior: pd.DataFrame | None = None,
    opening_difficulty: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attaches a `predicted_points` column: the pre-season blend of prior form, regression-to-mean, and promoted-club prior."""
    if season_label is None:
        with SETTINGS_PATH.open(encoding="utf-8") as f:
            season_label = yaml.safe_load(f)["season"]["label"]
    if prior_stats is None:
        prior_stats = load_prior_season_player_stats(season_label)
    if opening_difficulty is None:
        opening_difficulty = pd.DataFrame(columns=["team_id", "difficulty"])

    current_positions = players[["web_name_full", "position"]].rename(
        columns={"web_name_full": "player_name"}
    ).assign(player_name=lambda d: d["player_name"].str.strip().str.lower())

    if weak_team_prior is None:
        weak_team_prior = default_weak_team_prior(season_label, current_positions, prior_stats)

    pos_avg = (
        position_average_pp90(prior_stats, current_positions) if not prior_stats.empty
        else pd.DataFrame(columns=["position", "position_avg_pp90"])
    )
    prior_teams = set(prior_stats["team"].unique()) if not prior_stats.empty else set()

    prior_for_merge = prior_stats.rename(
        columns={
            "player_name": "_join_key", "team": "prior_team", "total_points": "prior_total_points",
            "minutes": "prior_minutes", "appearances": "prior_appearances",
        }
    )
    join_key = players["web_name_full"].str.strip().str.lower()
    merged = players.assign(_join_key=join_key).merge(prior_for_merge, on="_join_key", how="left")
    merged = merged.merge(pos_avg, on="position", how="left")
    merged = merged.merge(weak_team_prior, on="position", how="left")
    merged = merged.merge(opening_difficulty[["team_id", "difficulty"]], on="team_id", how="left")

    has_prior_data = merged["prior_total_points"].notna()
    per_appearance_minutes = (merged["prior_minutes"] / merged["prior_appearances"].replace(0, pd.NA)).clip(upper=90)
    blended_pp90 = blended_points_per_90(
        merged["prior_total_points"].fillna(0), merged["prior_minutes"].fillna(0), merged["position_avg_pp90"]
    )
    per_gw_from_history = blended_pp90 * (per_appearance_minutes.fillna(0) / 90)

    changed_clubs = has_prior_data & (merged["prior_team"] != merged["team_name"])
    per_gw_from_history = per_gw_from_history.where(~changed_clubs, per_gw_from_history * TRANSFER_DISCOUNT)

    difficulty_factor = 1 + (3 - merged["difficulty"].fillna(3)) * DIFFICULTY_SWING_PER_STEP
    base_estimate = per_gw_from_history.where(has_prior_data, merged["weak_team_prior_pp_gw"].fillna(0))
    merged["predicted_points"] = base_estimate * difficulty_factor

    return merged.drop(
        columns=[
            "_join_key", "prior_team", "prior_total_points", "prior_minutes", "prior_appearances",
            "position_avg_pp90", "weak_team_prior_pp_gw", "difficulty",
        ]
    )
