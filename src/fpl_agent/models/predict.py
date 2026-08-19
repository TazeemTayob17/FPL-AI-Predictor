# Routes point prediction to cold-start priors pre-season, or the trained LightGBM models once enough data exists.

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

from fpl_agent.features.build_features import EXPECTED_ROLL_COLUMNS, build_current_features, collapse_to_gameweek
from fpl_agent.features.rolling_stats import add_minutes_volatility
from fpl_agent.ingestion import fpl_api
from fpl_agent.models.cold_start import build_opening_difficulty, predict_cold_start_points
from fpl_agent.models.live_adjustments import apply_live_adjustments
from fpl_agent.storage.repository import PROCESSED_DIR, infer_current_gameweek

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"
PLAYER_HISTORY_FRESHNESS_HOURS = 20.0  # reuse a same-day fetch instead of re-pulling all ~600 players on every manual run


# Counts how many of this season's gameweeks have finished, to decide whether cold-start or the trained model applies.
def completed_gameweeks_this_season(bootstrap: dict) -> int:
    return sum(1 for event in bootstrap.get("events", []) if event.get("finished"))


# True until enough current-season gameweeks exist for the trained model's rolling features to be meaningful.
def should_use_cold_start(completed_gameweeks: int, threshold: int) -> bool:
    return completed_gameweeks < threshold


# Loads each position's trained model and its expected feature list from the registry.
def load_position_models(registry_path: Path = REGISTRY_PATH) -> dict[str, tuple[lgb.Booster, list[str]]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    models = {}
    for position, info in registry.items():
        model_path = registry_path.parent / "artifacts" / info["model_file"]
        models[position] = (lgb.Booster(model_file=str(model_path)), info["features"])
    return models


# Runs each position's model over its rows and attaches a `predicted_points` column.
def predict_with_trained_models(features: pd.DataFrame, models: dict[str, tuple[lgb.Booster, list[str]]]) -> pd.DataFrame:
    features = features.copy()
    features["predicted_points"] = 0.0
    for position, (model, columns) in models.items():
        mask = features["position"] == position
        if mask.any():
            features.loc[mask, "predicted_points"] = model.predict(features.loc[mask, columns])
    return features


# Returns (players_with_predicted_points, used_cold_start), routing per the configured cold-start GW threshold.
def predict_points(
    players: pd.DataFrame,
    bootstrap: dict,
    player_histories: dict[int, pd.DataFrame] | None = None,
    fixtures_current: pd.DataFrame | None = None,
    teams_current: pd.DataFrame | None = None,
    target_gameweek: int | None = None,
    registry_path: Path = REGISTRY_PATH,
    historical_fixtures: pd.DataFrame | None = None,
    historical_teams: pd.DataFrame | None = None,
    previous_season_form: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, bool]:
    with SETTINGS_PATH.open(encoding="utf-8") as f:
        threshold = yaml.safe_load(f)["model"]["cold_start_gw_threshold"]
    completed = completed_gameweeks_this_season(bootstrap)

    if should_use_cold_start(completed, threshold) or not registry_path.exists():
        opening_difficulty = None
        if fixtures_current is not None and teams_current is not None:
            opening_difficulty = build_opening_difficulty(fixtures_current, teams_current)
        return predict_cold_start_points(players, opening_difficulty=opening_difficulty), True

    if player_histories is None or fixtures_current is None or teams_current is None or target_gameweek is None:
        raise ValueError("player_histories, fixtures_current, teams_current, and target_gameweek are required past cold-start.")

    models = load_position_models(registry_path)
    features = build_current_features(
        player_histories, players, fixtures_current, teams_current, target_gameweek,
        historical_fixtures=historical_fixtures, historical_teams=historical_teams,
        previous_season_form=previous_season_form,
    )
    return predict_with_trained_models(features, models), False


# Fetches each current player's this-season match history, reusing same-day cached pulls to avoid ~600 calls per run.
def fetch_player_histories(players: pd.DataFrame, max_age_hours: float = PLAYER_HISTORY_FRESHNESS_HOURS) -> dict[int, pd.DataFrame]:
    histories = {}
    for player_id in players["player_id"]:
        summary = fpl_api.get_element_summary(int(player_id), max_age_hours=max_age_hours)
        history = pd.DataFrame(summary.get("history", []))
        if not history.empty:
            history = history.rename(columns={"round": "GW"})
        histories[int(player_id)] = history
    return histories


# Loads the current season's fixtures/teams if available, for opening-fixture-difficulty or live rolling context.
def load_fixtures_and_teams_current(bootstrap: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    fixtures_path = PROCESSED_DIR / "fixtures_current.parquet"
    fixtures_current = pd.read_parquet(fixtures_path) if fixtures_path.exists() else None
    teams_current = pd.DataFrame(bootstrap["teams"]) if "teams" in bootstrap else None
    return fixtures_current, teams_current


# Loads every past season's fixtures/teams, so this season's team defensive/attacking form can carry over instead of starting blank.
def load_historical_fixtures_and_teams() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    fixtures_path = PROCESSED_DIR / "all_seasons_fixtures.parquet"
    teams_path = PROCESSED_DIR / "all_seasons_teams.parquet"
    if not fixtures_path.exists() or not teams_path.exists():
        return None, None
    return pd.read_parquet(fixtures_path), pd.read_parquet(teams_path)


# Each current player's end-of-last-season rolling form, matched by full name - the fallback used until they've played their first game of the current season.
def load_previous_season_form(players: pd.DataFrame, training_table_path: Path = PROCESSED_DIR / "training_table.parquet") -> pd.DataFrame:
    if not training_table_path.exists():
        return pd.DataFrame(columns=["player_id"])
    table = pd.read_parquet(training_table_path, columns=["season", "GW", "name", *EXPECTED_ROLL_COLUMNS])
    last_season = table["season"].max()
    last_rows = table[table["season"] == last_season].sort_values("GW").groupby("name", as_index=False).tail(1)
    matched = players[["player_id", "web_name_full"]].merge(last_rows, left_on="web_name_full", right_on="name", how="inner")
    return matched[["player_id", *EXPECTED_ROLL_COLUMNS]]


# Each current player's rolling minutes volatility computed from this season's real history so far - empty pre-season, since there are no games to compute it from yet.
def compute_current_minutes_volatility(player_histories: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for player_id, history in player_histories.items():
        if history.empty:
            continue
        collapsed = collapse_to_gameweek(history.assign(season="current", element=player_id))
        volatility = add_minutes_volatility(collapsed, group_keys=["season", "element"])
        rows.append(volatility.iloc[[-1]][["element", "minutes_volatility"]])
    if not rows:
        return pd.DataFrame(columns=["player_id", "minutes_volatility"])
    return pd.concat(rows, ignore_index=True).rename(columns={"element": "player_id"})


# Each current player's end-of-last-season minutes volatility, matched by full name - the GW1 fallback until this season has real history.
def load_previous_season_minutes_volatility(players: pd.DataFrame, training_table_path: Path = PROCESSED_DIR / "training_table.parquet") -> pd.DataFrame:
    if not training_table_path.exists() or "web_name_full" not in players.columns:
        return pd.DataFrame(columns=["player_id", "minutes_volatility"])
    table = pd.read_parquet(training_table_path, columns=["season", "GW", "name", "element", "minutes"])
    last_season = table["season"].max()
    volatility = add_minutes_volatility(table[table["season"] == last_season], group_keys=["season", "element"])
    last_rows = volatility.sort_values("GW").groupby("name", as_index=False).tail(1)
    matched = players[["player_id", "web_name_full"]].merge(last_rows, left_on="web_name_full", right_on="name", how="inner")
    return matched[["player_id", "minutes_volatility"]]


# Attaches minutes_volatility to predictions - current-season if the player has played this season, else last season's - for the squad optimizer's bench-reliability weighting.
def attach_minutes_volatility(predictions: pd.DataFrame, player_histories: dict[int, pd.DataFrame]) -> pd.DataFrame:
    current = compute_current_minutes_volatility(player_histories)
    result = predictions.merge(current, on="player_id", how="left")
    missing = result["minutes_volatility"].isna()
    if missing.any():
        previous = load_previous_season_minutes_volatility(predictions)
        if not previous.empty:
            fallback = result.loc[missing, ["player_id"]].merge(previous, on="player_id", how="left")
            result.loc[missing, "minutes_volatility"] = fallback["minutes_volatility"].values
    result["minutes_volatility"] = result["minutes_volatility"].fillna(0.0)
    return result


# Attaches `horizon_points`, summing the routed predictor's output across horizon_gws gameweeks.
def predict_horizon_points(players: pd.DataFrame, bootstrap: dict, horizon_gws: int) -> tuple[pd.DataFrame, bool]:
    with SETTINGS_PATH.open(encoding="utf-8") as f:
        threshold = int(yaml.safe_load(f)["model"]["cold_start_gw_threshold"])
    completed = completed_gameweeks_this_season(bootstrap)

    if should_use_cold_start(completed, threshold) or not REGISTRY_PATH.exists():
        fixtures_current, teams_current = load_fixtures_and_teams_current(bootstrap)
        predictions, used_cold_start = predict_points(players, bootstrap, fixtures_current=fixtures_current, teams_current=teams_current)
        predictions = predictions.copy()
        predictions["horizon_points"] = predictions["predicted_points"] * horizon_gws
        predictions = apply_live_adjustments(predictions, players)
        return predictions, used_cold_start

    next_gw = infer_current_gameweek(bootstrap)
    if next_gw is None:
        raise ValueError("Cannot determine the current/next gameweek from bootstrap-static (is the season over?).")

    fixtures_current, teams_current = load_fixtures_and_teams_current(bootstrap)
    if fixtures_current is None or teams_current is None:
        raise ValueError("The trained-model path needs fixtures_current.parquet and bootstrap['teams'] - run the refresh first.")
    player_histories = fetch_player_histories(players)
    historical_fixtures, historical_teams = load_historical_fixtures_and_teams()
    previous_form = load_previous_season_form(players) if historical_fixtures is not None else None

    horizon_frames = []
    for gw in range(next_gw, next_gw + horizon_gws):
        gw_predictions, _used_cold_start = predict_points(
            players, bootstrap, player_histories=player_histories, fixtures_current=fixtures_current,
            teams_current=teams_current, target_gameweek=gw,
            historical_fixtures=historical_fixtures, historical_teams=historical_teams,
            previous_season_form=previous_form,
        )
        gw_predictions = gw_predictions.copy()
        gw_predictions.loc[gw_predictions["fixture_count"] == 0, "predicted_points"] = 0.0
        horizon_frames.append(gw_predictions[["player_id", "predicted_points"]])

    horizon_sum = pd.concat(horizon_frames, ignore_index=True).groupby("player_id", as_index=False)["predicted_points"].sum()
    horizon_sum = horizon_sum.rename(columns={"predicted_points": "horizon_points"})

    predictions = players.merge(horizon_sum, on="player_id", how="left")
    predictions["horizon_points"] = predictions["horizon_points"].fillna(0.0)
    predictions["predicted_points"] = predictions["horizon_points"] / horizon_gws
    predictions = attach_minutes_volatility(predictions, player_histories)
    predictions = apply_live_adjustments(predictions, players, current_gw=next_gw)
    return predictions, False
