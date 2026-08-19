# Shared backtest/replay logic: walk-forward accuracy scoring and historical-GW horizon prediction.

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_agent.features.rolling_stats import WINDOWS, add_minutes_volatility
from fpl_agent.models.naive_predictor import previous_season_label
from fpl_agent.models.train import POSITIONS, TARGET_COLUMN, feature_columns, train_position_model
from fpl_agent.optimizer.captaincy import choose_captaincy
from fpl_agent.optimizer.constraints import SquadRules
from fpl_agent.optimizer.season_planner import classify_core_vs_rotational
from fpl_agent.optimizer.squad_optimizer import select_squad, select_squad_and_starting_xi, select_starting_xi
from fpl_agent.optimizer.transfer_optimizer import recommend_transfers
from fpl_agent.storage.repository import PROCESSED_DIR

SQUAD_COLUMNS = ["web_name", "player_id", "position", "team_name", "now_cost_million"]

MIN_EVAL_GW = 6
HORIZON = 5
MIN_SQUAD_FORMATION_GW = 6


# Splits time-ordered rows into an earlier training slice and a small most-recent validation slice for early stopping.
def recency_split(rows: pd.DataFrame, valid_fraction: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = rows.sort_values(["season", "GW"])
    cutoff = max(1, int(len(ordered) * (1 - valid_fraction)))
    return ordered.iloc[:cutoff], ordered.iloc[cutoff:]


# Builds the naive per-gameweek baseline: last season's total points divided evenly across that season's gameweeks.
def naive_prior_points(target_season: str) -> pd.DataFrame:
    prev_season = previous_season_label(target_season)
    gw_rows = pd.read_parquet(PROCESSED_DIR / "all_seasons_gw.parquet")
    prior_season_rows = gw_rows[gw_rows["season"] == prev_season]
    gameweeks_played = prior_season_rows["GW"].max()
    prior = prior_season_rows.groupby("name", as_index=False)["total_points"].sum()
    prior["naive_prediction"] = prior["total_points"] / gameweeks_played
    return prior[["name", "naive_prediction"]]


# For each evaluated gameweek, retrains on strictly-prior data only and predicts that gameweek's actual points.
def walk_forward_backtest(table: pd.DataFrame, target_season: str, min_gw: int = MIN_EVAL_GW) -> pd.DataFrame:
    columns = feature_columns(table)
    naive = naive_prior_points(target_season)
    target_gws = sorted(g for g in table.loc[table["season"] == target_season, "GW"].unique() if g >= min_gw)

    folds = []
    for gw in target_gws:
        train_mask = (table["season"] < target_season) | ((table["season"] == target_season) & (table["GW"] < gw))
        eval_mask = (table["season"] == target_season) & (table["GW"] == gw)
        train_rows = table[train_mask]
        id_cols = [c for c in ("name", "position", "GW", "total_points", "selected") if c not in columns]
        keep_cols = id_cols + columns
        eval_rows = table.loc[eval_mask, keep_cols].copy()
        if eval_rows.empty:
            continue

        eval_rows["model_prediction"] = np.nan
        for position in POSITIONS:
            pos_train_all = train_rows[train_rows["position"] == position].dropna(subset=[TARGET_COLUMN])
            pos_eval_mask = eval_rows["position"] == position
            if len(pos_train_all) < 50 or not pos_eval_mask.any():
                continue
            pos_train, pos_valid = recency_split(pos_train_all)
            model, _ = train_position_model(pos_train, pos_valid, columns)
            eval_rows.loc[pos_eval_mask, "model_prediction"] = model.predict(eval_rows.loc[pos_eval_mask, columns])

        eval_rows = eval_rows.merge(naive, on="name", how="left")
        eval_rows["naive_prediction"] = eval_rows["naive_prediction"].fillna(0)
        folds.append(eval_rows)

    return pd.concat(folds, ignore_index=True)


# Computes MAE, RMSE, and mean per-gameweek Spearman rank correlation for the model vs the naive baseline.
def accuracy_report(predictions: pd.DataFrame) -> dict:
    report = {}
    for label, column in (("model", "model_prediction"), ("naive", "naive_prediction")):
        valid = predictions.dropna(subset=[column, "total_points"])
        errors = valid[column] - valid["total_points"]
        rank_corrs = [
            g[column].corr(g["total_points"], method="spearman") for _, g in valid.groupby("GW") if len(g) > 2
        ]
        report[label] = {
            "mae": float(errors.abs().mean()),
            "rmse": float((errors**2).mean() ** 0.5),
            "mean_rank_correlation": float(np.nanmean(rank_corrs)),
        }
    return report


# Builds the static player pool (name, position, team, cost) as it stood at the season's first evaluated gameweek.
def target_season_player_pool(table: pd.DataFrame, target_season: str, first_gw: int) -> pd.DataFrame:
    season_rows = table[(table["season"] == target_season) & (table["GW"] == first_gw)]
    pool = season_rows[["name", "position", "team", "team_id", "value"]].drop_duplicates("name").copy()
    pool["now_cost_million"] = pool["value"] / 10
    return pool.rename(columns={"team": "team_name", "name": "web_name"})


# Picks a 15-man squad once from that predictor's earliest predictions, to be held all season (no transfers).
def build_fixed_squad(pool: pd.DataFrame, predictions: pd.DataFrame, prediction_column: str, first_gw: int) -> set[str]:
    first_gw_pred = predictions.loc[predictions["GW"] == first_gw, ["name", prediction_column]]
    candidates = pool.merge(first_gw_pred, left_on="web_name", right_on="name", how="inner")
    candidates = candidates.rename(columns={prediction_column: "predicted_points"})
    squad = select_squad(candidates)
    return set(squad["web_name"])


# Sums a fixed squad's actual points for one gameweek, doubling whoever captaincy_column ranks highest that week.
def gameweek_points_with_captain(gw_rows: pd.DataFrame, squad_names: set[str], captaincy_column: str) -> float:
    squad_gw = gw_rows[gw_rows["name"].isin(squad_names)]
    if squad_gw.empty:
        return 0.0
    total = squad_gw["total_points"].sum()
    ranked = squad_gw.dropna(subset=[captaincy_column])
    captain_bonus = squad_gw.loc[ranked[captaincy_column].idxmax(), "total_points"] if not ranked.empty else 0
    return float(total + captain_bonus)


# Simulates a fixed, never-transferred squad under model-, naive-, and ownership-driven captaincy.
def simulate_cumulative_points(predictions: pd.DataFrame, pool: pd.DataFrame) -> dict[str, float]:
    first_gw = predictions["GW"].min()
    model_squad = build_fixed_squad(pool, predictions, "model_prediction", first_gw)
    naive_squad = build_fixed_squad(pool, predictions, "naive_prediction", first_gw)

    totals = {"model": 0.0, "naive": 0.0, "most_owned": 0.0}
    for _, gw_rows in predictions.groupby("GW"):
        totals["model"] += gameweek_points_with_captain(gw_rows, model_squad, "model_prediction")
        totals["naive"] += gameweek_points_with_captain(gw_rows, naive_squad, "naive_prediction")
        totals["most_owned"] += gameweek_points_with_captain(gw_rows, naive_squad, "selected")
    return totals


# Trains one model per position using only data strictly before as_of_gw, exactly as a live run would see it.
def train_as_of(table: pd.DataFrame, target_season: str, as_of_gw: int, columns: list[str]) -> dict:
    train_mask = (table["season"] < target_season) | ((table["season"] == target_season) & (table["GW"] < as_of_gw))
    train_rows = table[train_mask]
    models = {}
    for position in POSITIONS:
        pos_train_all = train_rows[train_rows["position"] == position].dropna(subset=[TARGET_COLUMN])
        pos_train, pos_valid = recency_split(pos_train_all)
        model, _ = train_position_model(pos_train, pos_valid, columns)
        models[position] = model
    return models


# Predicts the next `horizon` gameweeks: form held at each player's latest row at/before as_of_gw, fixture context varies per GW.
def build_horizon_points(
    table: pd.DataFrame, target_season: str, as_of_gw: int, horizon: int, models: dict, columns: list[str]
) -> pd.DataFrame:
    roll_cols = [c for c in columns if any(c.endswith(f"_roll{w}") for w in WINDOWS)]
    context_cols = [c for c in columns if c not in roll_cols]

    prior_rows = table[(table["season"] == target_season) & (table["GW"] <= as_of_gw)].sort_values("GW")
    form = prior_rows.groupby("name", as_index=False).tail(1)[["name", "position", *roll_cols]]

    horizon_rows = []
    for gw in range(as_of_gw, as_of_gw + horizon):
        gw_context = table[(table["season"] == target_season) & (table["GW"] == gw)][["name", *context_cols]]
        merged = form.merge(gw_context, on="name", how="left")
        horizon_rows.append(merged)
    horizon_table = pd.concat(horizon_rows, ignore_index=True)

    horizon_table["gw_prediction"] = 0.0
    for position in POSITIONS:
        mask = horizon_table["position"] == position
        if mask.any():
            horizon_table.loc[mask, "gw_prediction"] = models[position].predict(horizon_table.loc[mask, columns])
    horizon_table.loc[horizon_table["fixture_count"].isna(), "gw_prediction"] = 0

    grouped = horizon_table.groupby(["name", "position"], as_index=False)["gw_prediction"].sum()
    return grouped.rename(columns={"gw_prediction": "horizon_points", "name": "web_name"})


# Simulates weekly transfers (season_planner-guided if use_season_planner else greedy-only) against actual outcomes.
def simulate_season_transfers(
    table: pd.DataFrame, target_season: str, first_gw: int, last_gw: int, horizon: int, rules: SquadRules,
    use_season_planner: bool,
) -> dict:
    columns = feature_columns(table)

    early_models = train_as_of(table, target_season, first_gw, columns)
    first_points = build_horizon_points(table, target_season, first_gw, 1, early_models, columns)
    pool = target_season_player_pool(table, target_season, first_gw)
    candidates = pool.merge(first_points[["web_name", "horizon_points"]], on="web_name", how="inner")
    candidates = candidates.rename(columns={"horizon_points": "predicted_points"})

    vol_table = add_minutes_volatility(table[table["season"] == target_season], group_keys=["season", "element"])
    prior_volatility = vol_table[vol_table["GW"] < first_gw].sort_values("GW").groupby("name", as_index=False).tail(1)
    candidates = candidates.merge(prior_volatility[["name", "minutes_volatility"]], left_on="web_name", right_on="name", how="left")
    candidates["minutes_volatility"] = candidates["minutes_volatility"].fillna(0.0)

    squad, _initial_xi, _initial_bench = select_squad_and_starting_xi(candidates, rules)
    squad["player_id"] = squad["web_name"]
    pool["player_id"] = pool["web_name"]
    squad = squad[SQUAD_COLUMNS]
    pool = pool[SQUAD_COLUMNS]

    bank = rules.budget_million - squad["now_cost_million"].sum()
    free_transfers = rules.free_transfers_per_week
    cumulative_points = 0.0
    weekly_log = []

    for gw in range(first_gw, last_gw + 1):
        models = train_as_of(table, target_season, gw, columns)
        horizon_points = build_horizon_points(table, target_season, gw, horizon, models, columns)

        squad_h = squad.merge(horizon_points[["web_name", "horizon_points"]], on="web_name", how="left")
        squad_h["horizon_points"] = squad_h["horizon_points"].fillna(0)
        pool_h = pool.merge(horizon_points[["web_name", "horizon_points"]], on="web_name", how="left")
        pool_h["horizon_points"] = pool_h["horizon_points"].fillna(0)

        core_player_ids: set = set()
        if use_season_planner:
            classified = classify_core_vs_rotational(squad_h.assign(predicted_points=squad_h["horizon_points"] / horizon))
            core_player_ids = set(classified.loc[classified["classification"] == "core", "player_id"])

        recommendation = recommend_transfers(squad_h, pool_h, bank, free_transfers, rules, core_player_ids=core_player_ids)

        kept = squad_h[~squad_h["player_id"].isin(recommendation["dropped"]["player_id"])]
        squad = pd.concat([kept[SQUAD_COLUMNS], recommendation["bought"][SQUAD_COLUMNS]], ignore_index=True)

        bank = bank - (recommendation["bought"]["now_cost_million"].sum() - recommendation["dropped"]["now_cost_million"].sum())
        free_transfers = min(
            rules.free_transfers_max_banked,
            max(0, free_transfers - recommendation["num_transfers"]) + rules.free_transfers_per_week,
        )

        squad_for_xi = squad.merge(horizon_points[["web_name", "horizon_points"]], on="web_name", how="left")
        squad_for_xi["horizon_points"] = squad_for_xi["horizon_points"].fillna(0)
        squad_for_xi["predicted_points"] = squad_for_xi["horizon_points"] / horizon
        starting_xi, _bench = select_starting_xi(squad_for_xi, rules)
        captain, _vice_captain = choose_captaincy(starting_xi)

        gw_rows = table[(table["season"] == target_season) & (table["GW"] == gw) & (table["name"].isin(starting_xi["web_name"]))]
        gw_points = gw_rows["total_points"].sum()
        captain_actual = gw_rows.loc[gw_rows["name"] == captain["web_name"], "total_points"].sum()
        hit_cost = recommendation["hits"] * abs(rules.points_hit_per_extra_transfer)
        gw_total = float(gw_points + captain_actual - hit_cost)
        cumulative_points += gw_total

        weekly_log.append(
            {
                "GW": gw, "gw_points": gw_total, "num_transfers": recommendation["num_transfers"],
                "hits": recommendation["hits"], "cumulative_points": cumulative_points,
            }
        )

    return {"cumulative_points": cumulative_points, "weekly_log": pd.DataFrame(weekly_log), "final_squad": squad}


# Sums each named player's real total_points across the `span` gameweeks starting at from_gw, for sanity-checking.
def actual_points_following(table: pd.DataFrame, target_season: str, names: list[str], from_gw: int, span: int) -> pd.Series:
    window = table[
        (table["season"] == target_season) & (table["GW"] >= from_gw) & (table["GW"] < from_gw + span) & (table["name"].isin(names))
    ]
    return window.groupby("name")["total_points"].sum()
