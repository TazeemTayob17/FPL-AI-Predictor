"""Walk-forward backtest: retrains on strictly-prior data per gameweek, scoring the model against baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fpl_agent.models.naive_predictor import previous_season_label
from fpl_agent.models.train import POSITIONS, TARGET_COLUMN, feature_columns, train_position_model
from fpl_agent.optimizer.squad_optimizer import select_squad

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MIN_EVAL_GW = 6


def recency_split(rows: pd.DataFrame, valid_fraction: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits time-ordered rows into an earlier training slice and a small most-recent validation slice for early stopping."""
    ordered = rows.sort_values(["season", "GW"])
    cutoff = max(1, int(len(ordered) * (1 - valid_fraction)))
    return ordered.iloc[:cutoff], ordered.iloc[cutoff:]


def naive_prior_points(target_season: str) -> pd.DataFrame:
    """Builds the naive per-gameweek baseline: last season's total points divided evenly across that season's gameweeks."""
    prev_season = previous_season_label(target_season)
    gw_rows = pd.read_parquet(PROCESSED_DIR / "all_seasons_gw.parquet")
    prior_season_rows = gw_rows[gw_rows["season"] == prev_season]
    gameweeks_played = prior_season_rows["GW"].max()
    prior = prior_season_rows.groupby("name", as_index=False)["total_points"].sum()
    prior["naive_prediction"] = prior["total_points"] / gameweeks_played
    return prior[["name", "naive_prediction"]]


def walk_forward_backtest(table: pd.DataFrame, target_season: str, min_gw: int = MIN_EVAL_GW) -> pd.DataFrame:
    """For each evaluated gameweek, retrains on strictly-prior data only and predicts that gameweek's actual points."""
    columns = feature_columns(table)
    naive = naive_prior_points(target_season)
    target_gws = sorted(g for g in table.loc[table["season"] == target_season, "GW"].unique() if g >= min_gw)

    folds = []
    for gw in target_gws:
        train_mask = (table["season"] < target_season) | ((table["season"] == target_season) & (table["GW"] < gw))
        eval_mask = (table["season"] == target_season) & (table["GW"] == gw)
        train_rows = table[train_mask]
        keep_cols = ["name", "position", "GW", "total_points", "selected", *columns]
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


def accuracy_report(predictions: pd.DataFrame) -> dict:
    """Computes MAE, RMSE, and mean per-gameweek Spearman rank correlation for the model vs the naive baseline."""
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


def target_season_player_pool(table: pd.DataFrame, target_season: str, first_gw: int) -> pd.DataFrame:
    """Builds the static player pool (name, position, team, cost) as it stood at the season's first evaluated gameweek."""
    season_rows = table[(table["season"] == target_season) & (table["GW"] == first_gw)]
    pool = season_rows[["name", "position", "team", "value"]].drop_duplicates("name").copy()
    pool["now_cost_million"] = pool["value"] / 10
    return pool.rename(columns={"team": "team_name", "name": "web_name"})


def build_fixed_squad(pool: pd.DataFrame, predictions: pd.DataFrame, prediction_column: str, first_gw: int) -> set[str]:
    """Picks a 15-man squad once from that predictor's earliest predictions, to be held all season (no transfers)."""
    first_gw_pred = predictions.loc[predictions["GW"] == first_gw, ["name", prediction_column]]
    candidates = pool.merge(first_gw_pred, left_on="web_name", right_on="name", how="inner")
    candidates = candidates.rename(columns={prediction_column: "predicted_points"})
    squad = select_squad(candidates)
    return set(squad["web_name"])


def gameweek_points_with_captain(gw_rows: pd.DataFrame, squad_names: set[str], captaincy_column: str) -> float:
    """Sums a fixed squad's actual points for one gameweek, doubling whoever captaincy_column ranks highest that week."""
    squad_gw = gw_rows[gw_rows["name"].isin(squad_names)]
    if squad_gw.empty:
        return 0.0
    total = squad_gw["total_points"].sum()
    ranked = squad_gw.dropna(subset=[captaincy_column])
    captain_bonus = squad_gw.loc[ranked[captaincy_column].idxmax(), "total_points"] if not ranked.empty else 0
    return float(total + captain_bonus)


def simulate_cumulative_points(predictions: pd.DataFrame, pool: pd.DataFrame) -> dict[str, float]:
    """Simulates a fixed, never-transferred squad under model-, naive-, and ownership-driven captaincy."""
    first_gw = predictions["GW"].min()
    model_squad = build_fixed_squad(pool, predictions, "model_prediction", first_gw)
    naive_squad = build_fixed_squad(pool, predictions, "naive_prediction", first_gw)

    totals = {"model": 0.0, "naive": 0.0, "most_owned": 0.0}
    for _, gw_rows in predictions.groupby("GW"):
        totals["model"] += gameweek_points_with_captain(gw_rows, model_squad, "model_prediction")
        totals["naive"] += gameweek_points_with_captain(gw_rows, naive_squad, "naive_prediction")
        totals["most_owned"] += gameweek_points_with_captain(gw_rows, naive_squad, "selected")
    return totals


if __name__ == "__main__":
    import sys

    target_season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    training_table = pd.read_parquet(PROCESSED_DIR / "training_table.parquet")

    print(f"Running walk-forward backtest for {target_season} (from GW{MIN_EVAL_GW})...")
    predictions = walk_forward_backtest(training_table, target_season)

    print("\nAccuracy (all evaluated gameweeks):")
    for label, metrics in accuracy_report(predictions).items():
        print(f"  {label}: MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  rank_corr={metrics['mean_rank_correlation']:.3f}")

    pool = target_season_player_pool(training_table, target_season, predictions["GW"].min())
    cumulative = simulate_cumulative_points(predictions, pool)
    print("\nCumulative points (fixed squad, no transfers, captained weekly by each strategy):")
    for label, points in cumulative.items():
        print(f"  {label}: {points:.0f}")
