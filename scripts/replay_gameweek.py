"""Historical replay: recommends a transfer and captain for a past gameweek, checkable against what actually happened."""

from __future__ import annotations

import sys

import pandas as pd
from backtest_season import PROCESSED_DIR, recency_split, target_season_player_pool

from fpl_agent.features.rolling_stats import WINDOWS, add_minutes_volatility
from fpl_agent.models.train import POSITIONS, TARGET_COLUMN, feature_columns, train_position_model
from fpl_agent.optimizer.captaincy import captaincy_candidates
from fpl_agent.optimizer.constraints import load_rules
from fpl_agent.optimizer.squad_optimizer import select_squad, select_starting_xi
from fpl_agent.optimizer.transfer_optimizer import recommend_transfers

HORIZON = 5
MIN_SQUAD_FORMATION_GW = 6


def train_as_of(table: pd.DataFrame, target_season: str, as_of_gw: int, columns: list[str]) -> dict:
    """Trains one model per position using only data strictly before as_of_gw, exactly as a live run would see it."""
    train_mask = (table["season"] < target_season) | ((table["season"] == target_season) & (table["GW"] < as_of_gw))
    train_rows = table[train_mask]
    models = {}
    for position in POSITIONS:
        pos_train_all = train_rows[train_rows["position"] == position].dropna(subset=[TARGET_COLUMN])
        pos_train, pos_valid = recency_split(pos_train_all)
        model, _ = train_position_model(pos_train, pos_valid, columns)
        models[position] = model
    return models


def build_horizon_points(
    table: pd.DataFrame, target_season: str, as_of_gw: int, horizon: int, models: dict, columns: list[str]
) -> pd.DataFrame:
    """Predicts the next `horizon` gameweeks: rolling form held at its as_of_gw snapshot, fixture context varies per week."""
    roll_cols = [c for c in columns if any(c.endswith(f"_roll{w}") for w in WINDOWS)]
    context_cols = [c for c in columns if c not in roll_cols]

    form = table[(table["season"] == target_season) & (table["GW"] == as_of_gw)][["name", "position", *roll_cols]]

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


def actual_points_following(table: pd.DataFrame, target_season: str, names: list[str], from_gw: int, span: int) -> pd.Series:
    """Sums each named player's real total_points across the `span` gameweeks starting at from_gw, for sanity-checking."""
    window = table[
        (table["season"] == target_season) & (table["GW"] >= from_gw) & (table["GW"] < from_gw + span) & (table["name"].isin(names))
    ]
    return window.groupby("name")["total_points"].sum()


if __name__ == "__main__":
    target_season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    as_of_gw = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    table = pd.read_parquet(PROCESSED_DIR / "training_table.parquet")
    columns = feature_columns(table)
    rules = load_rules()

    print(f"Replaying {target_season}, standing right before GW{as_of_gw}...\n")

    print("Forming the held squad from early-season predictions...")
    first_gw = MIN_SQUAD_FORMATION_GW
    early_models = train_as_of(table, target_season, first_gw, columns)
    first_gw_points = build_horizon_points(table, target_season, first_gw, 1, early_models, columns)
    pool = target_season_player_pool(table, target_season, first_gw)
    candidates = pool.merge(first_gw_points[["web_name", "horizon_points"]], on="web_name", how="inner")
    candidates = candidates.rename(columns={"horizon_points": "predicted_points"})
    squad = select_squad(candidates, rules)
    squad["player_id"] = squad["web_name"]
    pool["player_id"] = pool["web_name"]
    print(f"Squad held since GW{first_gw}, {len(squad)} players, £{squad['now_cost_million'].sum():.1f}m.\n")

    print(f"Training on data strictly before GW{as_of_gw}, predicting the next {HORIZON} gameweeks...")
    models = train_as_of(table, target_season, as_of_gw, columns)
    horizon_points = build_horizon_points(table, target_season, as_of_gw, HORIZON, models, columns)

    squad_h = squad.merge(horizon_points[["web_name", "horizon_points"]], on="web_name", how="left")
    squad_h["horizon_points"] = squad_h["horizon_points"].fillna(0)
    pool_h = pool.merge(horizon_points[["web_name", "horizon_points"]], on="web_name", how="left")
    pool_h["horizon_points"] = pool_h["horizon_points"].fillna(0)

    bank = rules.budget_million - squad["now_cost_million"].sum()
    free_transfers = rules.free_transfers_max_banked
    print(f"\nBank: £{bank:.1f}m | Free transfers assumed: {free_transfers} (never transferred, banked to the cap)\n")

    recommendation = recommend_transfers(squad_h, pool_h, bank, free_transfers, rules)
    print("Transfer recommendation:")
    for line in recommendation["reasoning"]:
        print(f"  - {line}")

    if recommendation["num_transfers"] > 0:
        involved = list(recommendation["dropped"]["web_name"]) + list(recommendation["bought"]["web_name"])
        actual = actual_points_following(table, target_season, involved, as_of_gw, HORIZON)
        print(f"\nWhat actually happened over the following {HORIZON} gameweeks:")
        for name in involved:
            print(f"  {name}: {actual.get(name, 0):.0f} actual points")

    print(f"\nCaptaincy for GW{as_of_gw}:")
    xi_input = squad_h.assign(predicted_points=squad_h["horizon_points"] / HORIZON)
    starting_xi, _ = select_starting_xi(xi_input, rules)

    vol_table = add_minutes_volatility(table[table["season"] == target_season], group_keys=["season", "element"])
    latest_vol = vol_table[vol_table["GW"] == as_of_gw - 1][["name", "minutes_volatility"]]
    starting_xi = starting_xi.merge(latest_vol, left_on="web_name", right_on="name", how="left")
    starting_xi["chance_of_playing_next_round"] = 100
    ranked = captaincy_candidates(starting_xi)
    print(ranked[["web_name", "predicted_points", "risk_score"]].head(3).to_string(index=False))

    actual_captain_gw = actual_points_following(table, target_season, ranked["web_name"].tolist(), as_of_gw, 1)
    print(f"\nWhat actually happened in GW{as_of_gw}:")
    for name in ranked["web_name"].head(3):
        print(f"  {name}: {actual_captain_gw.get(name, 0):.0f} actual points")
