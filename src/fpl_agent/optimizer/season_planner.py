# Long-horizon strategic layer guiding transfer_optimizer's weekly decisions with season-shape context.

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CORE_CHANCE_OF_PLAYING_MIN = 75
CORE_MINUTES_VOLATILITY_MAX = 15.0
CORE_PRICE_PERCENTILE = 0.6  # top 40% of the squad's price within each position = a season-long "premium" investment

# FPL's price-change algorithm is unpublished; net transfers-in/out this event is a coarse proxy for momentum.
PRICE_MOMENTUM_RISE_THRESHOLD = 50_000
PRICE_MOMENTUM_FALL_THRESHOLD = -50_000

WILDCARD_DIFFICULTY_GAP_THRESHOLD = 0.5

RISK_POSTURE_PROTECT = "protect"
RISK_POSTURE_BALANCED = "balanced"
RISK_POSTURE_CHASE = "chase"
CHASE_GWS_REMAINING_THRESHOLD = 8


# Tags each squad player "core" (premium-priced within their position and reliable) or "rotational".
def classify_core_vs_rotational(squad: pd.DataFrame) -> pd.DataFrame:
    squad = squad.copy()
    chance = squad.get("chance_of_playing_next_round", pd.Series(100, index=squad.index)).fillna(100)
    volatility = squad.get("minutes_volatility", pd.Series(0.0, index=squad.index)).fillna(0.0)
    reliable = (chance >= CORE_CHANCE_OF_PLAYING_MIN) & (volatility <= CORE_MINUTES_VOLATILITY_MAX)

    price_cutoff = squad.groupby("position")["now_cost_million"].transform(lambda prices: prices.quantile(CORE_PRICE_PERCENTILE))
    is_premium_priced = squad["now_cost_million"] >= price_cutoff

    squad["classification"] = "rotational"
    squad.loc[reliable & is_premium_priced, "classification"] = "core"
    return squad


# Flags players trending toward a price rise or fall from this event's net transfer activity.
def price_momentum(players: pd.DataFrame) -> pd.DataFrame:
    players = players.copy()
    net_transfers = players["transfers_in_event"] - players["transfers_out_event"]
    players["price_trend"] = "stable"
    players.loc[net_transfers >= PRICE_MOMENTUM_RISE_THRESHOLD, "price_trend"] = "likely_rise"
    players.loc[net_transfers <= PRICE_MOMENTUM_FALL_THRESHOLD, "price_trend"] = "likely_fall"
    return players


# Sums a specific squad's total value at each historical snapshot timestamp - the actual value trajectory.
def squad_value_trajectory(snapshot_history: pd.DataFrame, squad_player_ids: set[int]) -> pd.DataFrame:
    squad_rows = snapshot_history[snapshot_history["player_id"].isin(squad_player_ids)]
    if squad_rows.empty:
        return pd.DataFrame(columns=["pulled_at", "total_value"])
    return squad_rows.groupby("pulled_at", as_index=False)["now_cost"].sum().rename(columns={"now_cost": "total_value"})


# Scores each gameweek in gw_range for Bench Boost / Free Hit / Triple Captain suitability.
def forward_chip_window_targets(squad: pd.DataFrame, team_fixture_counts: pd.DataFrame, gw_range: range) -> pd.DataFrame:
    squad_teams = squad[["player_id", "web_name", "team_id", "classification", "predicted_points"]]

    rows = []
    for gw in gw_range:
        gw_counts = team_fixture_counts.loc[team_fixture_counts["GW"] == gw, ["team_id", "fixture_count"]]
        merged = squad_teams.merge(gw_counts, on="team_id", how="left")
        merged["fixture_count"] = merged["fixture_count"].fillna(0).astype(int)

        blanks = merged[merged["fixture_count"] == 0]
        doubles = merged[merged["fixture_count"] >= 2]
        best_double = doubles.loc[doubles["predicted_points"].idxmax()] if not doubles.empty else None

        rows.append(
            {
                "GW": gw,
                "bench_boost_score": len(doubles),
                "free_hit_score": len(blanks),
                "triple_captain_candidate": best_double["web_name"] if best_double is not None else None,
                "triple_captain_score": float(best_double["predicted_points"]) if best_double is not None else 0.0,
            }
        )
    return pd.DataFrame(rows)


# Flags a wildcard window when the squad's average forward fixture difficulty is notably worse than the league's.
def wildcard_fixture_swing_signal(squad: pd.DataFrame, team_difficulty: pd.DataFrame, gw_range: range) -> dict:
    window = team_difficulty[team_difficulty["GW"].isin(gw_range)]
    league_avg = window["difficulty"].mean()
    squad_avg = window.loc[window["team_id"].isin(set(squad["team_id"].unique())), "difficulty"].mean()
    gap = (squad_avg - league_avg) if pd.notna(squad_avg) and pd.notna(league_avg) else 0.0
    return {
        "squad_avg_difficulty": None if pd.isna(squad_avg) else float(squad_avg),
        "league_avg_difficulty": None if pd.isna(league_avg) else float(league_avg),
        "gap": float(gap),
        "suggest_wildcard": bool(gap >= WILDCARD_DIFFICULTY_GAP_THRESHOLD),
    }


# Sets template-vs-differential posture from the user's mini-league rank/gap (protect/chase/balanced).
def mini_league_risk_posture(standings: pd.DataFrame, team_id: int, gws_remaining: int) -> dict:
    if standings.empty or team_id not in standings["team_id"].values:
        return {
            "posture": RISK_POSTURE_BALANCED, "rank": None, "points_behind_leader": None,
            "reasoning": "No mini-league standings available yet (not configured, or configured but no gameweek has been scored yet).",
        }

    row = standings.loc[standings["team_id"] == team_id].iloc[0]
    rank = int(row["rank"])
    gap = int(row["points_behind_leader"])

    if gws_remaining > CHASE_GWS_REMAINING_THRESHOLD:
        posture = RISK_POSTURE_BALANCED
        reasoning = f"Rank {rank}, {gap} pts behind the leader, {gws_remaining} GWs left - plenty of time, no need to force it yet."
    elif rank == 1:
        posture = RISK_POSTURE_PROTECT
        reasoning = f"Leading with {gws_remaining} GWs left - leaning template/safe to protect the lead."
    elif gap > 0:
        posture = RISK_POSTURE_CHASE
        reasoning = f"Rank {rank}, {gap} pts behind the leader with only {gws_remaining} GWs left - leaning differential to close the gap."
    else:
        posture = RISK_POSTURE_BALANCED
        reasoning = f"Rank {rank}, tied on points, {gws_remaining} GWs left - staying balanced."

    return {"posture": posture, "rank": rank, "points_behind_leader": gap, "reasoning": reasoning}


# The full rolling season plan: classified squad shape, forward chip-window scores, wildcard signal, and mini-league risk posture.
@dataclass
class SeasonPlan:
    squad_classified: pd.DataFrame
    chip_window_scores: pd.DataFrame
    wildcard_signal: dict
    risk_posture: dict

    # The set of player_ids the plan is currently building around - transfer_optimizer penalizes selling these.
    @property
    def core_player_ids(self) -> set[int]:
        return set(self.squad_classified.loc[self.squad_classified["classification"] == "core", "player_id"])


# Builds the full rolling season plan from the current squad, forward fixture data, and (optionally) mini-league standings.
def build_season_plan(
    squad: pd.DataFrame,
    team_fixture_counts: pd.DataFrame,
    team_difficulty: pd.DataFrame,
    gw_range: range,
    standings: pd.DataFrame | None = None,
    team_id: int | None = None,
    gws_remaining_in_season: int = 38,
) -> SeasonPlan:
    classified = classify_core_vs_rotational(squad)
    chip_scores = forward_chip_window_targets(classified, team_fixture_counts, gw_range)
    wildcard = wildcard_fixture_swing_signal(classified, team_difficulty, gw_range)
    if standings is not None and team_id is not None:
        posture = mini_league_risk_posture(standings, team_id, gws_remaining_in_season)
    else:
        posture = {
            "posture": RISK_POSTURE_BALANCED, "rank": None, "points_behind_leader": None,
            "reasoning": "No mini-league configured.",
        }
    return SeasonPlan(squad_classified=classified, chip_window_scores=chip_scores, wildcard_signal=wildcard, risk_posture=posture)
