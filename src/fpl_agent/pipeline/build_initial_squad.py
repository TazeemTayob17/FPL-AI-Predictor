# Entrypoint: runs the real predictor (cold-start prior or trained model) + squad optimizer + captaincy against local player data.

from __future__ import annotations

import pandas as pd

from fpl_agent.ingestion.cache import load_latest_json
from fpl_agent.models.predict import predict_horizon_points
from fpl_agent.optimizer.captaincy import choose_captaincy
from fpl_agent.optimizer.constraints import load_rules
from fpl_agent.optimizer.squad_optimizer import select_squad_and_starting_xi
from fpl_agent.storage.repository import PROCESSED_DIR

INITIAL_SQUAD_HORIZON_GWS = 1  # a single-GW-equivalent estimate, matching this entrypoint's original semantics


# Loads current players from local storage (or uses players/bootstrap if already fetched, e.g. from the shared multi-visitor cache) and returns (squad, starting_xi, bench, all_players, used_cold_start).
def build_initial_squad(
    players: pd.DataFrame | None = None, bootstrap: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    if players is None:
        players_path = PROCESSED_DIR / "players_current.parquet"
        if not players_path.exists():
            raise FileNotFoundError(f"{players_path} not found - run `python -m fpl_agent.pipeline.refresh_data` first.")
        players = pd.read_parquet(players_path)
    if bootstrap is None:
        bootstrap = load_latest_json("bootstrap")
        if bootstrap is None:
            raise FileNotFoundError("No cached bootstrap payload found - run `python -m fpl_agent.pipeline.refresh_data` first.")

    all_players, used_cold_start = predict_horizon_points(players, bootstrap, horizon_gws=INITIAL_SQUAD_HORIZON_GWS)
    squad, starting_xi, bench = select_squad_and_starting_xi(all_players)
    return squad, starting_xi, bench, all_players, used_cold_start


# Prints the squad, starting XI, bench, and captaincy choice in a human-readable form.
def print_squad_report(squad: pd.DataFrame, starting_xi: pd.DataFrame, bench: pd.DataFrame) -> None:
    rules = load_rules()
    columns = ["web_name", "team_short", "position", "now_cost_million", "predicted_points"]

    print(f"Squad cost: £{squad['now_cost_million'].sum():.1f}m / £{rules.budget_million:.1f}m")
    print(squad[columns].sort_values(["position", "predicted_points"], ascending=[True, False]).to_string(index=False))

    print("\nStarting XI:")
    print(starting_xi[columns].sort_values(["position", "predicted_points"], ascending=[True, False]).to_string(index=False))

    print("\nBench:")
    print(bench[columns].to_string(index=False))

    captain, vice_captain = choose_captaincy(starting_xi)
    print(f"\nCaptain: {captain['web_name']} ({captain['predicted_points']:.0f} pts)")
    print(f"Vice-captain: {vice_captain['web_name']} ({vice_captain['predicted_points']:.0f} pts)")


if __name__ == "__main__":
    squad, starting_xi, bench, _all_players, _used_cold_start = build_initial_squad()
    print_squad_report(squad, starting_xi, bench)
