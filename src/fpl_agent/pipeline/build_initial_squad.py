"""Entrypoint: runs the naive predictor + squad optimizer + captaincy against real local player data.

Superseded by weekly_pipeline.py once live team sync (Phase 3) and the real model (Phase 4) land;
useful now to pick the initial pre-season squad and to sanity-check the optimizer end to end.
"""

from __future__ import annotations

import pandas as pd

from fpl_agent.models.naive_predictor import predict_naive_points
from fpl_agent.optimizer.captaincy import choose_captaincy
from fpl_agent.optimizer.constraints import load_rules
from fpl_agent.optimizer.squad_optimizer import select_squad, select_starting_xi
from fpl_agent.storage.repository import PROCESSED_DIR


def build_initial_squad() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads current players from local storage and returns (squad, starting_xi, bench)."""
    players_path = PROCESSED_DIR / "players_current.parquet"
    if not players_path.exists():
        raise FileNotFoundError(
            f"{players_path} not found - run `python -m fpl_agent.pipeline.refresh_data` first."
        )
    players = pd.read_parquet(players_path)
    players = predict_naive_points(players)
    squad = select_squad(players)
    starting_xi, bench = select_starting_xi(squad)
    return squad, starting_xi, bench


def _print_report(squad: pd.DataFrame, starting_xi: pd.DataFrame, bench: pd.DataFrame) -> None:
    """Prints the squad, starting XI, bench, and captaincy choice in a human-readable form."""
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
    _print_report(*build_initial_squad())
