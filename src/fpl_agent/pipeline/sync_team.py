# Entrypoint: syncs the manager's real FPL squad once it's live, falling back to a GW1 recommendation before then.

from __future__ import annotations

import requests

from fpl_agent.ingestion import fpl_api
from fpl_agent.optimizer.constraints import load_rules
from fpl_agent.pipeline.build_initial_squad import build_initial_squad, print_squad_report
from fpl_agent.storage import repository
from fpl_agent.utils.env import get_team_id


# Pulls entry/history/picks for the manager and saves a team_snapshot; returns None if picks aren't live yet.
def sync_team(team_id: int) -> dict | None:
    entry = fpl_api.get_entry(team_id)
    history = fpl_api.get_entry_history(team_id)

    target_gw = _determine_target_gameweek(entry, history)
    if target_gw is None:
        return None

    try:
        picks = fpl_api.get_entry_picks(team_id, target_gw)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise

    return repository.save_team_snapshot(entry, history, picks, load_rules())


# Picks the gameweek to fetch picks for: the manager's own current_event, or their last-played gameweek.
def _determine_target_gameweek(entry: dict, history: dict) -> int | None:
    if entry.get("current_event"):
        return entry["current_event"]
    current_history = history.get("current", [])
    return current_history[-1]["event"] if current_history else None


if __name__ == "__main__":
    snapshot = sync_team(get_team_id())
    if snapshot is None:
        print("Your live squad isn't available yet (the first gameweek deadline hasn't passed).")
        print("Here's a recommended squad to enter yourself before the deadline:\n")
        squad, starting_xi, bench, _all_players, _used_cold_start = build_initial_squad()
        print_squad_report(squad, starting_xi, bench)
    else:
        print(f"Synced GW{snapshot['gameweek']} squad from the live API.")
        print(
            f"Bank: £{snapshot['bank']:.1f}m | "
            f"Squad value: £{snapshot['squad_value']:.1f}m | "
            f"Free transfers: {snapshot['free_transfers']}"
        )
        print(f"Chips used: {snapshot['chips_used'] or 'none'}")
