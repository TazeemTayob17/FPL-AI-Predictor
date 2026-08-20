# Each team's next unplayed fixture (opponent short name + home/away), read from the locally cached current-season data - no live API calls, matching the dashboard's "never call live APIs on page load" rule.

from __future__ import annotations

import pandas as pd

from fpl_agent.storage.repository import PROCESSED_DIR


# team_id -> short name (e.g. "ARS"), from the locally cached current-players table.
def load_team_short_names() -> dict[int, str]:
    players_path = PROCESSED_DIR / "players_current.parquet"
    if not players_path.exists():
        return {}
    players = pd.read_parquet(players_path, columns=["team_id", "team_short"])
    return players.drop_duplicates("team_id").set_index("team_id")["team_short"].to_dict()


# team_id -> {"opponent_short": str, "is_home": bool, "difficulty": int} for that team's earliest not-yet-finished fixture.
def next_fixture_by_team() -> dict[int, dict]:
    fixtures_path = PROCESSED_DIR / "fixtures_current.parquet"
    if not fixtures_path.exists():
        return {}
    fixtures = pd.read_parquet(fixtures_path)
    unplayed = fixtures[~fixtures["finished"]].sort_values("event")
    if unplayed.empty:
        return {}

    short_names = load_team_short_names()
    result: dict[int, dict] = {}
    for _, fixture in unplayed.iterrows():
        for team_id, opponent_id, is_home, difficulty in (
            (fixture["team_h"], fixture["team_a"], True, fixture.get("team_h_difficulty")),
            (fixture["team_a"], fixture["team_h"], False, fixture.get("team_a_difficulty")),
        ):
            if team_id not in result:
                result[team_id] = {"opponent_short": short_names.get(opponent_id, "?"), "is_home": is_home, "difficulty": difficulty}
    return result
