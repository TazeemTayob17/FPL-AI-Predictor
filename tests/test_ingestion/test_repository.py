# Checks that bootstrap-static rows get normalized into a readable players table.

import pandas as pd

from fpl_agent.storage.db import get_connection, init_db
from fpl_agent.storage.repository import _build_players_frame, normalize_league_standings, read_snapshot_history

SAMPLE_BOOTSTRAP = {
    "elements": [
        {
            "id": 1, "first_name": "Erling", "second_name": "Haaland", "web_name": "Haaland",
            "team": 1, "element_type": 4, "now_cost": 150, "form": "5.2", "total_points": 20,
            "selected_by_percent": "45.3", "status": "a", "news": "",
            "chance_of_playing_this_round": None, "chance_of_playing_next_round": None,
            "transfers_in_event": 1000, "transfers_out_event": 200, "cost_change_event": 1,
        },
    ],
    "teams": [{"id": 1, "name": "Man City", "short_name": "MCI"}],
    "element_types": [
        {"id": 1, "singular_name_short": "GKP"}, {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"}, {"id": 4, "singular_name_short": "FWD"},
    ],
}


# Checks that team name and position get attached and cost is converted to millions.
def test_build_players_frame_joins_team_and_position():
    players = _build_players_frame(SAMPLE_BOOTSTRAP)
    row = players.iloc[0]
    assert row["team_short"] == "MCI"
    assert row["position"] == "FWD"
    assert row["now_cost_million"] == 15.0
    assert row["form"] == 5.2


RAW_STANDINGS = {
    "standings": {
        "results": [
            {"entry": 111, "entry_name": "Leader FC", "player_name": "Alice", "rank": 1, "last_rank": 1, "total": 500, "event_total": 60},
            {"entry": 222, "entry_name": "Chaser FC", "player_name": "Bob", "rank": 2, "last_rank": 3, "total": 470, "event_total": 55},
        ]
    }
}


# The leader's own gap must be 0, and the chaser's gap must be leader_total - their_total.
def test_normalize_league_standings_computes_points_behind_leader():
    standings = normalize_league_standings(RAW_STANDINGS, league_id=999)
    leader = standings[standings["team_id"] == 111].iloc[0]
    chaser = standings[standings["team_id"] == 222].iloc[0]
    assert leader["points_behind_leader"] == 0
    assert chaser["points_behind_leader"] == 30
    assert (standings["league_id"] == 999).all()


# A league with no results yet (e.g. before anyone has a score) must return an empty frame, not crash.
def test_normalize_league_standings_handles_an_empty_results_list():
    standings = normalize_league_standings({"standings": {"results": []}}, league_id=999)
    assert standings.empty


# Reading a specific player's history must exclude other players' rows and come back time-ordered.
def test_read_snapshot_history_filters_by_player_and_orders_by_time(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO snapshots (pulled_at, gameweek, player_id, now_cost) VALUES ('2026-08-14T10:00:00', 1, 1, 100)"
        )
        conn.execute(
            "INSERT INTO snapshots (pulled_at, gameweek, player_id, now_cost) VALUES ('2026-08-15T10:00:00', 1, 1, 101)"
        )
        conn.execute(
            "INSERT INTO snapshots (pulled_at, gameweek, player_id, now_cost) VALUES ('2026-08-15T10:00:00', 1, 2, 55)"
        )
        conn.commit()
    finally:
        conn.close()

    history = read_snapshot_history(player_id=1, db_path=db_path)

    assert list(history["now_cost"]) == [100, 101]
