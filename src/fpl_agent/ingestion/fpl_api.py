# Thin wrappers around the free FPL API endpoints; every response gets cached as raw JSON.

from __future__ import annotations

import requests

from fpl_agent.ingestion.cache import load_latest_json_if_fresh, save_json

BASE_URL = "https://fantasy.premierleague.com/api"


# Fetches all players, teams, gameweeks, and prices.
def get_bootstrap_static() -> dict:
    data = _get(f"{BASE_URL}/bootstrap-static/")
    save_json("bootstrap", data)
    return data


# Fetches the full fixture list, or one gameweek's fixtures if event is given.
def get_fixtures(event: int | None = None) -> list:
    params = {"event": event} if event is not None else None
    data = _get(f"{BASE_URL}/fixtures/", params=params)
    save_json("fixtures", data)
    return data


# Fetches one player's match-by-match history, reusing a cached pull younger than max_age_hours if given.
def get_element_summary(player_id: int, max_age_hours: float | None = None) -> dict:
    if max_age_hours is not None:
        cached = load_latest_json_if_fresh(f"element_summary/{player_id}", max_age_hours)
        if cached is not None:
            return cached
    data = _get(f"{BASE_URL}/element-summary/{player_id}/")
    save_json(f"element_summary/{player_id}", data)
    return data


# Fetches live scoring for one gameweek.
def get_event_live(gameweek: int) -> dict:
    data = _get(f"{BASE_URL}/event/{gameweek}/live/")
    save_json(f"event_live/{gameweek}", data)
    return data


# Fetches a manager's team info: bank, overall rank, current squad value.
def get_entry(team_id: int) -> dict:
    data = _get(f"{BASE_URL}/entry/{team_id}/")
    save_json(f"entry/{team_id}/info", data)
    return data


# Fetches a manager's squad picks and chip used for one gameweek; 404s before that gameweek's deadline.
def get_entry_picks(team_id: int, gameweek: int) -> dict:
    data = _get(f"{BASE_URL}/entry/{team_id}/event/{gameweek}/picks/")
    save_json(f"entry/{team_id}/picks_gw{gameweek}", data)
    return data


# Fetches a manager's full-season history: per-gameweek bank/value/transfers, and chips used.
def get_entry_history(team_id: int) -> dict:
    data = _get(f"{BASE_URL}/entry/{team_id}/history/")
    save_json(f"entry/{team_id}/history", data)
    return data


# Fetches a classic mini-league's current standings (public, no auth); paginates past the first ~50 entries.
def get_league_standings(league_id: int, page: int | None = None) -> dict:
    params = {"page_standings": page} if page is not None else None
    data = _get(f"{BASE_URL}/leagues-classic/{league_id}/standings/", params=params)
    save_json(f"leagues_classic/{league_id}", data)
    return data


# Sends a GET request and returns the parsed JSON body, raising on any HTTP error.
def _get(url: str, params: dict | None = None) -> dict | list:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()
