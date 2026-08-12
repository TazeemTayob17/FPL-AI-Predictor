"""Thin wrappers around the free FPL API endpoints; every response gets cached as raw JSON."""

from __future__ import annotations

import requests

from fpl_agent.ingestion.cache import save_json

BASE_URL = "https://fantasy.premierleague.com/api"


def get_bootstrap_static() -> dict:
    """Fetches all players, teams, gameweeks, and prices."""
    data = _get(f"{BASE_URL}/bootstrap-static/")
    save_json("bootstrap", data)
    return data


def get_fixtures(event: int | None = None) -> list:
    """Fetches the full fixture list, or one gameweek's fixtures if event is given."""
    params = {"event": event} if event is not None else None
    data = _get(f"{BASE_URL}/fixtures/", params=params)
    save_json("fixtures", data)
    return data


def get_element_summary(player_id: int) -> dict:
    """Fetches one player's match-by-match history."""
    data = _get(f"{BASE_URL}/element-summary/{player_id}/")
    save_json(f"element_summary/{player_id}", data)
    return data


def get_event_live(gameweek: int) -> dict:
    """Fetches live scoring for one gameweek."""
    data = _get(f"{BASE_URL}/event/{gameweek}/live/")
    save_json(f"event_live/{gameweek}", data)
    return data


def get_entry(team_id: int) -> dict:
    """Fetches a manager's team info: bank, overall rank, current squad value."""
    data = _get(f"{BASE_URL}/entry/{team_id}/")
    save_json(f"entry/{team_id}/info", data)
    return data


def get_entry_picks(team_id: int, gameweek: int) -> dict:
    """Fetches a manager's squad picks and chip used for one gameweek; 404s before that gameweek's deadline."""
    data = _get(f"{BASE_URL}/entry/{team_id}/event/{gameweek}/picks/")
    save_json(f"entry/{team_id}/picks_gw{gameweek}", data)
    return data


def get_entry_history(team_id: int) -> dict:
    """Fetches a manager's full-season history: per-gameweek bank/value/transfers, and chips used."""
    data = _get(f"{BASE_URL}/entry/{team_id}/history/")
    save_json(f"entry/{team_id}/history", data)
    return data


def _get(url: str, params: dict | None = None) -> dict | list:
    """Sends a GET request and returns the parsed JSON body, raising on any HTTP error."""
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()
