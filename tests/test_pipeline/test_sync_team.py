"""Checks gameweek resolution and the pre-deadline fallback behavior of the team-sync entrypoint, with no network calls."""

import requests

from fpl_agent.pipeline import sync_team
from fpl_agent.pipeline.sync_team import _determine_target_gameweek


def test_determine_target_gameweek_prefers_entrys_own_current_event():
    """When the entry payload already names a current_event, trust it over history."""
    entry = {"current_event": 5}
    history = {"current": [{"event": 3}]}
    assert _determine_target_gameweek(entry, history) == 5


def test_determine_target_gameweek_falls_back_to_last_played_gameweek():
    """No current_event (common between gameweeks) - fall back to the manager's last-played GW in history."""
    entry = {"current_event": None}
    history = {"current": [{"event": 1}, {"event": 2}, {"event": 3}]}
    assert _determine_target_gameweek(entry, history) == 3


def test_determine_target_gameweek_returns_none_pre_season():
    """Nothing played yet and no current_event - there's no gameweek to fetch picks for."""
    entry = {"current_event": None}
    history = {"current": []}
    assert _determine_target_gameweek(entry, history) is None


def test_sync_team_returns_none_when_picks_404s(monkeypatch):
    """A 404 from the picks endpoint (pre-deadline) must be treated as 'not live yet', not an error."""
    monkeypatch.setattr(sync_team.fpl_api, "get_entry", lambda team_id: {"current_event": None})
    monkeypatch.setattr(sync_team.fpl_api, "get_entry_history", lambda team_id: {"current": [{"event": 1}], "chips": []})

    def _raise_404(team_id, gameweek):
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(sync_team.fpl_api, "get_entry_picks", _raise_404)

    assert sync_team.sync_team(team_id=123) is None


def test_sync_team_returns_none_before_any_gameweek_exists(monkeypatch):
    """Pre-season with no current_event and empty history - must short-circuit without calling the picks endpoint."""
    monkeypatch.setattr(sync_team.fpl_api, "get_entry", lambda team_id: {"current_event": None})
    monkeypatch.setattr(sync_team.fpl_api, "get_entry_history", lambda team_id: {"current": [], "chips": []})

    def _fail_if_called(team_id, gameweek):
        raise AssertionError("get_entry_picks should not be called when there is no target gameweek")

    monkeypatch.setattr(sync_team.fpl_api, "get_entry_picks", _fail_if_called)

    assert sync_team.sync_team(team_id=123) is None
