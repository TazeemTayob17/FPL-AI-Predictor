"""Checks whole-word player-name matching against news text, including the short-surname safety cutoff."""

import pandas as pd

from fpl_agent.ingestion.news_rss import match_players_in_text

CANDIDATES = pd.DataFrame(
    [
        {"player_id": 1, "web_name": "Haaland"},
        {"player_id": 2, "web_name": "Saka"},
        {"player_id": 3, "web_name": "Cox"},
    ]
)


def test_matches_a_clear_whole_word_surname():
    """A surname appearing as a distinct word in the text must be matched."""
    result = match_players_in_text("Haaland ruled out for two weeks with a knock", CANDIDATES)
    assert result["web_name"].tolist() == ["Haaland"]


def test_does_not_match_a_substring_inside_another_word():
    """"Saka" must not match inside an unrelated longer word like "Osaka"."""
    result = match_players_in_text("Osaka hosted the tournament", CANDIDATES)
    assert result.empty


def test_short_surnames_below_the_length_cutoff_are_skipped():
    """A 3-letter surname ("Cox") is below the safety cutoff and must never be matched, even as a clean whole word."""
    result = match_players_in_text("Cox starts up front today", CANDIDATES)
    assert result.empty


def test_no_mention_returns_an_empty_frame():
    """Text mentioning none of the candidates must return an empty result, not an error."""
    result = match_players_in_text("Unrelated transfer news about another club entirely", CANDIDATES)
    assert result.empty


def test_matching_is_case_insensitive():
    """A lowercase mention should still match, since headlines aren't always properly cased."""
    result = match_players_in_text("breaking: haaland injury update", CANDIDATES)
    assert result["web_name"].tolist() == ["Haaland"]


def test_bare_low_content_entries_are_skipped():
    """A near-empty entry like a bare team-name tag must not produce a match, even if it happens to contain a surname."""
    result = match_players_in_text("Saka", CANDIDATES)
    assert result.empty
