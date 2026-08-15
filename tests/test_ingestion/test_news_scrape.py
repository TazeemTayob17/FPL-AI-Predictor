"""Checks the FFScout injuries table parser: name-word matching, status mapping, and graceful skipping of unmatched rows."""

import pandas as pd

from fpl_agent.ingestion.news_scrape import _match_by_name_words, parse_injuries

PLAYERS = pd.DataFrame(
    [
        {"player_id": 1, "web_name_full": "Kaoru Mitoma", "team_short": "BHA"},
        {"player_id": 2, "web_name_full": "Amadou Onana", "team_short": "AVL"},
        {"player_id": 3, "web_name_full": "Ryan Christie", "team_short": "BOU"},
    ]
)

SAMPLE_HTML = """
<table>
<tr class="injuries-bans-item" data-team-code="bha">
    <td><img alt="Avatar of Mitoma Kaoru"></td>
    <td>badge</td>
    <td><span class="status doubt-50" title="Doubt 50%"><span>Doubt 50%</span></span></td>
    <td>01/09/2026</td>
    <td><strong>Knee injury</strong> Expected to return soon.</td>
    <td>14/08/2026</td>
</tr>
<tr class="injuries-bans-item" data-team-code="avl">
    <td><img alt="Avatar of Mvom Onana Amadou"></td>
    <td>badge</td>
    <td><span class="status injured" title="Injured"><span>Injured</span></span></td>
    <td>01/12/2026</td>
    <td><strong>ACL injury</strong> Long-term absence.</td>
    <td>14/08/2026</td>
</tr>
<tr class="injuries-bans-item" data-team-code="xyz">
    <td><img alt="Avatar of Unknown Player"></td>
    <td>badge</td>
    <td><span class="status injured" title="Injured"><span>Injured</span></span></td>
    <td>01/09/2026</td>
    <td><strong>Unknown</strong></td>
    <td>14/08/2026</td>
</tr>
</table>
"""


def test_match_by_name_words_handles_reversed_word_order():
    """FFScout's "Surname FirstName" order must still match our "FirstName Surname" records."""
    result = _match_by_name_words("Mitoma Kaoru", PLAYERS[PLAYERS["team_short"] == "BHA"])
    assert result.iloc[0]["player_id"] == 1


def test_match_by_name_words_handles_an_extra_middle_name():
    """A scraped name with an extra middle word must still match via subset containment."""
    result = _match_by_name_words("Mvom Onana Amadou", PLAYERS[PLAYERS["team_short"] == "AVL"])
    assert result.iloc[0]["player_id"] == 2


def test_match_by_name_words_returns_empty_when_nothing_matches():
    """A name with no overlapping words against any candidate must return empty, not raise."""
    result = _match_by_name_words("Completely Different Name", PLAYERS[PLAYERS["team_short"] == "BOU"])
    assert result.empty


def test_parse_injuries_extracts_status_and_chance_correctly():
    """A "doubt-50" row must map to status "d" and chance_of_playing_next_round 50."""
    result = parse_injuries(SAMPLE_HTML, PLAYERS)
    mitoma = result[result["player_id"] == 1].iloc[0]
    assert mitoma["status"] == "d"
    assert mitoma["chance_of_playing_next_round"] == 50
    assert mitoma["news"] == "Knee injury"


def test_parse_injuries_maps_injured_status_to_zero_chance():
    """An "injured" row must map to status "i" and chance_of_playing_next_round 0."""
    result = parse_injuries(SAMPLE_HTML, PLAYERS)
    onana = result[result["player_id"] == 2].iloc[0]
    assert onana["status"] == "i"
    assert onana["chance_of_playing_next_round"] == 0


def test_parse_injuries_skips_rows_with_no_matching_player():
    """A team code or name with no corresponding player in our pool must be skipped, not raise or fabricate a row."""
    result = parse_injuries(SAMPLE_HTML, PLAYERS)
    assert len(result) == 2
