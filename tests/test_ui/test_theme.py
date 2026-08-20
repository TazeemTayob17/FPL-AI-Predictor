# Checks the kit-card rendering helpers: real shirt URLs (never hardcoded images), goalkeeper-specific kit, fixture-difficulty coloring, and captain/vice-captain badges.

import pandas as pd

from fpl_agent.ui.components.theme import _fixture_difficulty_colors, _player_card_html, _shirt_url

ROW = pd.Series({"web_name": "Salah", "position": "MID", "team_id": 1, "team_code": 14, "now_cost_million": 13.0, "predicted_points": 8.4})


# Outfield players get the standard shirt image, keyed by the club's stable code - not a locally hardcoded file.
def test_shirt_url_uses_team_code_for_outfield_players():
    url = _shirt_url(14, "MID")
    assert url == "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_14-66.png"


# Goalkeepers get FPL's distinct goalkeeper kit image (the "_1" suffix), not the outfield shirt.
def test_shirt_url_uses_goalkeeper_kit_for_gkp():
    url = _shirt_url(14, "GKP")
    assert url == "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_14_1-66.png"


# A missing team code (e.g. before the club-code column existed in cached data) must degrade to no image, not crash.
def test_shirt_url_returns_none_for_missing_team_code():
    assert _shirt_url(float("nan"), "MID") is None


# Easy fixtures (difficulty 1-2) get a green tint, hard fixtures (4-5) get a pink/red tint - mirrors FPL's own difficulty coloring.
def test_fixture_difficulty_colors_are_green_for_easy_and_pink_for_hard():
    easy_bg, _ = _fixture_difficulty_colors(2)
    hard_bg, _ = _fixture_difficulty_colors(5)
    assert easy_bg != hard_bg
    assert "ba" in easy_bg.lower() or easy_bg.startswith("#ba")


# A missing difficulty (no fixture data yet) must fall back to a neutral color, not raise.
def test_fixture_difficulty_colors_handles_missing_difficulty():
    bg, color = _fixture_difficulty_colors(None)
    assert bg and color


# The captain badge, the real shirt image, and the next-fixture text (colored by difficulty) must all appear in the rendered card.
def test_player_card_html_includes_shirt_fixture_and_captain_badge():
    fixtures_by_team = {1: {"opponent_short": "MCI", "is_home": True, "difficulty": 4}}
    html = _player_card_html(ROW, "predicted_points", fixtures_by_team, badge="C")

    assert "shirt_14-66.png" in html
    assert "MCI (H)" in html
    assert ">C<" in html


# A player with no badge gets the default "nailed on" checkmark badge, not a blank one.
def test_player_card_html_shows_checkmark_badge_when_not_captain_or_vice():
    html = _player_card_html(ROW, "predicted_points", {}, badge=None)
    assert "&#10003;" in html
