# Checks settings.yaml's editable fields round-trip via surgical text edits without stripping comments.

from fpl_agent.utils.settings import (
    load_differential_aggressiveness,
    load_mini_league_ids,
    save_differential_aggressiveness,
    save_mini_league_ids,
)

SAMPLE_SETTINGS = """\
# a top-level comment that must survive every edit
team:
  mini_league_ids: [111]

strategy:
  differential_aggressiveness: "balanced"   # template | balanced | differential
"""


# Writing a new value must update it in place and leave the inline comment and every other line untouched.
def test_save_differential_aggressiveness_preserves_comments(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(SAMPLE_SETTINGS, encoding="utf-8")

    save_differential_aggressiveness("chase", settings_path=settings_path)

    text = settings_path.read_text(encoding="utf-8")
    assert 'differential_aggressiveness: "chase"' in text
    assert "template | balanced | differential" in text
    assert "a top-level comment that must survive every edit" in text
    assert load_differential_aggressiveness(settings_path=settings_path) == "chase"


# Writing a new league-ID list must replace the flow-style list in place without disturbing the rest of the file.
def test_save_mini_league_ids_preserves_comments(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(SAMPLE_SETTINGS, encoding="utf-8")

    save_mini_league_ids([222, 333], settings_path=settings_path)

    text = settings_path.read_text(encoding="utf-8")
    assert "mini_league_ids: [222, 333]" in text
    assert "a top-level comment that must survive every edit" in text
    assert load_mini_league_ids(settings_path=settings_path) == [222, 333]
