# Checks FPL_TEAM_ID gets written to .env surgically, without disturbing other lines.

from fpl_agent.utils.env import set_team_id


# Writing to a file with no FPL_TEAM_ID line yet must append one, not clobber existing content.
def test_set_team_id_appends_when_missing(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("# a comment\nOTHER_VAR=1\n", encoding="utf-8")

    set_team_id(12345, env_path=env_path)

    text = env_path.read_text(encoding="utf-8")
    assert "OTHER_VAR=1" in text
    assert "FPL_TEAM_ID=12345" in text


# Writing when FPL_TEAM_ID already has a value must replace it in place, not duplicate the line.
def test_set_team_id_replaces_existing_value(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("FPL_TEAM_ID=111\nOTHER_VAR=1\n", encoding="utf-8")

    set_team_id(999, env_path=env_path)

    text = env_path.read_text(encoding="utf-8")
    assert text.count("FPL_TEAM_ID=") == 1
    assert "FPL_TEAM_ID=999" in text
    assert "OTHER_VAR=1" in text


# Writing to a file that doesn't exist yet must create it.
def test_set_team_id_creates_file_when_missing(tmp_path):
    env_path = tmp_path / ".env"

    set_team_id(42, env_path=env_path)

    assert env_path.read_text(encoding="utf-8") == "FPL_TEAM_ID=42\n"
