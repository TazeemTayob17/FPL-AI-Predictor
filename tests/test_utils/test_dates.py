# Checks gameweek deadline lookup and first/second-half chip-window arithmetic.

from datetime import datetime, timezone

from fpl_agent.utils.dates import current_chip_half, gameweek_deadline, gws_remaining_in_first_half

BOOTSTRAP = {
    "events": [
        {"id": 1, "deadline_time": "2026-08-21T17:30:00Z"},
        {"id": 19, "deadline_time": "2027-01-02T13:30:00Z"},
    ]
}


# GW1's deadline must be parsed as an aware UTC datetime matching the raw ISO string.
def test_gameweek_deadline_parses_the_matching_events_deadline():
    result = gameweek_deadline(BOOTSTRAP, 1)
    assert result == datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)


# A gameweek not present in bootstrap-static's events list must return None, not raise.
def test_gameweek_deadline_returns_none_for_an_unknown_gameweek():
    assert gameweek_deadline(BOOTSTRAP, 99) is None


# GW19 itself (the deadline gameweek) still counts as first-half - the cutoff is inclusive.
def test_current_chip_half_at_and_before_the_cutoff_is_first_half():
    assert current_chip_half(1, first_half_deadline_gw=19) == "first_half"
    assert current_chip_half(19, first_half_deadline_gw=19) == "first_half"


# GW20, the gameweek right after the deadline, must count as second-half.
def test_current_chip_half_after_the_cutoff_is_second_half():
    assert current_chip_half(20, first_half_deadline_gw=19) == "second_half"


# At GW19 itself, zero gameweeks remain - this is the deadline gameweek.
def test_gws_remaining_in_first_half_counts_down_to_zero():
    assert gws_remaining_in_first_half(current_gw=15, first_half_deadline_gw=19) == 4
    assert gws_remaining_in_first_half(current_gw=19, first_half_deadline_gw=19) == 0


# Past the cutoff, the count goes negative rather than clamping - callers can tell "passed" from "imminent".
def test_gws_remaining_in_first_half_goes_negative_once_passed():
    assert gws_remaining_in_first_half(current_gw=20, first_half_deadline_gw=19) == -1
