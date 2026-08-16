# Checks chip-half availability tracking, per-GW suggestion selection, and the first-half use-it-or-lose-it warning.

import pandas as pd

from fpl_agent.optimizer.chip_strategy import (
    available_chips,
    build_chip_suggestions,
    chips_used_by_half,
    first_half_urgency_warning,
    suggest_chip_for_gameweek,
)

FIRST_HALF_DEADLINE_GW = 19

CHIP_SCORES = pd.DataFrame(
    [
        {"GW": 10, "bench_boost_score": 5, "free_hit_score": 0, "triple_captain_candidate": "Star", "triple_captain_score": 15.0},
        {"GW": 11, "bench_boost_score": 0, "free_hit_score": 6, "triple_captain_candidate": None, "triple_captain_score": 0.0},
        {"GW": 12, "bench_boost_score": 0, "free_hit_score": 0, "triple_captain_candidate": None, "triple_captain_score": 0.0},
    ]
)
NO_WILDCARD_SIGNAL = {"suggest_wildcard": False, "gap": 0.0}
WILDCARD_SIGNAL = {"suggest_wildcard": True, "gap": 1.2}
ALL_CHIPS = {"wildcard", "freehit", "bboost", "3xc"}


# A GW5 chip is first-half, a GW25 chip is second-half, given a GW19 cutoff.
def test_chips_used_by_half_splits_correctly_around_the_cutoff():
    chips_used = [{"chip": "wildcard", "gameweek": 5}, {"chip": "bboost", "gameweek": 25}]
    result = chips_used_by_half(chips_used, FIRST_HALF_DEADLINE_GW)
    assert result["first_half"] == {"wildcard"}
    assert result["second_half"] == {"bboost"}


# A wildcard used in the first half must be unavailable while still in the first half...
def test_available_chips_excludes_only_the_current_halfs_used_chips():
    chips_used = [{"chip": "wildcard", "gameweek": 5}]
    result = available_chips(chips_used, current_gw=10, first_half_deadline_gw=FIRST_HALF_DEADLINE_GW)
    assert result == ALL_CHIPS - {"wildcard"}


# ...but once GW20 (second half) arrives, the full set is available again regardless of first-half usage.
def test_available_chips_resets_once_the_second_half_starts():
    chips_used = [{"chip": "wildcard", "gameweek": 5}]
    result = available_chips(chips_used, current_gw=20, first_half_deadline_gw=FIRST_HALF_DEADLINE_GW)
    assert result == ALL_CHIPS


# GW10: both bboost and 3xc qualify - CHIP_PRIORITY picks bboost, not the larger raw (incomparable) score.
def test_suggest_chip_picks_bench_boost_over_triple_captain_by_priority():
    result = suggest_chip_for_gameweek(10, CHIP_SCORES, NO_WILDCARD_SIGNAL, available=ALL_CHIPS)
    assert result["chip"] == "bboost"


# Same GW10 double-gameweek data, but with bboost already used this half - 3xc is the next-priority qualifier.
def test_suggest_chip_picks_triple_captain_when_bench_boost_is_unavailable():
    result = suggest_chip_for_gameweek(10, CHIP_SCORES, NO_WILDCARD_SIGNAL, available=ALL_CHIPS - {"bboost"})
    assert result["chip"] == "3xc"


# GW11: free_hit_score (6) clears its threshold -> freehit suggested.
def test_suggest_chip_picks_free_hit_when_only_it_qualifies():
    result = suggest_chip_for_gameweek(11, CHIP_SCORES, NO_WILDCARD_SIGNAL, available=ALL_CHIPS)
    assert result["chip"] == "freehit"


# GW12: every score is below threshold and there's no wildcard signal -> no chip suggested.
def test_suggest_chip_returns_none_when_nothing_qualifies():
    result = suggest_chip_for_gameweek(12, CHIP_SCORES, NO_WILDCARD_SIGNAL, available=ALL_CHIPS)
    assert result["chip"] is None


# Even though bboost qualifies at GW10, it must not be suggested if it's not in the available set.
def test_suggest_chip_excludes_chips_already_used_this_half():
    result = suggest_chip_for_gameweek(10, CHIP_SCORES, NO_WILDCARD_SIGNAL, available=ALL_CHIPS - {"bboost"})
    assert result["chip"] != "bboost"


# GW12 has no bboost/freehit/3xc window, but the wildcard fixture-swing signal is active -> wildcard suggested.
def test_suggest_chip_falls_back_to_wildcard_when_no_gw_specific_window_qualifies():
    result = suggest_chip_for_gameweek(12, CHIP_SCORES, WILDCARD_SIGNAL, available=ALL_CHIPS)
    assert result["chip"] == "wildcard"


# GW10 has both a qualifying bench_boost window AND an active wildcard signal - the GW-specific one wins.
def test_suggest_chip_prefers_gw_specific_window_over_wildcard_in_the_same_week():
    result = suggest_chip_for_gameweek(10, CHIP_SCORES, WILDCARD_SIGNAL, available=ALL_CHIPS)
    assert result["chip"] == "bboost"


# Far from the GW19 cutoff, no urgency warning is needed even with chips unused.
def test_first_half_urgency_warning_is_silent_with_plenty_of_gws_left():
    assert first_half_urgency_warning(current_gw=5, chips_used=[], first_half_deadline_gw=FIRST_HALF_DEADLINE_GW) is None


# Inside the urgency window (GW19 - 3 = GW16) with nothing used yet, all four chips should be listed.
def test_first_half_urgency_warning_fires_close_to_the_cutoff_with_unused_chips():
    warning = first_half_urgency_warning(current_gw=17, chips_used=[], first_half_deadline_gw=FIRST_HALF_DEADLINE_GW)
    assert warning is not None
    assert "Wildcard" in warning and "Triple Captain" in warning


# If all four chips are already used this half, there's nothing left to warn about.
def test_first_half_urgency_warning_is_silent_once_everything_is_used():
    chips_used = [{"chip": c, "gameweek": 10} for c in ("wildcard", "freehit", "bboost", "3xc")]
    assert first_half_urgency_warning(current_gw=17, chips_used=chips_used, first_half_deadline_gw=FIRST_HALF_DEADLINE_GW) is None


# Past the cutoff, the first-half-forfeiture warning no longer applies at all.
def test_first_half_urgency_warning_is_silent_in_the_second_half():
    assert first_half_urgency_warning(current_gw=25, chips_used=[], first_half_deadline_gw=FIRST_HALF_DEADLINE_GW) is None


# The full report must cover every gameweek in the range and include the urgency warning field.
def test_build_chip_suggestions_produces_one_row_per_gameweek_plus_urgency():
    report = build_chip_suggestions(
        gw_range=range(10, 13), chip_window_scores=CHIP_SCORES, wildcard_signal=NO_WILDCARD_SIGNAL,
        chips_used=[], current_gw=10, first_half_deadline_gw=FIRST_HALF_DEADLINE_GW,
    )
    assert [s["GW"] for s in report["suggestions"]] == [10, 11, 12]
    assert report["available_chips"] == ALL_CHIPS
    assert "urgency_warning" in report
