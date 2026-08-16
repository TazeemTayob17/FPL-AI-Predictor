# Rule-aware chip suggestion layer, respecting the two-sets-per-season/one-per-GW rules; suggests only, never auto-plays.

from __future__ import annotations

import pandas as pd

from fpl_agent.utils.dates import current_chip_half, gws_remaining_in_first_half

CHIP_NAMES = ("wildcard", "freehit", "bboost", "3xc")  # the FPL API's own chip name strings
CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}

BENCH_BOOST_MIN_SCORE = 3        # at least this many doubled squad players to justify Bench Boost
FREE_HIT_MIN_SCORE = 4           # at least this many blank squad players to justify Free Hit
TRIPLE_CAPTAIN_MIN_SCORE = 8.0   # a double-gameweek/standout-fixture premium worth at least this many predicted points
FIRST_HALF_URGENCY_GWS = 3       # inside this many GWs of the cutoff, warn about unused first-half chips

# Fixed priority when more than one GW-specific window qualifies the same week (scores aren't comparable units).
CHIP_PRIORITY = ("bboost", "3xc", "freehit")


# Splits a manager's chips-used history into which chips have already been used in each half.
def chips_used_by_half(chips_used: list[dict], first_half_deadline_gw: int) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {"first_half": set(), "second_half": set()}
    for entry in chips_used:
        half = current_chip_half(entry["gameweek"], first_half_deadline_gw)
        used[half].add(entry["chip"])
    return used


# The chips still available in the manager's current half, given what they've already used this half.
def available_chips(chips_used: list[dict], current_gw: int, first_half_deadline_gw: int) -> set[str]:
    half = current_chip_half(current_gw, first_half_deadline_gw)
    used_this_half = chips_used_by_half(chips_used, first_half_deadline_gw)[half]
    return set(CHIP_NAMES) - used_this_half


# Suggests at most one chip for a single gameweek, restricted to available chips.
def suggest_chip_for_gameweek(gw: int, chip_window_scores: pd.DataFrame, wildcard_signal: dict, available: set[str]) -> dict:
    row = chip_window_scores.loc[chip_window_scores["GW"] == gw]
    qualifying: dict[str, str] = {}

    if "bboost" in available and not row.empty and row.iloc[0]["bench_boost_score"] >= BENCH_BOOST_MIN_SCORE:
        score = row.iloc[0]["bench_boost_score"]
        qualifying["bboost"] = f"{int(score)} squad players have a double gameweek - strong Bench Boost window."

    if "freehit" in available and not row.empty and row.iloc[0]["free_hit_score"] >= FREE_HIT_MIN_SCORE:
        score = row.iloc[0]["free_hit_score"]
        qualifying["freehit"] = f"{int(score)} squad players blank this gameweek - Free Hit covers the gap."

    if "3xc" in available and not row.empty and row.iloc[0]["triple_captain_score"] >= TRIPLE_CAPTAIN_MIN_SCORE:
        score = row.iloc[0]["triple_captain_score"]
        candidate_name = row.iloc[0]["triple_captain_candidate"]
        qualifying["3xc"] = f"{candidate_name} has a double gameweek worth {score:.1f} predicted points - strong Triple Captain window."

    for chip in CHIP_PRIORITY:
        if chip in qualifying:
            return {"GW": gw, "chip": chip, "chip_label": CHIP_LABELS[chip], "reasoning": qualifying[chip]}

    if "wildcard" in available and wildcard_signal.get("suggest_wildcard"):
        gap = wildcard_signal["gap"]
        return {
            "GW": gw, "chip": "wildcard", "chip_label": CHIP_LABELS["wildcard"],
            "reasoning": f"Squad's forward fixtures are notably worse than the league average (difficulty gap {gap:.1f}).",
        }

    return {"GW": gw, "chip": None, "chip_label": None, "reasoning": "No chip window meets the suggestion threshold this gameweek."}


# Warns when first-half chips are close to being forfeited and still haven't all been used.
def first_half_urgency_warning(current_gw: int, chips_used: list[dict], first_half_deadline_gw: int) -> str | None:
    if current_chip_half(current_gw, first_half_deadline_gw) != "first_half":
        return None
    remaining_gws = gws_remaining_in_first_half(current_gw, first_half_deadline_gw)
    if remaining_gws > FIRST_HALF_URGENCY_GWS:
        return None
    unused = set(CHIP_NAMES) - chips_used_by_half(chips_used, first_half_deadline_gw)["first_half"]
    if not unused:
        return None
    labels = ", ".join(CHIP_LABELS[chip] for chip in sorted(unused))
    return f"Only {remaining_gws} GW(s) left before first-half chips are forfeited - still unused: {labels}."


# Builds the full chip-suggestion report: per-GW suggestions across gw_range, plus a first-half urgency warning.
def build_chip_suggestions(
    gw_range: range,
    chip_window_scores: pd.DataFrame,
    wildcard_signal: dict,
    chips_used: list[dict],
    current_gw: int,
    first_half_deadline_gw: int,
) -> dict:
    available = available_chips(chips_used, current_gw, first_half_deadline_gw)
    suggestions = [suggest_chip_for_gameweek(gw, chip_window_scores, wildcard_signal, available) for gw in gw_range]
    urgency = first_half_urgency_warning(current_gw, chips_used, first_half_deadline_gw)
    return {"available_chips": available, "suggestions": suggestions, "urgency_warning": urgency}
