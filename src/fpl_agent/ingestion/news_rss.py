"""Pulls official team-news RSS feeds (BBC Sport per-team, Sky Sports general) and matches mentions to known players."""

from __future__ import annotations

import re

import feedparser
import pandas as pd

from fpl_agent.ingestion.cache import save_json

MIN_SURNAME_LENGTH = 4
MIN_TEXT_LENGTH = 20

BBC_TEAM_SLUGS = {
    "ars": "arsenal", "avl": "aston-villa", "bou": "afc-bournemouth", "bre": "brentford",
    "bha": "brighton-and-hove-albion", "che": "chelsea", "cov": "coventry-city",
    "cry": "crystal-palace", "eve": "everton", "ful": "fulham", "hul": "hull-city",
    "ips": "ipswich-town", "lee": "leeds-united", "liv": "liverpool", "mci": "manchester-city",
    "mun": "manchester-united", "new": "newcastle-united", "nfo": "nottingham-forest",
    "sun": "sunderland", "tot": "tottenham-hotspur",
}
BBC_TEAM_FEED_URL = "https://feeds.bbci.co.uk/sport/football/teams/{slug}/rss.xml"
SKY_GENERAL_FEED_URL = "https://www.skysports.com/rss/12040"


def fetch_team_feed(team_short: str) -> list[dict]:
    """Fetches and caches one team's BBC Sport RSS feed, returning its raw entries."""
    slug = BBC_TEAM_SLUGS[team_short.lower()]
    parsed = feedparser.parse(BBC_TEAM_FEED_URL.format(slug=slug))
    entries = [{"title": e.get("title", ""), "summary": e.get("summary", ""), "published": e.get("published", "")} for e in parsed.entries]
    save_json(f"news_rss/bbc_{team_short.lower()}", entries)
    return entries


def fetch_general_feed() -> list[dict]:
    """Fetches and caches Sky Sports' general football RSS feed, returning its raw entries."""
    parsed = feedparser.parse(SKY_GENERAL_FEED_URL)
    entries = [{"title": e.get("title", ""), "summary": e.get("summary", ""), "published": e.get("published", "")} for e in parsed.entries]
    save_json("news_rss/sky_general", entries)
    return entries


def match_players_in_text(text: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """Finds whole-word surname matches for candidate players in a block of text, skipping surnames too short to be safe."""
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return candidates.iloc[0:0]
    matches = []
    for _, player in candidates.iterrows():
        surname = str(player["web_name"])
        if len(surname) < MIN_SURNAME_LENGTH:
            continue
        if re.search(rf"\b{re.escape(surname)}\b", text, re.IGNORECASE):
            matches.append(player)
    return pd.DataFrame(matches)


def build_news_items(players: pd.DataFrame) -> pd.DataFrame:
    """Pulls every team's BBC feed plus Sky's general feed and returns one row per matched player mention."""
    items = []
    for team_short in players["team_short"].str.lower().unique():
        team_players = players[players["team_short"].str.lower() == team_short]
        for entry in fetch_team_feed(team_short):
            text = f"{entry['title']} {entry['summary']}"
            for _, player in match_players_in_text(text, team_players).iterrows():
                items.append({"player_id": player["player_id"], "news": text.strip(), "news_added": entry["published"]})

    for entry in fetch_general_feed():
        text = f"{entry['title']} {entry['summary']}"
        for _, player in match_players_in_text(text, players).iterrows():
            items.append({"player_id": player["player_id"], "news": text.strip(), "news_added": entry["published"]})

    return pd.DataFrame(items, columns=["player_id", "news", "news_added"]).drop_duplicates()
