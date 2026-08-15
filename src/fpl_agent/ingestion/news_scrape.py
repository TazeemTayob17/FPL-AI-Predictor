"""Rate-limited, cached fallback scraper for Fantasy Football Scout's injury table, filling gaps RSS misses."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

from fpl_agent.ingestion.cache import RAW_DIR

INJURIES_URL = "https://www.fantasyfootballscout.co.uk/fantasy-football-injuries"
USER_AGENT = "FPL-Agent-Personal/1.0 (private single-user FPL decision-support tool; contact: tazeemtayob17@gmail.com)"
MIN_REFETCH_INTERVAL_HOURS = 3.0

STATUS_MAP = {"injured": "i", "suspended": "s", "doubt-25": "d", "doubt-50": "d", "doubt-75": "d"}
CHANCE_MAP = {"injured": 0, "suspended": 0, "doubt-25": 25, "doubt-50": 50, "doubt-75": 75}


def fetch_injuries_page(min_interval_hours: float = MIN_REFETCH_INTERVAL_HOURS) -> str:
    """Returns the cached injuries page if it's fresher than min_interval_hours, otherwise fetches and caches a new copy."""
    cache_dir = RAW_DIR / "scrape" / "ffscout_injuries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_dir.glob("*.html"))
    if existing and (time.time() - existing[-1].stat().st_mtime) / 3600 < min_interval_hours:
        return existing[-1].read_text(encoding="utf-8")

    response = requests.get(INJURIES_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
    (cache_dir / f"{timestamp}.html").write_text(response.text, encoding="utf-8")
    return response.text


def _match_by_name_words(scraped_name: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """Matches a scraped name (word order and extra middle names vary) against candidates by word-set containment."""
    scraped_words = set(scraped_name.lower().split())
    for _, player in candidates.iterrows():
        player_words = set(str(player["web_name_full"]).lower().split())
        if player_words.issubset(scraped_words) or scraped_words.issubset(player_words):
            return candidates.loc[[player.name]]
    return candidates.iloc[0:0]


def parse_injuries(html: str, players: pd.DataFrame) -> pd.DataFrame:
    """Parses the injuries table into player_status-shaped rows, matched by team code plus full name."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for row in soup.select("tr.injuries-bans-item"):
        team_code = row.get("data-team-code", "")
        avatar = row.select_one("img[alt^='Avatar of ']")
        status_span = row.select_one("span.status")
        cells = row.select("td")
        if avatar is None or status_span is None or len(cells) < 5:
            continue

        full_name = avatar["alt"].removeprefix("Avatar of ").strip()
        status_class = next((c for c in status_span.get("class", []) if c != "status"), None)
        candidates = players[players["team_short"].str.lower() == team_code]
        match = _match_by_name_words(full_name, candidates)
        if match.empty:
            continue

        headline = cells[4].find("strong")
        rows.append(
            {
                "player_id": int(match.iloc[0]["player_id"]),
                "status": STATUS_MAP.get(status_class, "d"),
                "chance_of_playing_next_round": CHANCE_MAP.get(status_class, 50),
                "news": headline.get_text(strip=True) if headline else None,
            }
        )
    return pd.DataFrame(rows, columns=["player_id", "status", "chance_of_playing_next_round", "news"])
