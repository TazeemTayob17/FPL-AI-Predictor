# FPL-branded styling: purple/green CSS injection and a pitch-style squad layout, reused across pages.

from __future__ import annotations

import pandas as pd
import streamlit as st

PURPLE = "#37003c"
GREEN = "#00ff85"
PINK = "#e90052"

POSITION_COLORS = {"GKP": "#f9a825", "DEF": "#1cade4", "MID": "#e90052", "FWD": "#00ff85"}
POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]


# Injects the purple/green FPL-style CSS (header banner, buttons, metric cards, sidebar). Call once per page.
def inject_fpl_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp header {{ background: transparent; }}

        h1 {{
            background: linear-gradient(90deg, {PURPLE} 0%, #63006e 100%);
            color: white !important;
            padding: 0.75rem 1.25rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }}

        [data-testid="stMetric"] {{
            background: white;
            border: 1px solid #e6e0ea;
            border-left: 4px solid {GREEN};
            border-radius: 10px;
            padding: 0.75rem 1rem;
            box-shadow: 0 1px 3px rgba(55, 0, 60, 0.08);
        }}
        [data-testid="stMetricLabel"] {{ color: {PURPLE}; font-weight: 600; }}

        .stButton > button, .stFormSubmitButton > button {{
            background: {GREEN};
            color: {PURPLE};
            font-weight: 700;
            border: none;
            border-radius: 999px;
            padding: 0.4rem 1.25rem;
        }}
        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            background: #00e077;
            color: {PURPLE};
        }}

        section[data-testid="stSidebar"] {{
            background: {PURPLE};
        }}
        section[data-testid="stSidebar"] * {{ color: white !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Picks whichever predicted-points column is present so the pitch works for both live and horizon predictions.
def points_col_for(df: pd.DataFrame) -> str:
    return "horizon_points" if "horizon_points" in df.columns else "predicted_points"


# Renders one player as a shirt-style card: position-colored top bar, name, team/price, predicted points.
def _player_card_html(row: pd.Series, points_col: str, badge: str | None = None) -> str:
    position = row.get("position", "")
    color = POSITION_COLORS.get(position, "#999999")
    points = row.get(points_col, 0)
    price = row.get("now_cost_million", 0)
    badge_html = (
        f'<span style="position:absolute; top:4px; right:6px; background:{PINK}; color:white; '
        f'font-size:0.65rem; font-weight:700; border-radius:50%; width:18px; height:18px; '
        f'display:flex; align-items:center; justify-content:center;">{badge}</span>'
        if badge else ""
    )
    card_style = "position:relative; background:white; border-radius:8px; flex:1 1 100px; min-width:90px; max-width:130px; box-shadow:0 2px 5px rgba(0,0,0,0.25); overflow:hidden;"
    name_style = f"font-weight:700; font-size:0.85rem; color:{PURPLE}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
    return (
        f'<div style="{card_style}">{badge_html}'
        f'<div style="background:{color}; height:6px;"></div>'
        f'<div style="padding:6px 8px;">'
        f'<div style="{name_style}">{row.get("web_name", "")}</div>'
        f'<div style="font-size:0.7rem; color:#666;">{row.get("team_short", "")} &middot; £{price:.1f}m</div>'
        f'<div style="font-size:0.75rem; font-weight:700; color:{PURPLE};">{points:.1f} pts</div>'
        f'</div></div>'
    )


# Lays out one position row (or the bench) as a flexbox of player cards, wrapping on narrow (phone) screens.
def _row_html(players: pd.DataFrame, points_col: str, captain_name: str | None, vice_captain_name: str | None) -> str:
    cards = []
    for _, player in players.sort_values(points_col, ascending=False).iterrows():
        badge = "C" if player.get("web_name") == captain_name else ("V" if player.get("web_name") == vice_captain_name else None)
        cards.append(_player_card_html(player, points_col, badge))
    return f'<div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-bottom:10px;">{"".join(cards)}</div>'


# Renders the starting XI as a green pitch (GKP at the bottom, FWD at the top) plus a bench strip below.
def render_pitch(starting_xi: pd.DataFrame, bench: pd.DataFrame, captain_name: str | None = None, vice_captain_name: str | None = None) -> None:
    points_col = points_col_for(starting_xi)
    pitch_rows = [p for p in reversed(POSITION_ORDER) if p in starting_xi["position"].values]
    rows_html = "".join(
        _row_html(starting_xi[starting_xi["position"] == position], points_col, captain_name, vice_captain_name)
        for position in pitch_rows
    )
    pitch_html = (
        '<div style="background: repeating-linear-gradient(180deg, #1f8a3e, #1f8a3e 40px, '
        f'#249147 40px, #249147 80px); border-radius:12px; padding:16px 10px;">{rows_html}</div>'
    )
    st.markdown(pitch_html, unsafe_allow_html=True)

    st.caption("Bench")
    bench_html = (
        f'<div style="background:#e6e0ea; border-radius:12px; padding:12px 10px;">'
        f'{_row_html(bench, points_col_for(bench), None, None)}</div>'
    )
    st.markdown(bench_html, unsafe_allow_html=True)
