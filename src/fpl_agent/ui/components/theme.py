# FPL-branded design system: fonts, spacing, color tokens, and a pitch-style squad layout, reused across every page.

from __future__ import annotations

import pandas as pd
import streamlit as st

PURPLE = "#37003c"
PURPLE_LIGHT = "#6b2f74"
GREEN = "#00ff85"
PINK = "#e90052"
INK = "#1f1029"
MUTED = "#7a6f82"
SURFACE = "#faf8fb"
BORDER = "#e8e1ec"

POSITION_COLORS = {"GKP": "#f9a825", "DEF": "#1cade4", "MID": "#e90052", "FWD": "#00d97e"}
POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]


# Injects the full design system - fonts, layout width, sidebar, headers, cards, buttons, tables. Call once per page, right after st.set_page_config.
def inject_fpl_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{ background: {SURFACE}; }}
        .stApp header {{ background: transparent; }}

        .block-container {{
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        h1 {{
            background: linear-gradient(120deg, {PURPLE} 0%, {PURPLE_LIGHT} 100%);
            color: white !important;
            padding: 1.1rem 1.5rem;
            border-radius: 14px;
            margin-bottom: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            box-shadow: 0 4px 16px rgba(55, 0, 60, 0.18);
        }}

        h2, h3 {{
            color: {INK};
            font-weight: 700;
            letter-spacing: -0.01em;
            padding-left: 0.7rem;
            border-left: 4px solid {GREEN};
            margin-top: 1.75rem !important;
        }}

        p, span, div, label {{ color: {INK}; }}
        [data-testid="stCaptionContainer"], .stCaption {{ color: {MUTED} !important; }}

        [data-testid="stMetric"] {{
            background: white;
            border: 1px solid {BORDER};
            border-left: 4px solid {GREEN};
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 2px 8px rgba(55, 0, 60, 0.06);
        }}
        [data-testid="stMetricLabel"] {{ color: {PURPLE}; font-weight: 600; font-size: 0.85rem; }}
        [data-testid="stMetricValue"] {{ color: {INK}; font-weight: 800; }}

        .stButton > button, .stFormSubmitButton > button {{
            background: {GREEN};
            color: {PURPLE};
            font-weight: 700;
            border: none;
            border-radius: 999px;
            padding: 0.45rem 1.4rem;
            transition: transform 0.08s ease, box-shadow 0.08s ease;
            box-shadow: 0 2px 6px rgba(0, 255, 133, 0.35);
        }}
        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            background: #00e077;
            color: {PURPLE};
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(0, 255, 133, 0.45);
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(55, 0, 60, 0.05);
        }}

        div[data-baseweb="notification"] {{ border-radius: 12px; }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {PURPLE} 0%, #2a002e 100%);
        }}
        section[data-testid="stSidebar"] * {{ color: white !important; }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
            border-radius: 8px;
            margin: 0.1rem 0.5rem;
            padding: 0.55rem 0.75rem;
            font-weight: 500;
            opacity: 0.85;
            transition: background 0.1s ease, opacity 0.1s ease;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
            background: rgba(255, 255, 255, 0.08);
            opacity: 1;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(0, 255, 133, 0.16);
            opacity: 1;
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Picks whichever predicted-points column is present so the pitch works for both live and horizon predictions.
def points_col_for(df: pd.DataFrame) -> str:
    return "horizon_points" if "horizon_points" in df.columns else "predicted_points"


# Renders one player as a shirt-style card: position-colored top bar and label, name, team/price, predicted points.
def _player_card_html(row: pd.Series, points_col: str, badge: str | None = None) -> str:
    position = row.get("position", "")
    color = POSITION_COLORS.get(position, "#999999")
    points = row.get(points_col, 0)
    price = row.get("now_cost_million", 0)
    badge_html = (
        f'<span style="position:absolute; top:6px; right:8px; background:{PINK}; color:white; '
        f'font-size:0.65rem; font-weight:800; border-radius:50%; width:20px; height:20px; '
        f'box-shadow:0 1px 3px rgba(0,0,0,0.3); display:flex; align-items:center; justify-content:center;">{badge}</span>'
        if badge else ""
    )
    card_style = (
        "position:relative; background:white; border-radius:10px; flex:1 1 108px; min-width:96px; max-width:140px; "
        "box-shadow:0 3px 10px rgba(0,0,0,0.22); overflow:hidden;"
    )
    name_style = f"font-weight:700; font-size:0.85rem; color:{PURPLE}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
    return (
        f'<div style="{card_style}">{badge_html}'
        f'<div style="background:{color}; height:7px;"></div>'
        f'<div style="padding:8px 9px 9px;">'
        f'<div style="font-size:0.6rem; font-weight:700; color:{color}; letter-spacing:0.04em;">{position}</div>'
        f'<div style="{name_style}">{row.get("web_name", "")}</div>'
        f'<div style="font-size:0.7rem; color:#666; margin-top:1px;">{row.get("team_short", "")} &middot; £{price:.1f}m</div>'
        f'<div style="font-size:0.8rem; font-weight:800; color:{PURPLE}; margin-top:3px;">{points:.1f} <span style="font-weight:500; font-size:0.65rem;">pts</span></div>'
        f'</div></div>'
    )


# Lays out one position row (or the bench) as a flexbox of player cards, wrapping on narrow (phone) screens.
def _row_html(players: pd.DataFrame, points_col: str, captain_name: str | None, vice_captain_name: str | None) -> str:
    cards = []
    for _, player in players.sort_values(points_col, ascending=False).iterrows():
        badge = "C" if player.get("web_name") == captain_name else ("V" if player.get("web_name") == vice_captain_name else None)
        cards.append(_player_card_html(player, points_col, badge))
    return f'<div style="display:flex; flex-wrap:wrap; gap:12px; justify-content:center; margin-bottom:22px; position:relative; z-index:1;">{"".join(cards)}</div>'


# Renders the starting XI as a green pitch (GKP at the bottom, FWD at the top, with markings) plus a bench strip below.
def render_pitch(starting_xi: pd.DataFrame, bench: pd.DataFrame, captain_name: str | None = None, vice_captain_name: str | None = None) -> None:
    points_col = points_col_for(starting_xi)
    pitch_rows = [p for p in reversed(POSITION_ORDER) if p in starting_xi["position"].values]
    rows_html = "".join(
        _row_html(starting_xi[starting_xi["position"] == position], points_col, captain_name, vice_captain_name)
        for position in pitch_rows
    )
    markings_html = (
        '<div style="position:absolute; inset:0; pointer-events:none;">'
        '<div style="position:absolute; top:50%; left:0; right:0; height:2px; background:rgba(255,255,255,0.28);"></div>'
        '<div style="position:absolute; top:50%; left:50%; width:120px; height:120px; margin:-60px 0 0 -60px; '
        'border:2px solid rgba(255,255,255,0.28); border-radius:50%;"></div>'
        '<div style="position:absolute; inset:10px; border:2px solid rgba(255,255,255,0.22); border-radius:8px;"></div>'
        "</div>"
    )
    pitch_html = (
        '<div style="position:relative; background: repeating-linear-gradient(180deg, #1f8a3e, #1f8a3e 40px, '
        f'#249147 40px, #249147 80px); border-radius:16px; padding:24px 14px; box-shadow:0 4px 18px rgba(0,0,0,0.15);">'
        f'{markings_html}{rows_html}</div>'
    )
    st.markdown(pitch_html, unsafe_allow_html=True)

    st.markdown(
        f'<div style="margin-top:1.5rem; margin-bottom:0.5rem; font-weight:700; color:{PURPLE}; '
        'font-size:0.8rem; letter-spacing:0.04em; text-transform:uppercase;">Bench</div>',
        unsafe_allow_html=True,
    )
    bench_html = (
        f'<div style="background:{BORDER}; border-radius:14px; padding:16px 14px;">'
        f'{_row_html(bench, points_col_for(bench), None, None)}</div>'
    )
    st.markdown(bench_html, unsafe_allow_html=True)
