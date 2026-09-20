"""
Design tokens and CSS injection for the dashboard — a fleet dispatcher's
command console, not a generic data-app template. Every color choice below
carries meaning: each optimization mode has ONE accent color used
consistently across its button, active state, and map route line, so color
itself is information (which objective is active) rather than decoration.
"""

import streamlit as st

COLORS = {
    "bg": "#10141A",
    "panel": "#1A212C",
    "panel_border": "#2A3342",
    "text": "#E7EAEE",
    "text_muted": "#8993A6",
    "fastest": "#F2A93B",
    "eco": "#3FBE8E",
    "balanced": "#5B8DEF",
    "emergency": "#E5484D",
    "route_default": "#34D1C6",
}

MODE_META = {
    "fastest": {"label": "Fastest", "color": COLORS["fastest"],
                "blurb": "Minimizes travel time. Congestion counts as a secondary cost."},
    "eco": {"label": "Eco", "color": COLORS["eco"],
            "blurb": "Minimizes fuel and emissions, even if the trip takes a bit longer."},
    "balanced": {"label": "Balanced", "color": COLORS["balanced"],
                 "blurb": "Even weight across time, distance, congestion, and emissions."},
    "emergency": {"label": "Emergency", "color": COLORS["emergency"],
                  "blurb": "Time above all else. Other traffic yields on the shared route."},
}


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {COLORS['bg']};
        color: {COLORS['text']};
    }}

    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        color: {COLORS['text']} !important;
        letter-spacing: -0.01em;
    }}

    [data-testid="stSidebar"] {{
        background-color: {COLORS['panel']};
        border-right: 1px solid {COLORS['panel_border']};
    }}

    .console-panel {{
        background-color: {COLORS['panel']};
        border: 1px solid {COLORS['panel_border']};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }}

    .stat-tile {{
        background-color: {COLORS['panel']};
        border-left: 3px solid {COLORS['route_default']};
        border-radius: 6px;
        padding: 0.7rem 1rem;
    }}
    .stat-tile .stat-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: {COLORS['text_muted']};
        margin-bottom: 0.2rem;
    }}
    .stat-tile .stat-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: {COLORS['text']};
    }}

    .mode-card {{
        border: 1px solid {COLORS['panel_border']};
        border-left: 4px solid var(--mode-color);
        border-radius: 8px;
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.5rem;
    }}
    .mode-card .mode-name {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: {COLORS['text']};
    }}
    .mode-card .mode-blurb {{
        font-size: 0.8rem;
        color: {COLORS['text_muted']};
        margin-top: 0.1rem;
    }}

    .stButton > button {{
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 7px;
        border: 1px solid {COLORS['panel_border']};
    }}
    .stButton > button[kind="primary"] {{
        background-color: {COLORS['route_default']};
        border: none;
        color: #0A0E13;
    }}

    .confidence-badge {{
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
    }}

    [data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace;
    }}
    </style>
    """, unsafe_allow_html=True)


def mode_card_html(mode: str, selected: bool) -> str:
    meta = MODE_META[mode]
    ring = f"box-shadow: 0 0 0 1px {meta['color']};" if selected else ""
    return f"""
    <div class="mode-card" style="--mode-color: {meta['color']}; {ring}">
        <div class="mode-name">{meta['label']}</div>
        <div class="mode-blurb">{meta['blurb']}</div>
    </div>
    """


def stat_tile_html(label: str, value: str, color: str = None) -> str:
    border_color = color or COLORS["route_default"]
    return f"""
    <div class="stat-tile" style="border-left-color: {border_color};">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """


def confidence_badge_html(confidence: float) -> str:
    if confidence >= 0.8:
        color, bg = "#3FBE8E", "rgba(63,190,142,0.15)"
        text = "High confidence"
    elif confidence >= 0.5:
        color, bg = "#F2A93B", "rgba(242,169,59,0.15)"
        text = "Moderate confidence"
    else:
        color, bg = "#E5484D", "rgba(229,72,77,0.15)"
        text = "Low confidence"
    return (f'<span class="confidence-badge" style="color:{color}; background:{bg};">'
            f'{text} · {confidence*100:.0f}%</span>')
