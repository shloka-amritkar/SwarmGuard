"""The animated Home hero (docs/phases/phase-5.md section 5.6).

assets/hero.html is a template with __PLACEHOLDERS__; hero_html() fills them
and inserts the colour variables generated from core/palette.py, so the
hero never carries its own copy of a colour.

WHY an iframe (st.iframe): st.html sanitises HTML, which strips <script>
and can strip SVG animation elements. An iframe renders the hero exactly as
written, with its own fonts. (st.iframe replaces the older
streamlit.components.v1.html, which Streamlit has deprecated.) Only our own
template is ever passed to it, never user input: an iframe runs JavaScript.
The iframe cannot press Streamlit buttons, so the real call-to-action
buttons are placed directly under it; the hero's own button scrolls to them.

The telemetry and ribbon texts are illustrative examples of what the
platform shows (labelled as simulation output on the real pages).
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

from ui.theme import GOOGLE_FONTS_URL, css_variables

logger = logging.getLogger(__name__)

HERO_PATH = Path(__file__).resolve().parent.parent / "assets" / "hero.html"
DEFAULT_HEIGHT = 660
N_BARS = 22
SPIKE_BAR = 13

EYEBROW = "SwarmGuard · GPS integrity for drone swarms"
HEADLINE = ("Detect the drift,", "prove the attack.")
SUBTITLE = ("Simulates GPS spoofing on a drone swarm, catches it with three independent "
            "detectors, and seals the evidence.")
TELEMETRY = (
    "GPS 18.5204N 73.8567E · NIS 2.1 · range 7.98 m · anchor residual 0.6 m "
    "· CUSUM 0.0 · vote 1/9 · GPS 18.5205N 73.8569E · NIS 1.7 · "
    "range 8.04 m · velocity 2.01 m/s · NIS 48.2 · CUSUM 31.5 · "
    "anchor residual 10.2 m · manifest signed"
)
RIBBON_EVENTS: tuple[tuple[str, str], ...] = (
    ("Drone 04", "motion check · ALARM · t = 20.3 s"),
    ("swarm pattern", "A silent, B and C alarm"),
    ("Drone 07", "anchor check · ALARM · t = 25.2 s"),
    ("recovery", "common offset 10.10 m east"),
    ("evidence", "manifest signed · Ed25519"),
    ("verification", "all checks PASS"),
)


def _bars() -> str:
    """N_BARS waveform bars; one of them is the occasional red "detection" spike."""
    out = []
    for i in range(N_BARS):
        cls = "bar bar--spike" if i == SPIKE_BAR else "bar"
        out.append(f'<span class="{cls}" style="--i:{i}"></span>')
    return "".join(out)


def _ribbon() -> str:
    """Ribbon items "<b>Drone 04</b> motion check ..." separated by dots."""
    parts = [f"<span><b>{html.escape(who)}</b> · {html.escape(what)}</span>" for who, what in RIBBON_EVENTS]
    return '<span class="sep">···</span>'.join(parts)


def hero_html() -> str:
    """The complete hero document with every placeholder filled."""
    if not HERO_PATH.exists():
        raise ValueError(f"hero template missing: {HERO_PATH}")
    replacements = {
        "/*SG_VARS*/": css_variables(),
        "__FONTS_URL__": html.escape(GOOGLE_FONTS_URL, quote=True),
        "__EYEBROW__": html.escape(EYEBROW),
        "__HEADLINE_1__": html.escape(HEADLINE[0]),
        "__HEADLINE_2__": html.escape(HEADLINE[1]),
        "__SUBTITLE__": html.escape(SUBTITLE),
        "__TELEMETRY__": html.escape(TELEMETRY),
        "__BARS__": _bars(),
        "__RIBBON__": _ribbon(),
    }
    page = HERO_PATH.read_text(encoding="utf-8")
    for key, value in replacements.items():
        if key not in page:
            raise ValueError(f"hero template has no placeholder {key}")
        page = page.replace(key, value)
    return page


def render_hero(height: int = DEFAULT_HEIGHT) -> None:
    """Show the hero in an iframe with a transparent background."""
    import streamlit as st  # WHY local: keeps hero_html() testable without Streamlit

    st.iframe(hero_html(), height=height, width="stretch")
