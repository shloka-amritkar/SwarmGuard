"""The animated Home hero ("Ground Truth", docs/phases/phase-7.md B8).

assets/hero.html is a template with __PLACEHOLDERS__; hero_html() fills them
and inserts the colour variables generated from core/palette.py, so the
hero never carries its own copy of a colour.

WHAT IT SHOWS
  Left:  eyebrow, the headline "Where the drones / really are." and a
         two-line subtitle.
  Right: an amber radar scope. A sweep turns every 8 s; ten drone blips
         brighten as it passes. Every 10 s one blip splits into its TRUE
         position (filled) and its SPOOFED GPS position (hollow), joined by
         a growing dashed line; "ALARM · motion check" appears, holds and
         fades; the NIS readout jumps 2.1 -> 48.6; the blip rejoins.
  Around both: coordinate rulers (18.510°N … 18.530°N, 73.845°E … 73.865°E).
The geometry (ruler ticks, bearing ticks, wedges, blips, the spoof event) is
computed here in Python, so the maths is in one readable place:
  a point at bearing b (degrees clockwise from north) and radius r is
      x = r sin(b),  y = -r cos(b)      (SVG's y axis points down)

WHY an iframe (st.iframe): st.html sanitises HTML, which strips <script>
and SVG animation elements. An iframe renders the hero exactly as written,
with its own fonts. Only our own template is ever passed to it, never user
input: an iframe runs JavaScript. The iframe cannot press Streamlit buttons,
so the real call-to-action buttons are placed directly under it.

The radar is an illustration of what the platform does (a motion-check
alarm on one spoofed drone), labelled "simulation" inside the panel.
"""

from __future__ import annotations

import html
import json
import logging
import math
from pathlib import Path

from ui.theme import GOOGLE_FONTS_URL, css_variables

logger = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
HERO_PATH = ASSETS / "hero.html"
TEST_COUNT_PATH = ASSETS / "test_count.json"
DEFAULT_HEIGHT = 600

EYEBROW = "SwarmGuard · GPS integrity & digital forensics"
HEADLINE = ("Where the drones", "really are.")
SUBTITLE = ("Simulates GPS spoofing on a drone swarm, catches it with three independent "
            "detectors, and seals the evidence so it can't be quietly changed.")

LAT_RANGE = (18.510, 18.530)       # left ruler, bottom -> top
LON_RANGE = (73.845, 73.865)       # top ruler, left -> right
RULER_STEPS = 20                   # a tick every 0.001 degrees; labels every 5th

SWEEP_SECONDS = 8.0
EVENT_SECONDS = 10.0
SCOPE_R = 180.0
FORMATION_CENTRE = (34.0, -26.0)   # where the swarm sits on the scope (SVG units)
FORMATION_R = 34.0
SPOOFED_BLIP = 6                   # which of the ten blips is spoofed (upper left of the ring)
SPOOF_OFFSET = (-52.0, -58.0)      # how far its GPS position is pushed (SVG units)


def _fmt(v: float) -> str:
    """Short number for SVG attributes."""
    return f"{v:.1f}".rstrip("0").rstrip(".")


def polar(bearing_deg: float, radius: float) -> tuple[float, float]:
    """(x, y) on the scope for a bearing (degrees clockwise from north) and a radius."""
    b = math.radians(bearing_deg)
    return radius * math.sin(b), -radius * math.cos(b)


def bearing_of(x: float, y: float) -> float:
    """Inverse of polar: bearing in [0, 360) of the point (x, y)."""
    return math.degrees(math.atan2(x, -y)) % 360.0


def ruler_html(axis: str) -> str:
    """Tick marks and labels for the top (longitude) or left (latitude) ruler."""
    if axis not in ("top", "left"):
        raise ValueError(f"axis must be 'top' or 'left', got {axis!r}")
    lo, hi = LON_RANGE if axis == "top" else LAT_RANGE
    parts = []
    for i in range(RULER_STEPS + 1):
        frac = i / RULER_STEPS
        major = i % 5 == 0
        # top ruler runs west -> east; left ruler runs north (top) -> south
        pos = 100.0 * frac if axis == "top" else 100.0 * (1.0 - frac)
        prop = "left" if axis == "top" else "top"
        parts.append(f'<span class="tick{" major" if major else ""}" style="{prop}:{pos:.2f}%"></span>')
        if major:
            # WHY: the first and last labels are aligned inwards so they are never cut off
            edge = " first" if i == 0 else (" last" if i == RULER_STEPS else "")
            value = lo + frac * (hi - lo)
            text = f"{value:.3f}°E" if axis == "top" else f"{value:.3f}°N"
            parts.append(f'<span class="label{edge}" style="{prop}:{pos:.2f}%">{text}</span>')
    return "".join(parts)


def bearings_svg() -> str:
    """Bearing ticks every 30 degrees outside the outer ring, labelled 000 ... 330."""
    parts = []
    for b in range(0, 360, 30):
        x1, y1 = polar(b, SCOPE_R)
        x2, y2 = polar(b, SCOPE_R + 7)
        lx, ly = polar(b, SCOPE_R + 16)
        parts.append(f'<line class="bearing" x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}"/>'
                     f'<text class="bearing-label" x="{_fmt(lx)}" y="{_fmt(ly + 3)}">{b:03d}</text>')
    return "".join(parts)


def wedge_path(start_deg: float, end_deg: float, radius: float = SCOPE_R) -> str:
    """SVG path of a pie slice from bearing start_deg to end_deg (end > start)."""
    x1, y1 = polar(start_deg, radius)
    x2, y2 = polar(end_deg, radius)
    large = 1 if end_deg - start_deg > 180 else 0
    return f"M0 0 L{_fmt(x1)} {_fmt(y1)} A{_fmt(radius)} {_fmt(radius)} 0 {large} 1 {_fmt(x2)} {_fmt(y2)} Z"


def blip_positions() -> list[tuple[float, float]]:
    """Ten drones in a circle formation, as (x, y) scope coordinates."""
    cx, cy = FORMATION_CENTRE
    return [(cx + FORMATION_R * math.cos(2 * math.pi * k / 10), cy + FORMATION_R * math.sin(2 * math.pi * k / 10))
            for k in range(10)]


def blips_svg() -> str:
    """The ten blips. Each brightens when the sweep reaches its bearing.

    The sweep starts pointing north and turns clockwise, so it reaches a
    blip at bearing b after t = b / 360 * 8 s. The blip's animation starts
    bright, so its delay is t - 8 s (negative: already running, in phase).
    """
    parts = []
    for k, (x, y) in enumerate(blip_positions()):
        if k == SPOOFED_BLIP:
            continue  # drawn by event_svg as the true dot
        delay = bearing_of(x, y) / 360.0 * SWEEP_SECONDS - SWEEP_SECONDS
        parts.append(f'<circle class="blip" cx="{_fmt(x)}" cy="{_fmt(y)}" r="4" style="animation-delay:{delay:.2f}s"/>')
    return "".join(parts)


# keyTimes of the 10 s spoof event (fractions of EVENT_SECONDS)
#   0.00-0.10 together | 0.10-0.25 GPS position drifts away (1.5 s)
#   0.25-0.60 hold (ALARM label from 0.25 to ~0.58) | 0.60-0.75 rejoin | rest together
_MOVE_TIMES = "0;0.10;0.25;0.60;0.75;1"


def _move(attr: str, start: float, end: float) -> str:
    """SMIL animation of one attribute from start out to end and back, on the event clock."""
    values = ";".join(_fmt(v) for v in (start, start, end, end, start, start))
    return (f'<animate attributeName="{attr}" dur="{EVENT_SECONDS:g}s" repeatCount="indefinite" '
            f'keyTimes="{_MOVE_TIMES}" values="{values}"/>')


def _fade(values: str, key_times: str, attr: str = "opacity") -> str:
    """SMIL opacity animation with its own keyTimes on the event clock."""
    return (f'<animate attributeName="{attr}" dur="{EVENT_SECONDS:g}s" repeatCount="indefinite" '
            f'keyTimes="{key_times}" values="{values}"/>')


def event_svg() -> str:
    """The spoof event: true dot, hollow spoofed dot, growing dashed line, ALARM label."""
    x, y = blip_positions()[SPOOFED_BLIP]
    sx, sy = x + SPOOF_OFFSET[0], y + SPOOF_OFFSET[1]
    true_delay = bearing_of(x, y) / 360.0 * SWEEP_SECONDS - SWEEP_SECONDS
    visible = _fade("0;0;1;1;0;0", _MOVE_TIMES)
    label = _fade("0;0;1;1;0;0", "0;0.25;0.28;0.55;0.62;1")
    lx, ly = sx - 75, sy - 34          # label centred above the spoofed dot, inside the scope
    return (
        f'<line class="spoof-line" x1="{_fmt(x)}" y1="{_fmt(y)}" x2="{_fmt(x)}" y2="{_fmt(y)}" opacity="0">'
        f'{_move("x2", x, sx)}{_move("y2", y, sy)}{visible}</line>'
        f'<circle class="blip true-dot" cx="{_fmt(x)}" cy="{_fmt(y)}" r="4" style="animation-delay:{true_delay:.2f}s"/>'
        f'<circle class="spoof-dot" cx="{_fmt(x)}" cy="{_fmt(y)}" r="5" opacity="0">'
        f'{_move("cx", x, sx)}{_move("cy", y, sy)}{visible}</circle>'
        f'<g opacity="0">{label}'
        f'<rect class="alarm-pill" x="{_fmt(lx)}" y="{_fmt(ly)}" width="150" height="18" rx="9"/>'
        f'<text class="alarm-text" x="{_fmt(lx + 75)}" y="{_fmt(ly + 12.5)}" text-anchor="middle">ALARM · motion check</text></g>'
    )


def nis_hot_svg() -> str:
    """The " → 48.6" part of the NIS readout, visible only during the event."""
    return f'<tspan class="hot" fill-opacity="0"> → 48.6{_fade("0;0;1;1;0;0", "0;0.2;0.22;0.6;0.62;1", "fill-opacity")}</tspan>'


def test_count() -> int | None:
    """The number of tests recorded by `python cli.py count-tests`, or None if never recorded."""
    try:
        data = json.loads(TEST_COUNT_PATH.read_text(encoding="utf-8"))
        return int(data["tests"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def stats_line() -> str:
    """'239 tests · 3 detectors · Ed25519-signed evidence' (falls back without a number)."""
    n = test_count()
    tests = f"{n} automated tests" if n else "Automated tests"
    return f"{tests} · 3 detectors · Ed25519-signed evidence"


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
        "__RULER_TOP__": ruler_html("top"),
        "__RULER_LEFT__": ruler_html("left"),
        "__BEARINGS__": bearings_svg(),
        "__WEDGE_1__": wedge_path(-12.0, 0.0),
        "__WEDGE_2__": wedge_path(-26.0, -12.0),
        "__WEDGE_3__": wedge_path(-44.0, -26.0),
        "__BLIPS__": blips_svg(),
        "__EVENT__": event_svg(),
        "__NIS_HOT__": nis_hot_svg(),
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
