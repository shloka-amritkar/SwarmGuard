"""Design tokens for the Streamlit app ("Ground Truth", docs/phases/phase-7.md).

COLOURS are NOT defined here. They come from core/palette.py (PALETTE), the
one source of truth shared with the PDF report; this module re-exports it.
This module adds only the non-colour tokens: fonts, radii, outline widths,
shadows, motion timings and the type scale.

HOW THE CSS GETS ITS COLOURS
assets/style.css never contains a hex value. It uses CSS custom properties
such as var(--sg-olive). css_variables() generates those properties from
PALETTE and TOKENS, and stylesheet() puts them in front of style.css:
    :root { --sg-olive: #3F4F24; --sg-radius-card: 14px; ... }
WHY: change a colour in core/palette.py and the app, the PDF, the charts
and the hero all follow, with no second copy to forget.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

from core.palette import PALETTE, Palette, contrast_ratio, hex_to_rgb01

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
STYLE_PATH = ASSETS_DIR / "style.css"

# Fonts (all SIL Open Font License): Google Fonts first, then system
# fallbacks so the app still reads well offline.
#   Fraunces       titles and the hero; "opsz" lets it adapt to its size
#   IBM Plex Sans  body, navigation, tables, buttons
#   IBM Plex Mono  numbers, coordinates, hashes, case numbers, eyebrows
FONT_SERIF = "'Fraunces', Georgia, 'Times New Roman', serif"
FONT_SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace"
GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Fraunces:ital,opsz,wght@0,9..144,500..600;1,9..144,500..600"
    "&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=IBM+Plex+Mono:wght@400;500"
    "&display=swap"
)


@dataclass(frozen=True)
class Tokens:
    """Non-colour design tokens (pixels and milliseconds unless noted)."""

    # Shapes
    radius_card: int = 14
    radius_button: int = 8
    radius_input: int = 8
    outline_card: float = 1.5
    outline_button: float = 1.5
    shadow_button: int = 3          # 3px 3px 0 espresso
    shadow_feature: int = 4         # 4px 4px 0 espresso
    corner: int = 10                # crosshair corner tick length (motif M2)
    navbar_height: int = 64
    # Spacing (8px grid)
    grid: int = 8
    gutter: int = 24
    section_gap: int = 56
    max_width: int = 1320
    # Motion
    ease: str = "cubic-bezier(0.2, 0.8, 0.2, 1)"
    dur_press: int = 90
    dur_hover: int = 140
    dur_enter: int = 240
    dur_stamp: int = 280
    dur_draw: int = 400
    # Type scale
    size_hero: int = 60
    size_hero_small: int = 38       # hero below 900px width
    size_title: int = 38
    size_section: int = 20
    size_eyebrow: int = 11
    size_body: int = 15
    size_table: int = 14
    size_caption: int = 13
    size_data: int = 13
    size_kpi: int = 34


TOKENS = Tokens()

STATUS_LABELS: dict[str, str] = {
    "normal": "NORMAL",
    "suspicious": "SUSPICIOUS",
    "alarm": "ALARM",
    "compromised": "COMPROMISED",
    "no_evidence": "NO EVIDENCE",
    "insufficient_evidence": "INSUFFICIENT EVIDENCE",
    "pass": "PASS",
    "fail": "FAIL",
    "warn": "WARN",
    "unknown": "UNKNOWN",
}
DETECTOR_NAMES: dict[str, str] = {
    "A": "Relative geometry",
    "B": "Motion consistency",
    "C": "Absolute anchors",
    "D": "Experimental ML",
    "F": "Fusion",
}


def status_label(status: str) -> str:
    """Text label for a status ("no_evidence" -> "NO EVIDENCE"). Status is never colour alone."""
    key = str(status).lower()
    return STATUS_LABELS.get(key, key.replace("_", " ").upper())


def status_tone(status: str) -> str:
    """CSS modifier for a status: normal | suspicious | alarm | none | pass | fail | warn.

    Verdicts map onto the four status colours (compromised -> alarm,
    insufficient_evidence -> none); verification results keep their own tone.
    """
    key = str(status).lower()
    table = {"normal": "normal", "suspicious": "suspicious", "alarm": "alarm",
             "compromised": "alarm", "no_evidence": "none", "insufficient_evidence": "none",
             "pass": "pass", "fail": "fail", "warn": "warn", "unknown": "none"}
    return table.get(key, "none")


def rgba(hex_colour: str, alpha: float) -> str:
    """"#3F4F24", 0.2 -> "rgba(63,79,36,0.2)" (for Plotly bands and map layers)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    r, g, b = (round(c * 255) for c in hex_to_rgb01(hex_colour))
    return f"rgba({r},{g},{b},{alpha:g})"


def _var_name(field_name: str) -> str:
    """"char_line" -> "--sg-char-line"."""
    return "--sg-" + field_name.replace("_", "-")


def css_variables(palette: Palette = PALETTE, tokens: Tokens = TOKENS) -> str:
    """A ":root { ... }" block with every colour and token as a CSS custom property."""
    lines = [f"  {_var_name(f.name)}: {getattr(palette, f.name)};" for f in fields(palette)]
    for f in fields(tokens):
        value = getattr(tokens, f.name)
        unit = "ms" if f.name.startswith("dur_") else ("" if f.name == "ease" else "px")
        lines.append(f"  {_var_name(f.name)}: {value}{unit};")
    lines += [f"  --sg-font-serif: {FONT_SERIF};", f"  --sg-font-sans: {FONT_SANS};",
              f"  --sg-font-mono: {FONT_MONO};", f"  --sg-contour-bg: {contour_background()};",
              f"  --sg-corners-ink: {corner_marks(palette.espresso)};",
              f"  --sg-corners-panel: {corner_marks(palette.char_line)};",
              f"  --sg-corners-amber: {corner_marks(palette.amber)};"]
    return ":root {\n" + "\n".join(lines) + "\n}\n"


def _contour_ring(cx: float, cy: float, r0: float, phase: float, n: int = 40) -> str:
    """One closed, gently wobbling contour line as an SVG path "M x y L ... Z".

        r(theta) = r0 * (1 + 0.10 sin(3 theta + phase) + 0.05 sin(5 theta + 2 phase))
        point    = (cx + r cos theta,  cy + r sin theta)

    Two sine waves of different frequency make the ring look like a natural
    height line on a survey map instead of a perfect circle.
    """
    pts = []
    for k in range(n):
        t = 2.0 * math.pi * k / n
        r = r0 * (1.0 + 0.10 * math.sin(3.0 * t + phase) + 0.05 * math.sin(5.0 * t + 2.0 * phase))
        pts.append(f"{cx + r * math.cos(t):.0f} {cy + r * math.sin(t):.0f}")
    return "M" + " L".join(pts) + "Z"


def contour_background(palette: Palette = PALETTE) -> str:
    """Motif M1: faint topographic contour lines as a CSS url("data:image/svg+xml,...").

    Two "hills" of nested wobbling rings (see _contour_ring) on a 900 x 900
    tile that repeats behind the page, drawn thin in the "contour" colour at
    reduced opacity so the page stays calm (about 6 % visual weight).
    WHY generated here (not typed into style.css): the stroke colour comes
    from the palette, and style.css must contain no colour values.
    WHY "%23": inside a data URI the "#" of a hex colour must be escaped.
    """
    rings = [_contour_ring(260, 300, 26.0 * k, 0.4 * k) for k in range(1, 10)]
    rings += [_contour_ring(700, 690, 22.0 * k, 1.3 + 0.35 * k) for k in range(1, 8)]
    colour = palette.contour.replace("#", "%23")
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='900' height='900' viewBox='0 0 900 900'>"
           f"<g fill='none' stroke='{colour}' stroke-width='1' stroke-opacity='0.8'>"
           f"<path d='{' '.join(rings)}'/></g></svg>")
    return f'url("data:image/svg+xml,{svg}")'


def corner_marks(colour: str, size: int = 10, width: float = 1.5) -> str:
    """Motif M2: four L-shaped registration ticks as four CSS background layers.

    Each corner is a tiny size x size SVG holding one "L" (two strokes);
    the four layers are pinned to the four corners of a pseudo-element
    that sits just outside a card or panel:
        url(top-left L) top left / 10px 10px no-repeat, url(top-right L) ...
    WHY SVG layers and not borders: one element has only one border box, so
    borders could draw at most one continuous frame, never four separate
    corner ticks.
    """
    c = colour.replace("#", "%23")
    h = width / 2
    paths = {
        "top left": f"M{h} {size} V{h} H{size}",
        "top right": f"M0 {h} H{size - h} V{size}",
        "bottom left": f"M{h} 0 V{size - h} H{size}",
        "bottom right": f"M0 {size - h} H{size - h} V0",
    }
    layers = []
    for position, d in paths.items():
        svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}'>"
               f"<path d='{d}' fill='none' stroke='{c}' stroke-width='{width}'/></svg>")
        layers.append(f'url("data:image/svg+xml,{svg}") {position} / {size}px {size}px no-repeat')
    return ", ".join(layers)


@lru_cache(maxsize=1)
def stylesheet() -> str:
    """The full CSS: font import, generated variables, then assets/style.css.

    WHY the @import comes first: CSS ignores @import rules that appear after
    any other rule. Cached because the files do not change while the app runs.
    """
    if not STYLE_PATH.exists():
        raise ValueError(f"stylesheet missing: {STYLE_PATH}")
    return f"@import url('{GOOGLE_FONTS_URL}');\n{css_variables()}\n{STYLE_PATH.read_text(encoding='utf-8')}"


def inject_css() -> None:
    """Add the SwarmGuard stylesheet to the current page (call once per run, in app.py)."""
    import streamlit as st  # WHY local: keeps this module importable without Streamlit

    st.markdown(f"<style>{stylesheet()}</style>", unsafe_allow_html=True)


__all__ = [
    "DETECTOR_NAMES",
    "FONT_MONO",
    "FONT_SANS",
    "FONT_SERIF",
    "GOOGLE_FONTS_URL",
    "PALETTE",
    "STATUS_LABELS",
    "TOKENS",
    "Palette",
    "Tokens",
    "contour_background",
    "corner_marks",
    "contrast_ratio",
    "css_variables",
    "hex_to_rgb01",
    "inject_css",
    "rgba",
    "status_label",
    "status_tone",
    "stylesheet",
]
