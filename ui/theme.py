"""Design tokens for the Streamlit app ("Ivory + Console", docs/phases/phase-5.md).

COLOURS are NOT defined here. They come from core/palette.py (PALETTE), the
one source of truth shared with the PDF report; this module re-exports it.
This module adds only the non-colour tokens: fonts, radii, outline widths,
shadows, motion timings and the type scale.

HOW THE CSS GETS ITS COLOURS
assets/style.css never contains a hex value. It uses CSS custom properties
such as var(--sg-forest). css_variables() generates those properties from
PALETTE and TOKENS, and stylesheet() puts them in front of style.css:
    :root { --sg-forest: #0B4D40; --sg-radius-card: 16px; ... }
WHY: change a colour in core/palette.py and the app, the PDF, the charts
and the hero all follow, with no second copy to forget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

from core.palette import PALETTE, Palette, contrast_ratio, hex_to_rgb01

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
STYLE_PATH = ASSETS_DIR / "style.css"

# Fonts: Google Fonts first, then system fallbacks so the app still reads well offline.
FONT_SERIF = "'Instrument Serif', Georgia, 'Times New Roman', serif"
FONT_DISPLAY = "'Bricolage Grotesque', 'Inter', system-ui, sans-serif"
FONT_SANS = "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace"
GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Instrument+Serif:ital@0;1"
    "&family=Bricolage+Grotesque:opsz,wght@12..96,600..800"
    "&family=Inter:wght@300;400;500;600"
    "&family=JetBrains+Mono:wght@400;500"
    "&display=swap"
)


@dataclass(frozen=True)
class Tokens:
    """Non-colour design tokens (pixels and milliseconds unless noted)."""

    # Shapes
    radius_card: int = 16
    radius_button: int = 10
    radius_input: int = 10
    outline_card: float = 1.5
    outline_button: float = 2.0
    shadow_button: int = 3          # 3px 3px 0 ink
    shadow_feature: int = 4         # 4px 4px 0 ink
    # Spacing (8px grid)
    grid: int = 8
    gutter: int = 24
    section_gap: int = 48
    max_width: int = 1360
    # Motion
    ease: str = "cubic-bezier(0.2, 0.8, 0.2, 1)"
    dur_press: int = 90
    dur_hover: int = 140
    dur_enter: int = 240
    dur_draw: int = 400
    # Type scale
    size_hero: int = 64
    size_hero_small: int = 40       # hero below 900px width
    size_title: int = 40
    size_section: int = 22
    size_eyebrow: int = 11
    size_body: int = 15
    size_table: int = 14
    size_data: int = 13


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
    """"#0B4D40", 0.2 -> "rgba(11,77,64,0.2)" (for Plotly bands and map layers)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    r, g, b = (round(c * 255) for c in hex_to_rgb01(hex_colour))
    return f"rgba({r},{g},{b},{alpha:g})"


def _var_name(field_name: str) -> str:
    """"console_ln" -> "--sg-console-ln"."""
    return "--sg-" + field_name.replace("_", "-")


def css_variables(palette: Palette = PALETTE, tokens: Tokens = TOKENS) -> str:
    """A ":root { ... }" block with every colour and token as a CSS custom property."""
    lines = [f"  {_var_name(f.name)}: {getattr(palette, f.name)};" for f in fields(palette)]
    for f in fields(tokens):
        value = getattr(tokens, f.name)
        unit = "ms" if f.name.startswith("dur_") else ("" if f.name == "ease" else "px")
        lines.append(f"  {_var_name(f.name)}: {value}{unit};")
    lines += [f"  --sg-font-serif: {FONT_SERIF};", f"  --sg-font-display: {FONT_DISPLAY};",
              f"  --sg-font-sans: {FONT_SANS};", f"  --sg-font-mono: {FONT_MONO};"]
    return ":root {\n" + "\n".join(lines) + "\n}\n"


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
    "FONT_DISPLAY",
    "FONT_MONO",
    "FONT_SANS",
    "FONT_SERIF",
    "GOOGLE_FONTS_URL",
    "PALETTE",
    "STATUS_LABELS",
    "TOKENS",
    "Palette",
    "Tokens",
    "contrast_ratio",
    "css_variables",
    "hex_to_rgb01",
    "inject_css",
    "rgba",
    "status_label",
    "status_tone",
    "stylesheet",
]
