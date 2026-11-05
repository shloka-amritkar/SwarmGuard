"""Survey-style line-art illustrations ("stickers"), docs/phases/phase-7.md B11.

Each illustration is drawn on a 64 x 64 grid with 1.5-unit espresso strokes
and flat fills in card, vellum, brass or sage (all from core/palette.py).
They look like the precise drawings in a surveyor's field manual: no tilt,
no faces, no mascots, only technical subjects (quadcopters, satellites, a
radar dish, a GNSS constellation, a shield, a padlock, a fingerprint, a
sealed document, a crosshair and a survey tripod).

Use them on Home, scenario cards, empty states and section headers; never
inside charts (charts must stay readable first).

    sticker("radar_dish", size=72) -> "<svg ...>...</svg>"

WHY SVG drawn in code and not image files: an SVG scales without blurring,
needs no file serving, and its colours come straight from the palette
module, so a palette change restyles every illustration.
WHY no tilt any more: Phase 5 tilted stickers for a hand-placed look; the
Ground Truth design is about measured precision, so drawings sit square.
The `rotate` argument is kept so existing callers still work, but only 0
is accepted (MAX_TILT_DEG = 0).
WHY aliases: older pages ask for "radar", "drone_top", "document" ...;
ALIASES maps those names to the redrawn subjects so no page breaks.
"""

from __future__ import annotations

import base64
import html
import logging

from core.palette import PALETTE

logger = logging.getLogger(__name__)

MAX_TILT_DEG = 0.0
STROKE_WIDTH = "1.5"

_INK = PALETTE.espresso
_CARD = PALETTE.card
_VELLUM = PALETTE.vellum
_BRASS = PALETTE.brass
_SAGE = PALETTE.sage


def _rotor(cx: float, cy: float, r: float = 8.5) -> str:
    """One rotor disc with a hub and a thin blade line (used by the quadcopters)."""
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{_VELLUM}"/>'
            f'<line x1="{cx - r + 2}" y1="{cy}" x2="{cx + r - 2}" y2="{cy}" stroke-width="1"/>'
            f'<circle cx="{cx}" cy="{cy}" r="1.6" fill="{_INK}" stroke="none"/>')


# Inner SVG markup of each illustration (viewBox 0 0 64 64). The root <svg>
# sets stroke = espresso, stroke-width = 1.5 and fill = none, so shapes only
# state a fill when they have one.
_SHAPES: dict[str, str] = {
    "quadcopter_top": (
        f'<line x1="16" y1="16" x2="48" y2="48"/><line x1="48" y1="16" x2="16" y2="48"/>'
        f'{_rotor(16, 16)}{_rotor(48, 16)}{_rotor(16, 48)}{_rotor(48, 48)}'
        f'<rect x="24" y="24" width="16" height="16" rx="3" fill="{_BRASS}"/>'
        f'<circle cx="32" cy="32" r="3" fill="{_CARD}"/>'
        f'<line x1="32" y1="26" x2="32" y2="24" stroke-width="1"/>'),
    "quadcopter_side": f"""
      <line x1="4" y1="20" x2="22" y2="20"/><line x1="42" y1="20" x2="60" y2="20"/>
      <line x1="13" y1="20" x2="13" y2="28"/><line x1="51" y1="20" x2="51" y2="28"/>
      <line x1="8" y1="28" x2="56" y2="28"/>
      <rect x="21" y="26" width="22" height="12" rx="2.5" fill="{_BRASS}"/>
      <circle cx="32" cy="42" r="3.5" fill="{_VELLUM}"/>
      <path d="M25 38 L21 48 M39 38 L43 48 M16 48 H26 M38 48 H48"/>
      <path d="M8 56 H56" stroke-dasharray="2 3" stroke-width="1"/>""",
    "satellite": f"""
      <g transform="rotate(-30 32 32)">
        <rect x="2" y="25" width="20" height="14" fill="{_SAGE}"/>
        <path d="M7 25 V39 M12 25 V39 M17 25 V39" stroke-width="1"/>
        <rect x="42" y="25" width="20" height="14" fill="{_SAGE}"/>
        <path d="M47 25 V39 M52 25 V39 M57 25 V39" stroke-width="1"/>
        <line x1="22" y1="32" x2="26" y2="32"/><line x1="38" y1="32" x2="42" y2="32"/>
        <rect x="26" y="22" width="12" height="20" rx="2" fill="{_CARD}"/>
        <line x1="32" y1="42" x2="32" y2="46"/>
        <path d="M26 47 Q32 53 38 47 Z" fill="{_BRASS}"/>
      </g>""",
    "radar_dish": f"""
      <path d="M10 30 A22 22 0 0 0 40 50 Z" fill="{_VELLUM}"/>
      <line x1="25" y1="40" x2="40" y2="22"/>
      <circle cx="41" cy="21" r="2.6" fill="{_BRASS}"/>
      <path d="M46 14 A9 9 0 0 1 49 25" stroke-width="1.2"/>
      <path d="M50 9 A15 15 0 0 1 55 28" stroke-width="1.2"/>
      <path d="M28 45 L24 58 H40 L36 47"/>
      <line x1="18" y1="58" x2="46" y2="58"/>""",
    "constellation": f"""
      <circle cx="32" cy="32" r="9" fill="{_SAGE}"/>
      <path d="M26 29 Q32 26 38 29 M24 34 Q32 38 40 34" stroke-width="1"/>
      <ellipse cx="32" cy="32" rx="27" ry="11" transform="rotate(-20 32 32)"/>
      <ellipse cx="32" cy="32" rx="11" ry="27" transform="rotate(-20 32 32)" stroke-dasharray="2.5 3.5"/>
      <rect x="6" y="36" width="6" height="6" fill="{_BRASS}" transform="rotate(45 9 39)"/>
      <rect x="52" y="22" width="6" height="6" fill="{_BRASS}" transform="rotate(45 55 25)"/>
      <rect x="21" y="4" width="6" height="6" fill="{_BRASS}" transform="rotate(45 24 7)"/>
      <rect x="37" y="54" width="6" height="6" fill="{_BRASS}" transform="rotate(45 40 57)"/>""",
    "shield": f"""
      <path d="M32 5 L54 13 V30 C54 44 44 53 32 59 C20 53 10 44 10 30 V13 Z" fill="{_SAGE}"/>
      <path d="M32 11 L48 17 V30 C48 40 41 47 32 52 C23 47 16 40 16 30 V17 Z" stroke-width="1"/>
      <path d="M23 31 L29 37 L42 24" stroke-width="2.5"/>""",
    "padlock": f"""
      <path d="M21 28 V20 A11 11 0 0 1 43 20 V28"/>
      <path d="M26 28 V20 A6 6 0 0 1 38 20 V28" stroke-width="1"/>
      <rect x="13" y="28" width="38" height="28" rx="4" fill="{_BRASS}"/>
      <circle cx="32" cy="40" r="4" fill="{_CARD}"/>
      <line x1="32" y1="44" x2="32" y2="50" stroke-width="2.5"/>
      <path d="M17 52 H22 M42 52 H47" stroke-width="1"/>""",
    "fingerprint": f"""
      <rect x="6" y="6" width="52" height="52" rx="4" fill="{_VELLUM}" stroke-width="1"/>
      <path d="M15 37 C14 21 22 12 32 12 C42 12 50 21 49 37"/>
      <path d="M20 46 C16 35 19 20 32 19 C45 20 48 35 44 46"/>
      <path d="M25 51 C21 39 23 26 32 25 C41 26 43 39 39 51"/>
      <path d="M30 54 C27 43 28 31 32 31 C36 31 37 43 34 54"/>
      <path d="M6 18 H10 M54 46 H58 M18 6 V10 M46 54 V58" stroke-width="1"/>""",
    "sealed_document": f"""
      <path d="M13 5 H40 L51 16 V59 H13 Z" fill="{_CARD}"/>
      <path d="M40 5 V16 H51"/>
      <path d="M19 22 H40 M19 28 H45 M19 34 H33" stroke-width="1.2"/>
      <path d="M36 53 L34 62 L41 59 L48 62 L46 53" fill="{_BRASS}"/>
      <circle cx="41" cy="46" r="8.5" fill="{_BRASS}"/>
      <circle cx="41" cy="46" r="5.5" stroke-width="1"/>
      <path d="M38.5 46 L40.5 48 L44 44" stroke-width="1.5"/>""",
    "crosshair": f"""
      <circle cx="32" cy="32" r="22" fill="{_CARD}"/>
      <circle cx="32" cy="32" r="12" stroke-dasharray="2 3" stroke-width="1"/>
      <circle cx="32" cy="32" r="4" fill="{_BRASS}"/>
      <path d="M32 3 V20 M32 44 V61 M3 32 H20 M44 32 H61"/>
      <path d="M8 8 H14 M8 8 V14 M56 8 H50 M56 8 V14 M8 56 H14 M8 56 V50 M56 56 H50 M56 56 V50"
            stroke-width="1"/>""",
    "survey_tripod": f"""
      <rect x="20" y="8" width="24" height="12" rx="2" fill="{_BRASS}"/>
      <circle cx="44" cy="14" r="5" fill="{_CARD}"/>
      <circle cx="44" cy="14" r="2" fill="{_INK}" stroke="none"/>
      <rect x="27" y="20" width="10" height="5" fill="{_VELLUM}"/>
      <path d="M32 25 L14 58 M32 25 L50 58 M32 25 V58"/>
      <path d="M22 44 H42" stroke-width="1"/>
      <path d="M10 58 H54" stroke-dasharray="2 3" stroke-width="1"/>""",
    "radar_scope": f"""
      <circle cx="32" cy="32" r="28" fill="{_CARD}"/>
      <circle cx="32" cy="32" r="19"/>
      <circle cx="32" cy="32" r="10"/>
      <line x1="32" y1="32" x2="52" y2="12" stroke-width="2.5"/>
      <circle cx="32" cy="32" r="2.5" fill="{_INK}" stroke="none"/>""",
}

LABELS: dict[str, str] = {
    "quadcopter_top": "Quadcopter, top view",
    "quadcopter_side": "Quadcopter, side view",
    "satellite": "Satellite",
    "radar_dish": "Radar dish",
    "constellation": "GNSS satellite constellation",
    "shield": "Shield with check mark",
    "padlock": "Padlock",
    "fingerprint": "Fingerprint",
    "sealed_document": "Sealed document",
    "crosshair": "Crosshair",
    "survey_tripod": "Survey instrument on a tripod",
    "radar_scope": "Radar scope",
}

# Older names used by the views -> the redrawn subject.
ALIASES: dict[str, str] = {
    "drone_top": "quadcopter_top",
    "drone_side": "quadcopter_side",
    "radar": "radar_dish",
    "signal": "survey_tripod",
    "document": "sealed_document",
}

STICKER_NAMES: tuple[str, ...] = tuple(_SHAPES)


def resolve(name: str) -> str:
    """Canonical illustration name for `name` (an alias or a real name)."""
    real = ALIASES.get(name, name)
    if real not in _SHAPES:
        raise ValueError(f"unknown sticker {name!r}; choose from {list(STICKER_NAMES) + list(ALIASES)}")
    return real


def sticker(name: str, size: int = 72, rotate: float = 0.0, label: str | None = None) -> str:
    """Return one illustration as an inline <svg> string.

    name    one of STICKER_NAMES or ALIASES
    size    width and height in CSS pixels (> 0)
    rotate  kept for older callers; must be 0 (drawings sit square)
    label   accessible name (defaults to a description of the subject)
    """
    real = resolve(name)
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if abs(rotate) > MAX_TILT_DEG:
        raise ValueError(f"rotate must be 0 in the Ground Truth design (no tilt), got {rotate}")
    title = html.escape(label if label is not None else LABELS[real])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="sg-sticker sg-sticker--{real}" '
        f'width="{size}" height="{size}" viewBox="0 0 64 64" role="img" aria-label="{title}" '
        f'fill="none" stroke="{_INK}" stroke-width="{STROKE_WIDTH}" stroke-linecap="round" '
        f'stroke-linejoin="round"><title>{title}</title>{_SHAPES[real]}</svg>'
    )


def radar_mark(size: int = 22) -> str:
    """The small radar-scope mark used in the SWARMGUARD wordmark (navbar)."""
    return sticker("radar_scope", size=size, label="SwarmGuard")


def drone_mark(size: int = 22) -> str:
    """Kept for older callers: the wordmark's mark is now the radar scope."""
    return radar_mark(size)


def svg_img(svg: str, size: int, alt: str = "", css_class: str = "sg-icon") -> str:
    """Wrap an SVG string as an <img> with a base64 data URI.

    WHY: Streamlit's st.html sanitises markup and removes inline <svg>, but
    keeps <img> tags with data: URIs. The picture is identical; only the
    page can no longer restyle its strokes (colours are set inside the SVG).
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    alt_attr = html.escape(alt, quote=True)
    hidden = "" if alt else ' aria-hidden="true"'
    return (f'<img class="{html.escape(css_class, quote=True)}" src="data:image/svg+xml;base64,{data}" '
            f'width="{size}" height="{size}" alt="{alt_attr}"{hidden}>')


def sticker_img(name: str, size: int = 72, rotate: float = 0.0, label: str | None = None) -> str:
    """sticker() as an <img> that survives st.html (see svg_img)."""
    svg = sticker(name, size=size, rotate=rotate, label=label)
    alt = label if label is not None else LABELS[resolve(name)]
    return svg_img(svg, size, alt, "sg-sticker")


def radar_mark_img(size: int = 22) -> str:
    """radar_mark() as an <img> (the wordmark's small radar scope)."""
    return svg_img(radar_mark(size), size, "", "sg-sticker sg-wordmark__mark")


def drone_mark_img(size: int = 22) -> str:
    """Kept for older callers: same as radar_mark_img."""
    return radar_mark_img(size)
