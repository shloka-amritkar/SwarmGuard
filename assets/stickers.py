"""Inline SVG line-art "stickers" (docs/phases/phase-5.md section 5.5).

Each sticker is drawn on a 64 x 64 grid with 2px ink outlines and flat fills
in paper, lavender or blush (all from core/palette.py). Subjects are
technical only (drones, satellites, radar, shields, padlocks, documents);
no faces, no imagery copied from the design references.

Use them on Home, scenario cards, empty states and section headers; never
inside charts (charts must stay readable first).

    sticker("radar", size=72, rotate=-3) -> "<svg ...>...</svg>"

WHY inline SVG and not image files: an inline SVG scales without blurring,
inherits nothing from the page, needs no file serving, and its colours come
straight from the palette module, so a palette change restyles every sticker.
WHY an optional small tilt (-4 to 4 degrees): it gives a hand-placed feel
without looking messy; larger angles would.
"""

from __future__ import annotations

import html
import logging

from core.palette import PALETTE

logger = logging.getLogger(__name__)

MAX_TILT_DEG = 4.0

_INK = PALETTE.ink
_PAPER = PALETTE.paper
_LAV = PALETTE.lavender
_BLUSH = PALETTE.blush

# Inner SVG markup of each sticker (viewBox 0 0 64 64). The root <svg> sets
# stroke = ink, stroke-width = 2 and fill = none, so shapes only state a fill
# when they have one.
_SHAPES: dict[str, str] = {
    "drone_top": f"""
      <circle cx="16" cy="16" r="9" fill="{_LAV}"/>
      <circle cx="48" cy="16" r="9" fill="{_LAV}"/>
      <circle cx="16" cy="48" r="9" fill="{_LAV}"/>
      <circle cx="48" cy="48" r="9" fill="{_LAV}"/>
      <line x1="16" y1="16" x2="48" y2="48"/>
      <line x1="48" y1="16" x2="16" y2="48"/>
      <rect x="24" y="24" width="16" height="16" rx="4" fill="{_PAPER}"/>
      <circle cx="32" cy="32" r="2.5" fill="{_INK}" stroke="none"/>""",
    "drone_side": f"""
      <line x1="4" y1="22" x2="20" y2="22"/>
      <line x1="44" y1="22" x2="60" y2="22"/>
      <line x1="12" y1="22" x2="12" y2="30"/>
      <line x1="52" y1="22" x2="52" y2="30"/>
      <line x1="8" y1="30" x2="56" y2="30"/>
      <rect x="21" y="28" width="22" height="12" rx="3" fill="{_LAV}"/>
      <path d="M25 40 L21 49 M39 40 L43 49 M17 49 H25 M39 49 H47"/>
      <circle cx="32" cy="34" r="2.2" fill="{_INK}" stroke="none"/>""",
    "satellite": f"""
      <g transform="rotate(-30 32 32)">
        <rect x="3" y="25" width="19" height="14" fill="{_LAV}"/>
        <line x1="12.5" y1="25" x2="12.5" y2="39"/>
        <rect x="42" y="25" width="19" height="14" fill="{_LAV}"/>
        <line x1="51.5" y1="25" x2="51.5" y2="39"/>
        <line x1="22" y1="32" x2="26" y2="32"/>
        <line x1="38" y1="32" x2="42" y2="32"/>
        <rect x="26" y="23" width="12" height="18" rx="2" fill="{_PAPER}"/>
        <line x1="32" y1="41" x2="32" y2="46"/>
        <path d="M26 46 Q32 54 38 46 Z" fill="{_BLUSH}"/>
      </g>""",
    "radar": f"""
      <circle cx="32" cy="32" r="26" fill="{_PAPER}"/>
      <path d="M32 32 L32 6 A26 26 0 0 1 54.5 19 Z" fill="{_BLUSH}"/>
      <circle cx="32" cy="32" r="17"/>
      <circle cx="32" cy="32" r="8"/>
      <line x1="32" y1="32" x2="54.5" y2="19"/>
      <circle cx="43" cy="21" r="2.4" fill="{_INK}" stroke="none"/>
      <circle cx="21" cy="41" r="2.4" fill="{_INK}" stroke="none"/>
      <circle cx="32" cy="32" r="2" fill="{_INK}" stroke="none"/>""",
    "signal": f"""
      <path d="M24 58 L32 34 L40 58"/>
      <line x1="27" y1="49" x2="37" y2="49"/>
      <circle cx="32" cy="30" r="4.5" fill="{_LAV}"/>
      <path d="M22 20 A14 14 0 0 0 22 40"/>
      <path d="M42 20 A14 14 0 0 1 42 40"/>
      <path d="M14 12 A24 24 0 0 0 14 48"/>
      <path d="M50 12 A24 24 0 0 1 50 48"/>""",
    "constellation": f"""
      <ellipse cx="32" cy="32" rx="27" ry="12" transform="rotate(-20 32 32)"/>
      <ellipse cx="32" cy="32" rx="12" ry="27" transform="rotate(-20 32 32)" stroke-dasharray="3 4"/>
      <circle cx="32" cy="32" r="7" fill="{_BLUSH}"/>
      <circle cx="9" cy="39" r="3.5" fill="{_LAV}"/>
      <circle cx="55" cy="25" r="3.5" fill="{_LAV}"/>
      <circle cx="24" cy="7" r="3.5" fill="{_LAV}"/>
      <circle cx="40" cy="57" r="3.5" fill="{_LAV}"/>""",
    "shield": f"""
      <path d="M32 6 L54 14 V30 C54 44 44 53 32 58 C20 53 10 44 10 30 V14 Z" fill="{_LAV}"/>
      <path d="M22 32 L29 39 L43 24" stroke-width="3"/>""",
    "padlock": f"""
      <path d="M22 28 V20 A10 10 0 0 1 42 20 V28"/>
      <rect x="14" y="28" width="36" height="28" rx="5" fill="{_LAV}"/>
      <circle cx="32" cy="40" r="4" fill="{_INK}" stroke="none"/>
      <line x1="32" y1="42" x2="32" y2="49" stroke-width="3"/>""",
    "fingerprint": f"""
      <circle cx="32" cy="32" r="28" fill="{_BLUSH}" stroke="none"/>
      <path d="M14 36 C13 20 22 10 32 10 C42 10 51 20 50 36"/>
      <path d="M20 46 C16 36 18 20 32 18 C46 20 48 36 44 46"/>
      <path d="M25 50 C21 38 23 25 32 24 C41 25 43 38 39 50"/>
      <path d="M30 54 C27 42 28 30 32 30 C36 30 37 42 34 54"/>""",
    "document": f"""
      <path d="M14 6 H40 L50 16 V58 H14 Z" fill="{_PAPER}"/>
      <path d="M40 6 V16 H50"/>
      <line x1="20" y1="22" x2="40" y2="22"/>
      <line x1="20" y1="29" x2="44" y2="29"/>
      <line x1="20" y1="36" x2="32" y2="36"/>
      <path d="M35 52 L33 61 L40 58 L47 61 L45 52" fill="{_BLUSH}"/>
      <circle cx="40" cy="46" r="8" fill="{_BLUSH}"/>
      <path d="M36.5 46 L39 48.5 L43.5 43.5"/>""",
    "crosshair": f"""
      <circle cx="32" cy="32" r="20" fill="{_PAPER}"/>
      <circle cx="32" cy="32" r="8" fill="{_LAV}"/>
      <line x1="32" y1="4" x2="32" y2="18"/>
      <line x1="32" y1="46" x2="32" y2="60"/>
      <line x1="4" y1="32" x2="18" y2="32"/>
      <line x1="46" y1="32" x2="60" y2="32"/>
      <circle cx="32" cy="32" r="1.8" fill="{_INK}" stroke="none"/>""",
}

LABELS: dict[str, str] = {
    "drone_top": "Drone, top view",
    "drone_side": "Drone, side view",
    "satellite": "Satellite",
    "radar": "Radar sweep",
    "signal": "Radio signal waves",
    "constellation": "GNSS satellite constellation",
    "shield": "Shield with check mark",
    "padlock": "Padlock",
    "fingerprint": "Fingerprint",
    "document": "Document with seal",
    "crosshair": "Crosshair",
}

STICKER_NAMES: tuple[str, ...] = tuple(_SHAPES)


def sticker(name: str, size: int = 72, rotate: float = 0.0, label: str | None = None) -> str:
    """Return one sticker as an inline <svg> string.

    name    one of STICKER_NAMES
    size    width and height in CSS pixels (> 0)
    rotate  tilt in degrees, -4 to 4 (the "hand-placed" look)
    label   accessible name (defaults to a description of the subject)
    """
    if name not in _SHAPES:
        raise ValueError(f"unknown sticker {name!r}; choose from {list(STICKER_NAMES)}")
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if abs(rotate) > MAX_TILT_DEG:
        raise ValueError(f"rotate must be between -{MAX_TILT_DEG:g} and {MAX_TILT_DEG:g} degrees, got {rotate}")
    title = html.escape(label if label is not None else LABELS[name])
    tilt = f' style="transform: rotate({rotate:g}deg)"' if rotate else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="sg-sticker sg-sticker--{name}" '
        f'width="{size}" height="{size}" viewBox="0 0 64 64" role="img" aria-label="{title}" '
        f'fill="none" stroke="{_INK}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round"{tilt}><title>{title}</title>{_SHAPES[name]}</svg>'
    )


def drone_mark(size: int = 22) -> str:
    """The small line-art drone used in the SwarmGuard wordmark."""
    return sticker("drone_top", size=size, label="SwarmGuard")
