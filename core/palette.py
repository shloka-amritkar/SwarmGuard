"""The SwarmGuard colour palette: the ONE place where colours are defined.

Every part of the project that draws something reads its colours from here:
    reports/pdf_report.py   PDF headings, rules, table lines, figure colours
    ui/theme.py (Phase 5)   the Streamlit app's CSS and Plotly templates

WHY one shared module: the app's look (neo-brutalist "poster" style, see
CLAUDE.md and docs/phases/phase-5.md) may still change. If each file kept
its own hex strings, a palette change would need edits in many places and
the PDF would silently drift away from the app. With one module, changing a
value here changes it everywhere.

WHY it lives in core/ and not ui/: the PDF code runs from the command line
and must not import the Streamlit-facing ui package. core/ has no UI
dependencies, so anything may import it.

Rule from CLAUDE.md: status is always shown as colour PLUS a text label,
never colour alone, so colour-blind readers lose no information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Palette:
    """Named colours as "#RRGGBB" hex strings.

    frozen=True: nobody can change a colour at run time by accident
    (PALETTE.red = ... raises an error).
    """

    # Surfaces
    cream: str = "#F4F1EA"        # page background in the app
    white: str = "#FFFFFF"        # cards, and the PDF paper
    ink: str = "#111111"          # outlines, text, hard shadows
    # Accent
    red: str = "#C8221B"          # hero bands, headings, the primary action
    red_hover: str = "#9E1A14"    # darker red for hover / emphasis
    blush: str = "#F4B6B2"        # soft fills, table header rows
    muted: str = "#8A8577"        # secondary text, NO_EVIDENCE / INSUFFICIENT
    # Status colours (always paired with a text label)
    status_normal: str = "#2F6B3A"
    status_suspicious: str = "#C98A0B"
    status_alarm: str = "#C8221B"
    status_no_evidence: str = "#8A8577"

    def status_colour(self, status: str) -> str:
        """Colour for a status or verdict string, e.g. "alarm" -> red.

        Accepts DroneStatus values (normal, suspicious, alarm, no_evidence)
        and fused Verdict values (compromised, insufficient_evidence).
        Unknown strings get the muted colour rather than an error, because a
        wrong colour is harmless next to its text label.
        """
        table = {
            "normal": self.status_normal,
            "suspicious": self.status_suspicious,
            "alarm": self.status_alarm,
            "compromised": self.status_alarm,
            "no_evidence": self.status_no_evidence,
            "insufficient_evidence": self.status_no_evidence,
        }
        return table.get(str(status).lower(), self.muted)


def hex_to_rgb01(hex_colour: str) -> tuple[float, float, float]:
    """"#C8221B" -> (0.784, 0.133, 0.106): each channel scaled to [0, 1].

    WHY: ReportLab and matplotlib both accept 0-1 floats; converting in one
    helper keeps the hex string in Palette the single source of truth.
    """
    text = hex_colour.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a colour like '#RRGGBB', got {hex_colour!r}")
    try:
        r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"expected a colour like '#RRGGBB', got {hex_colour!r}") from exc
    return r / 255.0, g / 255.0, b / 255.0


# The palette everybody uses. Import this, never re-type the hex values.
PALETTE = Palette()
