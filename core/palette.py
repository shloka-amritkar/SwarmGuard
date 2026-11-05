"""The SwarmGuard colour palette: the ONE place where colours are defined.

Every part of the project that draws something reads its colours from here:
    reports/pdf_report.py   PDF headings, rules, table lines, figure colours
    ui/theme.py             re-exports PALETTE for the Streamlit app's CSS and
                            Plotly templates (it adds no colours of its own)

WHY one shared module: if each file kept its own hex strings, a palette
change would need edits in many places and the PDF would silently drift
away from the app. With one module, changing a value here changes it
everywhere. (This already paid off once: the design moved from a red
"poster" style to "Ivory + Console" by editing only this file's values.)

WHY it lives in core/ and not ui/: the PDF code runs from the command line
and must not import the Streamlit-facing ui package. core/ has no UI
dependencies, so anything may import it.

COLOUR ROLES ("Ivory + Console", docs/phases/phase-5.md section 5.1)
    forest    the ONE brand colour (links, active nav, key headings)
    lavender  the ONE interaction colour (primary buttons, selected states)
    red       ONLY alarms and destructive actions, so alarms cannot be missed
    console*  dark green-black panels for map, feed and verification

CONTRAST RULE: status colours are too light to be readable as text on white
(e.g. SUSPICIOUS #E0A100 is only 2.3:1, WCAG AA needs 4.5:1). So status is
always shown as ink text on a tinted fill, with the status colour on the
border, dot or bar, never as the text colour itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WCAG_AA_TEXT = 4.5


@dataclass(frozen=True)
class Palette:
    """Named colours as "#RRGGBB" hex strings.

    frozen=True: nobody can change a colour at run time by accident
    (PALETTE.forest = ... raises an error).
    """

    # Light page
    ivory: str = "#FFFEF2"        # page background
    paper: str = "#FFFFFF"        # cards, and the PDF paper
    ink: str = "#111111"          # text, outlines, hard shadows
    ink_2: str = "#3D3D3A"        # secondary text
    muted: str = "#77756E"        # captions, axis labels
    line: str = "#E4E1D3"         # hairlines inside cards, gridlines
    forest: str = "#0B4D40"       # brand colour
    forest_2: str = "#0F6B58"     # hover state of forest elements
    lavender: str = "#E6DDFE"     # primary button fill
    lavender_2: str = "#D7C9FD"   # button hover / pressed, selected states
    blush: str = "#F4B6B2"        # rare soft highlight (e.g. heatmap middle)
    # Console panels (dark)
    console: str = "#0E1F1B"      # panel background
    console_2: str = "#132A25"    # raised rows inside panels
    console_ln: str = "#24413A"   # panel borders and gridlines
    console_tx: str = "#E9F1EC"   # panel text
    console_mu: str = "#8FA79E"   # panel secondary text
    mint: str = "#7FD8B8"         # flight paths and positive signals on dark
    # Status (always paired with a text label)
    status_normal: str = "#2F9E6B"
    status_suspicious: str = "#E0A100"
    status_alarm: str = "#C8221B"
    status_no_evidence: str = "#8A8577"
    # Detectors (fixed everywhere)
    det_a: str = "#4C6EF5"        # A relative geometry, blue
    det_b: str = "#0CA5A5"        # B motion consistency, teal
    det_c: str = "#B7791F"        # C absolute anchors, ochre
    det_d: str = "#8C6FD8"        # D experimental ML, violet

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

    def detector_colour(self, short: str) -> str:
        """Colour of detector "A", "B", "C" or "D"; "F" (the fused alarm) is alarm red."""
        table = {"A": self.det_a, "B": self.det_b, "C": self.det_c, "D": self.det_d,
                 "F": self.status_alarm}
        if short not in table:
            raise ValueError(f"unknown detector {short!r}; expected one of {sorted(table)}")
        return table[short]


def hex_to_rgb01(hex_colour: str) -> tuple[float, float, float]:
    """"#0B4D40" -> (0.043, 0.302, 0.251): each channel scaled to [0, 1].

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


def relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance L of a colour (0 = black, 1 = white).

        each channel c in [0, 1] is linearised:
            c_lin = c / 12.92                    if c <= 0.03928
            c_lin = ((c + 0.055) / 1.055) ^ 2.4  otherwise
        L = 0.2126 R_lin + 0.7152 G_lin + 0.0722 B_lin

    WHY the weights: human eyes are most sensitive to green and least to blue.
    """
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                for c in hex_to_rgb01(hex_colour)]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio = (L_lighter + 0.05) / (L_darker + 0.05), from 1 to 21.

    WCAG AA requires at least 4.5 for normal text. Example: ink on ivory is
    about 18.6; muted (#77756E) on ivory is about 4.55, just above the limit.
    """
    light, dark = sorted((relative_luminance(foreground), relative_luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


# The palette everybody uses. Import this, never re-type the hex values.
PALETTE = Palette()
