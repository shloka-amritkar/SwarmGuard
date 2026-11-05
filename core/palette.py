"""The SwarmGuard colour palette: the ONE place where colours are defined.

Every part of the project that draws something reads its colours from here:
    reports/pdf_report.py   PDF headings, rules, table lines, figure colours
    ui/theme.py             re-exports PALETTE for the Streamlit app's CSS and
                            Plotly templates (it adds no colours of its own)
    assets/stickers.py      line-art illustrations

WHY one shared module: if each file kept its own hex strings, a palette
change would need edits in many places and the PDF would silently drift
away from the app. With one module, changing a value here changes it
everywhere. (This paid off twice: the design moved from a red "poster"
style to "Ivory + Console" and then to "Ground Truth" by editing this file
and renaming its fields.)

WHY it lives in core/ and not ui/: the PDF code runs from the command line
and must not import the Streamlit-facing ui package. core/ has no UI
dependencies, so anything may import it.

COLOUR ROLES ("Ground Truth", docs/phases/phase-7.md section B1)
    sand / card / vellum   survey paper: page, cards, secondary surfaces
    espresso               ink: text, outlines and hard shadows
    olive                  the ONE brand colour (active nav, links, headings)
    brass                  the ONE action colour (primary buttons)
    charcoal / char_*      the only dark surfaces: instrument panels
    amber                  radar sweep, flight trails on panels
    brick (status_alarm)   the ONLY red, used for alarms and the tamper test

CONTRAST RULE: status colours are not text colours (SUSPICIOUS amber on
card is about 2.6:1, WCAG AA needs 4.5:1). Status is shown as espresso text
on a tinted fill with the status colour on the border, dot or bar.
The spec's "stone" (#8C8579) is only 3.2:1 on sand, so it is used for
non-text marks (rulers, disabled, NO_EVIDENCE dots) and the slightly darker
stone_text (#6B6459, 5.0:1) is used for caption TEXT.
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
    (PALETTE.olive = ... raises an error).
    """

    # Paper and ink
    sand: str = "#F3EEE3"         # page background (survey paper)
    card: str = "#FBF8F2"         # cards, inputs (warm white, never pure white)
    vellum: str = "#ECE5D6"       # secondary surfaces, table headers, hover rows
    espresso: str = "#1E1A16"     # primary text, outlines, hard shadows
    ink_2: str = "#4A433B"        # secondary text
    stone: str = "#8C8579"        # ruler marks, disabled, NO_EVIDENCE (not body text)
    stone_text: str = "#6B6459"   # captions and eyebrows as TEXT (passes AA)
    rule: str = "#DDD5C4"         # hairlines inside cards
    contour: str = "#E4DCCB"      # background contour lines and grids
    # Brand and interaction
    olive: str = "#3F4F24"        # brand: active nav, links, key headings
    olive_2: str = "#566B33"      # hover state of olive elements
    sage: str = "#DCE3CC"         # soft olive tint: selected rows, pills, focus fills
    brass: str = "#D9A441"        # primary buttons
    brass_2: str = "#C68F2E"      # button hover / pressed, ripple
    terracotta: str = "#B5552F"   # secondary accent, used sparingly
    # Instrument panels (the only dark surfaces)
    charcoal: str = "#23201C"     # panel background
    charcoal_2: str = "#2D2924"   # raised rows inside panels
    char_line: str = "#3B362F"    # panel borders, gridlines, range rings
    char_text: str = "#EFE8DA"    # panel text
    char_mute: str = "#A69D8E"    # panel secondary text
    amber: str = "#F2B24C"        # radar sweep, flight trails, neutral traces on panels
    # Status (always paired with a text label)
    status_normal: str = "#5B7F3A"
    status_suspicious: str = "#D28A1E"
    status_alarm: str = "#B3261E"     # brick: the only red in the app
    status_no_evidence: str = "#8C8579"
    # Detectors (fixed everywhere)
    det_a: str = "#4B6E8E"        # A relative geometry, dusty steel
    det_b: str = "#2F7D6D"        # B motion consistency, deep teal
    det_c: str = "#B8862B"        # C absolute anchors, ochre
    det_d: str = "#B5552F"        # D experimental ML, terracotta

    def status_colour(self, status: str) -> str:
        """Colour for a status or verdict string, e.g. "alarm" -> brick red.

        Accepts DroneStatus values (normal, suspicious, alarm, no_evidence)
        and fused Verdict values (compromised, insufficient_evidence).
        Unknown strings get stone rather than an error, because a wrong
        colour is harmless next to its text label.
        """
        table = {
            "normal": self.status_normal,
            "suspicious": self.status_suspicious,
            "alarm": self.status_alarm,
            "compromised": self.status_alarm,
            "no_evidence": self.status_no_evidence,
            "insufficient_evidence": self.status_no_evidence,
        }
        return table.get(str(status).lower(), self.stone)

    def detector_colour(self, short: str) -> str:
        """Colour of detector "A", "B", "C" or "D"; "F" (the fused alarm) is brick red."""
        table = {"A": self.det_a, "B": self.det_b, "C": self.det_c, "D": self.det_d,
                 "F": self.status_alarm}
        if short not in table:
            raise ValueError(f"unknown detector {short!r}; expected one of {sorted(table)}")
        return table[short]


def hex_to_rgb01(hex_colour: str) -> tuple[float, float, float]:
    """"#3F4F24" -> (0.247, 0.310, 0.141): each channel scaled to [0, 1].

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

    WCAG AA requires at least 4.5 for normal text. Example: espresso on sand
    is about 14.9; stone_text (#6B6459) on sand is about 5.0.
    """
    light, dark = sorted((relative_luminance(foreground), relative_luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


# The palette everybody uses. Import this, never re-type the hex values.
PALETTE = Palette()
