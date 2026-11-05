"""PDF incident report (ReportLab platypus, A4) and its timeline figure.

LAYOUT (top to bottom)
    cover              SWARMGUARD, incident id, time, "SIMULATION REPORT"
    1 executive summary   the explainer's sentences
    2 configuration       key swarm, attack and threshold settings
    3 timeline            per-detector alarm bars per alarmed drone + onset line
    4 detector results    state, first alarm, drones, peak vs threshold
    5 fusion verdicts     per drone: verdict, belief, plausibility
    6 recovery            method, status, estimated attack vector
    7 evidence integrity  file hashes, key fingerprint, previous manifest hash
    8 limitations and disclaimer
Every page has a header and footer with the incident id and page number.

STYLE: white paper (printable), black outlines, deep-red headings, blush
table headers. All colours come from core/palette.py (PALETTE), the same
module the Streamlit app uses, so the report always matches the app.
Fonts: ReportLab's built-in Helvetica and Courier (always available, no
font files to ship; the spec allows Helvetica as the fallback).

WHY platypus ("flowables"): we list paragraphs, tables and images and
ReportLab breaks them across pages itself; we never compute coordinates by
hand except for the header/footer.

THE REPORT'S OWN HASH: the report lists the hashes of the other evidence
files, but cannot contain its own hash (adding the hash would change the
file, and therefore the hash). Its hash is in manifest.json instead, which
is signed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # WHY: render to files without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape  # noqa: E402

from core.palette import PALETTE  # noqa: E402
from core.utils import drone_label, format_drones  # noqa: E402
from core.schema import SimResult  # noqa: E402
from detection.base import DetectionRun  # noqa: E402

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
OUTLINE_W = 1.2          # points; the PDF version of the app's thick outline
DETECTOR_COLOURS = {"A": PALETTE.ink, "B": PALETTE.red, "C": PALETTE.muted, "F": PALETTE.red_hover}
TIMELINE_CAPTION = (
    "Each row is one detector on one drone; bars show when it was in ALARM. A = relative geometry, "
    "B = motion consistency, C = absolute anchors, F = fused alarm. The dashed line is the estimated onset."
)
KEY_LIMITATION = (
    "The signing key is stored unencrypted on disk; production use would require a protected key "
    "(passphrase or hardware module)."
)
DISCLAIMER = (
    "This report describes a SOFTWARE SIMULATION. All numbers are simulation results, not "
    "measurements of real drones. GPS spoofing was simulated by adding offsets to reported "
    "coordinates in software; no radio signals were generated. A valid signature proves the "
    "evidence is unchanged since signing; it does not prove the telemetry was genuine. "
    "No anomaly detected does not prove GPS was genuine. The system does not identify who "
    "caused an anomaly."
)


def _c(hex_colour: str) -> colors.Color:
    """Palette hex -> ReportLab colour."""
    return colors.HexColor(hex_colour)


# --------------------------------------------------------------------------
# Timeline figure (matplotlib)
# --------------------------------------------------------------------------
def _segments(mask: np.ndarray, times: np.ndarray, dt: float) -> list[tuple[float, float]]:
    """Contiguous True stretches of mask as (start, width) pairs for broken_barh."""
    out, start = [], None
    for k, on in enumerate(mask):
        if on and start is None:
            start = times[k]
        if not on and start is not None:
            out.append((start, times[k] - start))
            start = None
    if start is not None:
        out.append((start, times[-1] + dt - start))
    return out


def timeline_figure(runs: Sequence[DetectionRun], onset: float | None) -> Figure:
    """One row per (detector, alarmed drone): bars where that detector is in ALARM.

    A dashed vertical line marks the estimated onset. Colours from PALETTE.
    """
    rows = [(r, i) for r in runs for i in sorted(r.alarm_drones())]
    height = 1.2 + 0.28 * max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(8.0, min(height, 9.0)), dpi=150)
    if not rows:
        ax.text(0.5, 0.5, "No detector alarm in this run", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE.ink)
    for y, (run, i) in enumerate(rows):
        times = run.times()
        dt = float(times[1] - times[0]) if len(times) > 1 else 0.1
        ax.broken_barh(_segments(run.flags()[:, i], times, dt), (y - 0.35, 0.7),
                       facecolors=DETECTOR_COLOURS.get(run.short, PALETTE.ink))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r.short} | {drone_label(i)}" for r, i in rows], fontsize=7)
    ax.invert_yaxis()
    if runs:
        ax.set_xlim(0, float(runs[0].times()[-1]))
    if onset is not None:
        ax.axvline(onset, color=PALETTE.ink, ls="--", lw=1.2, label=f"estimated onset {onset:.1f} s")
        ax.legend(loc="lower right", fontsize=7, frameon=True, edgecolor=PALETTE.ink)
    ax.set_xlabel("time (s)")
    ax.set_title("Detector alarms per drone (simulation result)", fontsize=10, color=PALETTE.ink)
    for spine in ax.spines.values():
        spine.set_color(PALETTE.ink)
        spine.set_linewidth(1.5)
    ax.grid(axis="x", color=PALETTE.blush, lw=0.6)
    fig.tight_layout()
    return fig


def trajectory_figure(result: SimResult) -> Figure:
    """Top-down paths: reported GPS (dotted) and SIMULATION GROUND TRUTH (solid).

    Drones the simulation attacked are drawn in red. WHY show truth here:
    this figure goes into the evidence package as an evaluation aid and is
    labelled as ground truth, which no detector ever saw (rule R1).
    """
    true_pos, gps = result.stack("true_pos"), result.stack("gps_pos")
    attacked = result.attacked_drones()
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=150)
    for i in range(result.n_drones):
        colour = PALETTE.red if i in attacked else PALETTE.ink
        ax.plot(true_pos[:, i, 0], true_pos[:, i, 1], "-", color=colour, lw=1.0)
        ax.plot(gps[:, i, 0], gps[:, i, 1], ":", color=colour, lw=0.7)
    ax.plot([], [], "-", color=PALETTE.ink, label="true path (SIMULATION GROUND TRUTH)")
    ax.plot([], [], ":", color=PALETTE.ink, label="reported GPS")
    ax.plot([], [], "-", color=PALETTE.red, label="drone attacked in the simulation")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=PALETTE.blush, lw=0.6)
    ax.legend(loc="best", fontsize=7, edgecolor=PALETTE.ink)
    ax.set_title("Flight paths (simulation result)", fontsize=10, color=PALETTE.ink)
    for spine in ax.spines.values():
        spine.set_color(PALETTE.ink)
        spine.set_linewidth(1.5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Styles and small builders
# --------------------------------------------------------------------------
def _styles() -> dict[str, ParagraphStyle]:
    """Paragraph styles in the app palette (Helvetica family)."""
    base = getSampleStyleSheet()
    ink, red = _c(PALETTE.ink), _c(PALETTE.red)
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=40, leading=44, textColor=red, alignment=TA_CENTER),
        "cover": ParagraphStyle("cover", parent=base["Normal"], fontName="Helvetica-Bold",
                                fontSize=14, leading=20, textColor=ink, alignment=TA_CENTER),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=15, leading=19, textColor=red, spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["Normal"], fontName="Helvetica",
                               fontSize=9.5, leading=13, textColor=ink, spaceAfter=4),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontName="Helvetica",
                               fontSize=8, leading=10, textColor=ink),
        "mono": ParagraphStyle("mono", parent=base["Normal"], fontName="Courier",
                               fontSize=7, leading=9, textColor=ink),
    }


def _table(data: list[list[Any]], widths: list[float], st: dict[str, ParagraphStyle],
           mono_cols: Sequence[int] = ()) -> Table:
    """A table with a blush header row and black grid lines (cells wrap)."""
    def cell(value: Any, col: int, header: bool) -> Paragraph:
        text = escape("-" if value is None else str(value))
        style = st["mono"] if (col in mono_cols and not header) else st["cell"]
        return Paragraph(f"<b>{text}</b>" if header else text, style)

    body = [[cell(v, c, r == 0) for c, v in enumerate(row)] for r, row in enumerate(data)]
    table = Table(body, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _c(PALETTE.blush)),
        ("GRID", (0, 0), (-1, -1), 0.6, _c(PALETTE.ink)),
        ("BOX", (0, 0), (-1, -1), OUTLINE_W, _c(PALETTE.ink)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _fmt(value: Any, spec: str = ".2f", unit: str = "") -> str:
    """Number -> text, None -> "-"."""
    return "-" if value is None else f"{value:{spec}}{unit}"


def _header_footer(incident_id: str):  # noqa: ANN202 (ReportLab callback type)
    """Page decoration: red rule + id at the top, page number at the bottom."""
    def draw(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(_c(PALETTE.red))
        canvas.setLineWidth(2)
        canvas.line(MARGIN, PAGE_H - 12 * mm, PAGE_W - MARGIN, PAGE_H - 12 * mm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(_c(PALETTE.ink))
        canvas.drawString(MARGIN, PAGE_H - 10 * mm, f"SWARMGUARD  |  {incident_id}")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 10 * mm, "SIMULATION REPORT")
        canvas.setFont("Helvetica", 8)
        canvas.drawString(MARGIN, 10 * mm, f"{incident_id} - simulation result")
        canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def _cover(inc: dict[str, Any], st: dict[str, ParagraphStyle]) -> list[Any]:
    """Cover page: title, incident id, time, scenario, disclaimer."""
    return [Spacer(1, 60 * mm), Paragraph("SWARMGUARD", st["title"]), Spacer(1, 8 * mm),
            Paragraph("SIMULATION REPORT", st["cover"]), Spacer(1, 4 * mm),
            Paragraph(f"Incident {escape(inc['incident_id'])}", st["cover"]),
            Paragraph(f"Created {escape(inc['created_utc'])} (UTC)", st["cover"]),
            Paragraph(f"Scenario {escape(inc['scenario_key'])}, seed {inc['seed']}", st["cover"]),
            Spacer(1, 20 * mm), Paragraph(escape(DISCLAIMER), st["body"]), PageBreak()]


def _config_section(inc: dict[str, Any], st: dict[str, ParagraphStyle]) -> list[Any]:
    """Section 2: key swarm, attack and threshold settings."""
    sim, det, atk = inc["config"]["sim"], inc["config"]["detector"], inc["attack"]
    rows = [["Setting", "Value"],
            ["Drones / formation", f"{sim['n_drones']} / {sim['formation']}"],
            ["Duration / time step", f"{sim['duration']} s / {sim['dt']} s"],
            ["Control loop / centre_gain", f"{sim['loop_mode']} / {sim['centre_gain']}"],
            ["GPS / range / velocity noise", f"{sim['gps_sigma']} m / {sim['range_sigma']} m / {sim['vel_sigma']} m/s"],
            ["Anchor ratio / anchor noise", f"{sim['anchor_ratio']} / {sim['anchor_sigma']} m"],
            ["Simulated attack type", atk.get("type")],
            ["Thresholds A", f"tau_a {det['tau_a']} m, vote {det['vote_fraction']}"],
            ["Thresholds B", f"NIS {det['nis_threshold']}, CUSUM h {det['cusum_h']}, k {det['cusum_k']}"],
            ["Thresholds C", f"tau_c {det['tau_c']} m ({det['c_aggregate']})"],
            ["Persistence", f"{det['persist_m']} of {det['persist_n']} steps"],
            ["Calibration hash", inc["calibration_hash"]],
            ["Software version / git commit", f"{inc['software_version']} / {inc['git_commit']}"]]
    return [Paragraph("2. Configuration", st["h1"]),
            _table(rows, [0.32 * CONTENT_W, 0.68 * CONTENT_W], st)]


def _detector_section(inc: dict[str, Any], st: dict[str, ParagraphStyle]) -> list[Any]:
    """Section 4: one row per detector, plus the agreement pattern."""
    rows = [["Detector", "State", "First alarm", "Drones in alarm", "Peak score", "Threshold"]]
    for short, d in sorted(inc["detected"].items()):
        rows.append([f"{short} ({d['name']})", d["state"].upper().replace("_", " "),
                     _fmt(d["first_alarm_t"], ".1f", " s"), format_drones(d["drones"]),
                     f"{_fmt(d['peak_score'])} ({d['score_name']})", _fmt(d["threshold"])])
    w = CONTENT_W
    return [Paragraph("4. Detector results", st["h1"]),
            _table(rows, [0.2 * w, 0.12 * w, 0.12 * w, 0.24 * w, 0.2 * w, 0.12 * w], st),
            Spacer(1, 3 * mm),
            Paragraph(escape(f"Agreement pattern: {inc['pattern']} - {inc['pattern_description']}"),
                      st["body"])]


def _fusion_section(inc: dict[str, Any], st: dict[str, ParagraphStyle]) -> list[Any]:
    """Section 5: fused verdict per drone at its peak-belief step."""
    rows = [["Drone", "Verdict (at peak belief)", "Belief", "Plausibility", "At", "Fused alarm"]]
    for d in inc["fusion"]["drones"]:
        rows.append([drone_label(d["drone"]), d["verdict"].upper().replace("_", " "),
                     _fmt(d["belief"]), _fmt(d["plausibility"]), _fmt(d["t"], ".1f", " s"),
                     "YES" if d["fused_alarm"] else "no"])
    w = CONTENT_W
    return [Paragraph("5. Fusion verdicts", st["h1"]),
            Paragraph("Dempster-Shafer fusion of A, B and C. Belief = evidence for attack; "
                      "plausibility = belief + uncertainty.", st["body"]),
            _table(rows, [0.14 * w, 0.28 * w, 0.13 * w, 0.15 * w, 0.13 * w, 0.17 * w], st)]


def _recovery_section(inc: dict[str, Any], st: dict[str, ParagraphStyle]) -> list[Any]:
    """Section 6: recovery method, status and estimated attack vector."""
    rec = inc["recovery"]
    vec = rec.get("attack_vector")
    rows = [["Item", "Value"], ["Method", rec.get("method")], ["Status", rec.get("status")],
            ["Estimated attack vector (east, north)",
             f"({vec[0]:.2f}, {vec[1]:.2f}) m" if vec else "not available"],
            ["Magnitude", _fmt(rec.get("attack_magnitude"), ".2f", " m")],
            ["Outlier anchors", format_drones(rec.get("outlier_anchors", []))]]
    return [Paragraph("6. Recovery", st["h1"]), _table(rows, [0.4 * CONTENT_W, 0.6 * CONTENT_W], st)]


def _integrity_section(file_hashes: Sequence[dict[str, Any]], fingerprint: str, previous: str,
                       st: dict[str, ParagraphStyle]) -> list[Any]:
    """Section 7: file hashes, key fingerprint, previous manifest hash."""
    rows = [["File", "SHA-256", "Bytes"]]
    rows += [[f["path"], f["sha256"], f["size_bytes"]] for f in file_hashes]
    w = CONTENT_W
    return [Paragraph("7. Evidence integrity", st["h1"]),
            _table(rows, [0.3 * w, 0.58 * w, 0.12 * w], st, mono_cols=(1,)),
            Spacer(1, 3 * mm),
            Paragraph(f"Public key fingerprint: <font face='Courier'>{escape(fingerprint)}</font>", st["body"]),
            Paragraph(f"Previous manifest hash: <font face='Courier'>{escape(previous)}</font>", st["body"]),
            Paragraph("This report's own SHA-256 is listed in manifest.json, which is signed with "
                      "Ed25519 (manifest.sig). A report cannot contain its own hash.", st["body"])]


def build_pdf_report(
    path: str | Path,
    incident: dict[str, Any],
    file_hashes: Sequence[dict[str, Any]],
    public_key_fingerprint: str,
    previous_manifest_hash: str,
    timeline_png: str | Path | None = None,
) -> Path:
    """Write the incident report PDF to path and return it.

    incident     Incident.to_dict()
    file_hashes  [{path, sha256, size_bytes}] for the other evidence files
    """
    st = _styles()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    story: list[Any] = _cover(incident, st)
    story.append(Paragraph("1. Executive summary", st["h1"]))
    story += [Paragraph(escape(s), st["body"]) for s in incident.get("explanation", [])]
    story += _config_section(incident, st)
    timeline: list[Any] = [Paragraph("3. Timeline", st["h1"]), Paragraph(TIMELINE_CAPTION, st["body"])]
    if timeline_png is not None and Path(timeline_png).exists():
        img = Image(str(timeline_png))
        scale = CONTENT_W / img.imageWidth
        img.drawWidth, img.drawHeight = CONTENT_W, min(img.imageHeight * scale, 150 * mm)
        timeline.append(img)
    else:
        timeline.append(Paragraph("Timeline figure not available.", st["body"]))
    # WHY KeepTogether: a heading alone at the bottom of a page, with its
    # figure on the next page, is hard to read.
    story.append(KeepTogether(timeline))
    story += _detector_section(incident, st) + _fusion_section(incident, st)
    story += _recovery_section(incident, st)
    story += _integrity_section(file_hashes, public_key_fingerprint, previous_manifest_hash, st)
    story += [Paragraph("8. Limitations and disclaimer", st["h1"]), Paragraph(escape(DISCLAIMER), st["body"]),
              Paragraph("Dempster's rule assumes independent sources, but A and B read the same GPS; "
                        "reliability discounting limits, but does not remove, this double counting.",
                        st["body"]),
              Paragraph(escape(KEY_LIMITATION), st["body"])]
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=f"SwarmGuard {incident['incident_id']}", author="SwarmGuard",
                            subject="Simulation report")
    decorate = _header_footer(incident["incident_id"])
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    logger.info("wrote PDF report %s", out)
    return out
