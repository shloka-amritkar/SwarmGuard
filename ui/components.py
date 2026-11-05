"""Reusable UI components ("Ground Truth", docs/phases/phase-7.md; first built in phase 5).

Every component comes in two parts:
    *_html(...)   a PURE function that returns an HTML string (unit-tested)
    wrapper(...)  a thin function that puts that HTML on the page with st.html,
                  or combines it with real Streamlit widgets

WHY split them: the HTML can be tested without starting Streamlit, and the
wrappers stay tiny. All text is passed through html.escape(), so a drone
name or a file path can never inject markup.

WHY st.html and not st.markdown: markdown treats indentation and blank
lines specially and its own paragraph styles override ours; st.html
inserts markup as written. st.html does remove inline <svg>, so every
graphic (stickers, the drone mark, the verify icons) is embedded as an
<img> with an SVG data URI instead (assets/stickers.py, svg_img).

Widgets cannot live inside a raw HTML <div>, so cards and instrument panels
that contain buttons, sliders or charts use st.container(key="sg-...").
Streamlit adds the class "st-key-<key>" to such a container, and
assets/style.css styles every "st-key-sg-panel*" / "st-key-sg-card*".

GROUND TRUTH MOTIFS built here: crosshair corners (CSS on .sg-corners),
the scrolling hash ribbon (hash_ribbon_html), case-file cards
(case_card_html), rubber stamps (stamp_html), survey-mark dividers
(survey_divider_html) and the custody timeline (custody_timeline_html).

Status is always shown as text (rule from CLAUDE.md), never colour alone.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.stickers import radar_mark_img, sticker_img, svg_img
from analysis.experiments import PipelineResult
from core.utils import LAT0, LON0
from ui.state import (
    QUICK_MODEL_RUNS,
    QUICK_START,
    Event,
    RibbonEntry,
    get_ml_model,
    get_or_run_pipeline,
    hash_ribbon_entries,
    run_scenario,
    train_quick_model,
)
from ui.theme import DETECTOR_NAMES, PALETTE, status_label, status_tone

logger = logging.getLogger(__name__)

_E = html.escape


def _slug(text: str) -> str:
    """"Live map (Pune)" -> "live-map-pune" (for Streamlit container keys)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "panel"


# --------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------
def status_pill_html(status: str, label: str | None = None) -> str:
    """Outlined pill: dot + uppercase TEXT label, tinted by status (never colour alone)."""
    tone = status_tone(status)
    return f'<span class="sg-pill sg-pill--{tone}">{_E(label or status_label(status))}</span>'


def detector_pill_html(short: str, text: str | None = None) -> str:
    """Pill in a detector's fixed colour, e.g. "A · detect"."""
    tone = f"det-{short.lower()}" if short in "ABCD" else "brand"
    return f'<span class="sg-pill sg-pill--{tone}">{_E(text or short)}</span>'


def wordmark_html(size: int = 22) -> str:
    """Radar-scope mark + "SWARMGUARD" in IBM Plex Mono 500, letter-spaced 0.14em (B5)."""
    return (f'<span class="sg-wordmark" aria-label="SwarmGuard">{radar_mark_img(size)}'
            f'<span class="sg-wordmark__text">SWARMGUARD</span></span>')


def sim_label_html(text: str = "Simulation result") -> str:
    """Small mono label that marks numbers as simulation results (rule R4)."""
    return f'<span class="sg-sim-label">{_E(text)}</span>'


def eyebrow_html(text: str) -> str:
    """Small letter-spaced uppercase label above a title."""
    return f'<p class="sg-eyebrow">{_E(text)}</p>'


def page_header_html(title: str, italic: str | None = None, eyebrow: str | None = None,
                     caption: str | None = None) -> str:
    """Serif page title with an optional italic second part, eyebrow and caption."""
    parts = ['<div class="sg-page-head">']
    if eyebrow:
        parts.append(eyebrow_html(eyebrow))
    title_html = _E(title) + (f" <em>{_E(italic)}</em>" if italic else "")
    parts.append(f'<h1 class="sg-page-title">{title_html}</h1>')
    if caption:
        parts.append(f'<p class="sg-page-caption">{_E(caption)}</p>')
    parts.append("</div>")
    return "".join(parts)


def section_html(title: str, caption: str | None = None, eyebrow: str | None = None,
                 anchor: str | None = None) -> str:
    """Section heading (IBM Plex Sans 600) with optional eyebrow, caption and anchor id."""
    anchor_attr = f' id="{_E(anchor)}"' if anchor else ""
    parts = [f'<div class="sg-section"{anchor_attr}>']
    if eyebrow:
        parts.append(eyebrow_html(eyebrow))
    parts.append(f'<h2 class="sg-section__title">{_E(title)}</h2>')
    if caption:
        parts.append(f'<p class="sg-section__caption">{_E(caption)}</p>')
    parts.append("</div>")
    return "".join(parts)


def clock_html(now: datetime | None = None) -> str:
    """A UTC clock "hh:mm:ss" that keeps ticking with CSS only (no JavaScript).

    Three integer CSS properties (--sg-h, --sg-m, --sg-s) are animated with
    steps(): seconds count 0..59 every 60 s, minutes 0..59 every 3600 s,
    hours 0..23 every 86400 s. A NEGATIVE animation-delay starts each
    animation part-way through, at the server time when the page was drawn:
        delay_s = -(s),  delay_m = -(60 m + s),  delay_h = -(3600 h + 60 m + s)
    WHY CSS: st.html removes <script>, and a Streamlit rerun every second
    would make the whole page flicker. A screen-reader copy of the time is
    included as hidden text.
    """
    now = now or datetime.now(timezone.utc)
    h, m, sec = now.hour, now.minute, now.second
    return (f'<span class="sg-clock" aria-hidden="true">'
            f'<span class="sg-clock__h" style="--sg-delay:-{3600 * h + 60 * m + sec}s"></span>:'
            f'<span class="sg-clock__m" style="--sg-delay:-{60 * m + sec}s"></span>:'
            f'<span class="sg-clock__s" style="--sg-delay:-{sec}s"></span></span>'
            f'<span class="sg-sr">{h:02d}:{m:02d}:{sec:02d} UTC</span>')


def navbar_html(active_page: str, status: str | None, status_text: str | None,
                scenario: str | None, now: datetime | None = None) -> str:
    """The full-width top bar (B5) around Streamlit's own navigation links.

    Left: radar mark + SWARMGUARD wordmark. Right: live readout
    "18.5204°N 73.8567°E · UTC hh:mm:ss", the scenario and a status pill.
    The page links themselves are Streamlit's st.navigation(position="top"),
    which style.css centres on top of this bar.
    """
    right = [f'<span class="sg-readout">{LAT0:.4f}°N {LON0:.4f}°E · UTC {clock_html(now)}</span>']
    if scenario:
        right.append(f'<span class="sg-scenario-tag">{_E(scenario)}</span>')
    if status is not None:
        right.append(status_pill_html(status, status_text))
    return (f'<div class="sg-topbar" role="banner"><div class="sg-topbar__left">{wordmark_html()}'
            f'<span class="sg-sr">Current page: {_E(active_page)}</span></div>'
            f'<div class="sg-topbar__right">{"".join(right)}</div></div>')


# --------------------------------------------------------------------------
# KPI cards
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class KPI:
    """One KPI card: label, value (int values count up), optional status and delta text."""

    label: str
    value: int | float | str
    status: str | None = None
    delta: str | None = None


def kpi_card_html(label: str, value: int | float | str, status: str | None = None,
                  delta: str | None = None) -> str:
    """KPI card with IBM Plex Mono numerals; whole numbers count up (CSS @property).

    The visible number is drawn by a CSS counter, so a screen-reader copy is
    included as hidden text.
    """
    tone = status_tone(status) if status else "plain"
    if isinstance(value, int) and not isinstance(value, bool):
        shown = (f'<span class="sg-count" style="--sg-target:{value}" aria-hidden="true"></span>'
                 f'<span class="sg-sr">{value}</span>')
    else:
        shown = _E(str(value))
    delta_html = f'<p class="sg-kpi__delta">{_E(delta)}</p>' if delta else ""
    return (f'<div class="sg-kpi sg-kpi--{tone}"><p class="sg-kpi__label">{_E(label)}</p>'
            f'<p class="sg-kpi__value">{shown}</p>{delta_html}<span class="sg-kpi__bar"></span></div>')


def kpi_row_html(items: Sequence[KPI]) -> str:
    """A responsive grid of KPI cards."""
    cards = "".join(kpi_card_html(k.label, k.value, k.status, k.delta) for k in items)
    return f'<div class="sg-kpis">{cards}</div>'


# --------------------------------------------------------------------------
# Feed, ribbon, banner, inspector, verification
# --------------------------------------------------------------------------
def event_feed_html(events: Sequence[Event], limit: int = 12,
                    empty_text: str = "No detection events: no anomaly detected so far.") -> str:
    """Newest-first feed rows: mono timestamp, status dot, sentence, detector."""
    if not events:
        return f'<p class="sg-feed__text">{_E(empty_text)}</p>'
    rows = []
    for e in list(events)[:limit]:
        who = DETECTOR_NAMES.get(e.detector, e.detector.capitalize())
        rows.append(
            f'<li class="sg-feed__row"><span class="sg-feed__time">t = {e.t:.1f} s</span>'
            f'<span class="sg-feed__dot sg-feed__dot--{status_tone(e.status)}"></span>'
            f'<div><p class="sg-feed__text">{_E(e.text)}</p>'
            f'<p class="sg-feed__meta">{_E(who)} · {_E(status_label(e.status))}</p></div></li>'
        )
    return f'<ul class="sg-feed">{"".join(rows)}</ul>'


def ribbon_html(events: Sequence[Event], seconds: float = 60.0, limit: int = 10) -> str:
    """Espresso band of scrolling events. Content is doubled so the loop is seamless."""
    items = [f'<span class="sg-ribbon__item"><b>t = {e.t:.1f} s</b> · {_E(e.text)}</span>'
             for e in list(events)[:limit]]
    if not items:
        items = ['<span class="sg-ribbon__item">No detection events · no anomaly detected</span>']
    sep = '<span class="sg-ribbon__sep">···</span>'
    track = sep.join(items)
    return (f'<div class="sg-ribbon" aria-label="Recent detection events">'
            f'<div class="sg-ribbon__track" style="--sg-ribbon-duration:{seconds:g}s">'
            f'{track}{sep}{track}{sep}</div></div>')


def pattern_banner_html(title: str, body: str, tone: str = "alarm") -> str:
    """Outlined banner with a coloured left bar (red only for real alarms)."""
    extra = "" if tone == "alarm" else " sg-banner--info"
    label = status_pill_html("alarm", "Alarm") if tone == "alarm" else status_pill_html("none", "Note")
    return (f'<div class="sg-banner{extra}">{label}<div><p class="sg-banner__title">{_E(title)}</p>'
            f'<p class="sg-banner__body">{_E(body)}</p></div></div>')


def _fmt(value: float | None, digits: int = 2) -> str:
    """Number -> text, None -> "-"."""
    return "-" if value is None else f"{value:.{digits}f}"


def drone_inspector_html(data: dict[str, Any]) -> str:
    """Case-file side card for one drone (data from ui.state.inspector_data).

    A folder tab on the top edge carries "INSPECTOR · <drone>" in mono, like
    the tab of a forensic case file (motif M6).
    """
    rows = []
    for d in data["detectors"]:
        rows.append(f'<tr><th>{_E(d["short"])} · {_E(d["name"])}</th>'
                    f'<td>{status_pill_html(d["status"])}</td><td class="sg-num">{_fmt(d["score"])}</td></tr>')
    rows.append(f'<tr><th>B · NIS</th><td></td><td class="sg-num">{_fmt(data.get("nis"))}</td></tr>')
    rows.append(f'<tr><th>Fusion belief</th><td></td><td class="sg-num">{_fmt(data["belief"])}</td></tr>')
    rows.append(f'<tr><th>Plausibility</th><td></td><td class="sg-num">{_fmt(data["plausibility"])}</td></tr>')
    reasons = "".join(f"<li>{_E(r)}</li>" for r in data["reasons"])
    head = eyebrow_html("Drone inspector · t = {:.1f} s".format(data["t"]))
    return (
        f'<div class="sg-inspector sg-case"><span class="sg-case__tab">Inspector · {_E(data["label"])}</span>'
        f'<div class="sg-inspector__head">'
        f'<div>{head}'
        f'<p class="sg-inspector__name">{_E(data["label"])}</p></div>'
        f'{status_pill_html(data["verdict"])}</div>'
        f'<table class="sg-props">{"".join(rows)}</table>'
        f'{eyebrow_html("Why")}<ul class="sg-reasons">{reasons}</ul>'
        f'<div class="sg-action"><b>Recommended action.</b> {_E(data["action"])}</div></div>'
    )


def _icon(path_d: str, colour: str, css_class: str) -> str:
    """A 22 px stroke icon as an <img> (colours from the palette, never typed in)."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
           f'<path d="{path_d}" fill="none" stroke="{colour}" stroke-width="3" '
           f'stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return svg_img(svg, 22, "", f"sg-icon {css_class}")


_CHECK_SVG = _icon("M5 12.5 L10 17.5 L19 7", PALETTE.status_normal, "sg-check")
_CROSS_SVG = _icon("M6 6 L18 18 M18 6 L6 18", PALETTE.status_alarm, "sg-cross")
_DASH_SVG = _icon("M6 12 H18", PALETTE.status_suspicious, "sg-dash")


def verification_table_html(checks: Sequence[Any]) -> str:
    """Rows of (name, status, detail): PASS draws a check, FAIL shakes once and turns red."""
    rows = []
    for c in checks:
        status = str(c.status).upper()
        icon = {"PASS": _CHECK_SVG, "FAIL": _CROSS_SVG}.get(status, _DASH_SVG)
        cls = ' class="sg-verify--fail"' if status == "FAIL" else ""
        rows.append(f'<tr{cls}><td>{icon}</td><td class="sg-verify__name">{_E(c.name)}</td>'
                    f'<td>{status_pill_html(status.lower(), status)}</td>'
                    f'<td class="sg-verify__detail">{_E(c.detail)}</td></tr>')
    return f'<table class="sg-verify">{"".join(rows)}</table>'


LEGEND_SHAPES = ("dot", "diamond", "ring", "line")


def legend_html(items: Sequence[tuple[str, ...]]) -> str:
    """Legend with survey-style symbols.

    Each item is (label, CSS colour expression such as "var(--sg-amber)")
    or (label, colour, shape) with shape one of LEGEND_SHAPES (default dot).
    """
    spans = []
    for item in items:
        label, colour = item[0], item[1]
        shape = item[2] if len(item) > 2 else "dot"
        if shape not in LEGEND_SHAPES:
            raise ValueError(f"unknown legend shape {shape!r}; choose from {LEGEND_SHAPES}")
        style = f"border-color:{_E(colour)}" if shape in ("diamond", "ring") else f"background:{_E(colour)}"
        spans.append(f'<span><span class="sg-legend__swatch sg-legend__swatch--{shape}" style="{style}"></span>'
                     f'{_E(label)}</span>')
    return f'<div class="sg-legend">{"".join(spans)}</div>'


def feature_card_html(sticker_name: str, title: str, body: str, rotate: float = 0.0) -> str:
    """Feature card (hard shadow, crosshair corners, hover lift) with an illustration, title and copy.

    `rotate` is accepted for older callers but ignored: Ground Truth
    drawings sit square.
    """
    return (f'<div class="sg-card sg-card--feature sg-corners"><span class="sg-card__sticker">'
            f'{sticker_img(sticker_name, size=64)}</span>'
            f'<p class="sg-card__title">{_E(title)}</p><p class="sg-card__body">{_E(body)}</p></div>')


def empty_state_html(message: str, sticker_name: str = "radar") -> str:
    """Sticker + one sentence (the button is added by empty_state)."""
    return (f'<div class="sg-empty">{sticker_img(sticker_name, size=88)}'
            f'<p class="sg-empty__text">{_E(message)}</p></div>')


# --------------------------------------------------------------------------
# Ground Truth motifs (docs/phases/phase-7.md B4): hash ribbon, stamps,
# case files, survey dividers, custody timeline
# --------------------------------------------------------------------------
def utc_label(iso: str | None) -> str:
    """"2026-11-05T17:40:17+00:00" -> "2026-11-05 17:40 UTC" (other text is returned unchanged)."""
    if not iso:
        return "-"
    try:
        when = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return str(iso)
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc)
    return when.strftime("%Y-%m-%d %H:%M UTC")


def short_hash(digest: str, keep: int = 4) -> str:
    """"9f2c...e41a": first and last `keep` hex characters of a SHA-256 digest."""
    digest = str(digest)
    return digest if len(digest) <= 2 * keep + 1 else f"{digest[:keep]}…{digest[-keep:]}"


def hash_ribbon_html(entries: Sequence[RibbonEntry], slim: bool = False,
                     seconds_per_item: float = 7.0) -> str:
    """Motif M5: a thin espresso band slowly scrolling "SG-20261105-001 · 9f2c…e41a ·".

    The items are REAL manifest hashes (or, before any package exists, the
    hashes of the demo scenario configs, marked "config"). The track is
    doubled so the loop is seamless; style.css pauses it on hover.
    duration = seconds_per_item * number of items (so the speed stays calm).
    """
    if not entries:
        return ""
    items = []
    for e in entries:
        tag = "" if e.real else " · config"
        items.append(f'<span class="sg-hashribbon__item"><b>{_E(e.label)}</b>{tag} · '
                     f'<span title="{_E(e.sha256)}">{_E(short_hash(e.sha256))}</span></span>')
    track = '<span class="sg-hashribbon__sep">·</span>'.join(items) + '<span class="sg-hashribbon__sep">·</span>'
    slim_cls = " sg-hashribbon--slim" if slim else ""
    seconds = max(20.0, seconds_per_item * len(entries))
    return (f'<div class="sg-hashribbon{slim_cls}" aria-label="Recent evidence manifest hashes">'
            f'<div class="sg-hashribbon__track" style="--sg-ribbon-duration:{seconds:g}s">{track}{track}</div></div>')


STAMPS: dict[str, tuple[str, str]] = {
    # kind -> (word, palette colour)
    "verified": ("VERIFIED", PALETTE.olive),
    "tampered": ("TAMPERED", PALETTE.status_alarm),
    "filed": ("FILED", PALETTE.stone_text),
}
STAMP_W, STAMP_H = 220, 100


def stamp_svg(kind: str, case_id: str, utc: str | None = None) -> str:
    """Motif M7: a rubber-stamp mark as an SVG string.

    Double rounded border, the word in heavy mono capitals, and the case
    number + UTC time in tiny mono text inside the border. The rough ink
    edge is an SVG filter:
      feTurbulence      makes smooth random noise
      feDisplacementMap shifts each pixel of the stamp by that noise, so the
                        straight edges wobble like ink on paper
      a second, finer noise is turned into "holes" (feColorMatrix) and kept
                        only where both exist (feComposite "in"), so the ink
                        looks unevenly pressed.
    WHY a fixed seed: the same stamp always looks the same (rule R2 spirit).
    """
    if kind not in STAMPS:
        raise ValueError(f"unknown stamp {kind!r}; choose from {sorted(STAMPS)}")
    word, colour = STAMPS[kind]
    when = utc or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    small = _E(f"{case_id} · {when}" if case_id else when)
    mono = "'IBM Plex Mono','Courier New',monospace"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{STAMP_W}" height="{STAMP_H}" '
        f'viewBox="0 0 {STAMP_W} {STAMP_H}" role="img" aria-label="{word} stamp">'
        '<defs><filter id="ink" x="-6%" y="-6%" width="112%" height="112%">'
        '<feTurbulence type="fractalNoise" baseFrequency="0.05" numOctaves="2" seed="4" result="warp"/>'
        '<feDisplacementMap in="SourceGraphic" in2="warp" scale="2.4" xChannelSelector="R" '
        'yChannelSelector="G" result="rough"/>'
        '<feTurbulence type="fractalNoise" baseFrequency="1.1" numOctaves="1" seed="9" result="grain"/>'
        '<feColorMatrix in="grain" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -1.5 1.6" '
        'result="holes"/>'
        '<feComposite in="rough" in2="holes" operator="in"/></filter></defs>'
        f'<g filter="url(#ink)" fill="none" stroke="{colour}">'
        f'<rect x="5" y="5" width="{STAMP_W - 10}" height="{STAMP_H - 10}" rx="10" stroke-width="4"/>'
        f'<rect x="12" y="12" width="{STAMP_W - 24}" height="{STAMP_H - 24}" rx="6" stroke-width="1.4"/>'
        f'<text x="{STAMP_W / 2:g}" y="31" text-anchor="middle" font-family="{mono}" font-size="9" '
        f'letter-spacing="2" fill="{colour}" stroke="none">SWARMGUARD · EVIDENCE</text>'
        f'<text x="{STAMP_W / 2:g}" y="62" text-anchor="middle" font-family="{mono}" font-weight="700" '
        f'font-size="31" letter-spacing="3" fill="{colour}" stroke="none">{word}</text>'
        f'<line x1="26" y1="70" x2="{STAMP_W - 26}" y2="70" stroke-width="1"/>'
        f'<text x="{STAMP_W / 2:g}" y="82" text-anchor="middle" font-family="{mono}" font-size="8.5" '
        f'fill="{colour}" stroke="none">{small}</text></g></svg>'
    )


def stamp_html(kind: str, case_id: str, utc: str | None = None) -> str:
    """stamp_svg() as an <img> (st.html keeps <img> but removes inline <svg>).

    style.css lands it: scale 1.6 -> 1, rotate to -8 degrees, fade in with a
    slight overshoot (280 ms), and nudges the card 1 px (the "thud").
    """
    word = STAMPS.get(kind, ("", ""))[0]
    svg = stamp_svg(kind, case_id, utc)
    data = svg_img(svg, STAMP_W, f"{word} stamp, {case_id}".strip(", "), f"sg-stamp sg-stamp--{kind}")
    # svg_img writes a square size; the stamp is wider than it is tall.
    return data.replace(f'height="{STAMP_W}"', f'height="{STAMP_H}"', 1)


def case_card_html(case_id: str, title: str, filed_utc: str, pills: Sequence[str] = (),
                   detail: str | None = None, stamp: str | None = "filed") -> str:
    """Motif M6: an incident as a case-file card.

    A folder tab on the top edge holds the case number in mono; the card
    shows "FILED <time>", a title, pills and an optional status stamp.
    `pills` must be markup from this module's own builders (status_pill_html,
    detector_pill_html); every plain-text argument is escaped.
    """
    stamp_markup = f'<div class="sg-case__stamp">{stamp_html(stamp, case_id, filed_utc)}</div>' if stamp else ""
    detail_markup = f'<p class="sg-case__detail">{_E(detail)}</p>' if detail else ""
    return (f'<div class="sg-case sg-case--card"><span class="sg-case__tab">{_E(case_id)}</span>'
            f'<p class="sg-case__meta">Filed {_E(filed_utc)}</p>'
            f'<p class="sg-case__title">{_E(title)}</p>{detail_markup}'
            f'<div class="sg-pills">{"".join(pills)}</div>{stamp_markup}</div>')


def survey_divider_html(index: int, label: str | None = None) -> str:
    """Motif M8: a thin rule with a small crosshair in the middle and "§ 03 · 18.5204°N" on the right."""
    text = label if label is not None else f"{LAT0:.4f}°N"
    return (f'<div class="sg-survey" role="separator"><span class="sg-survey__rule"></span>'
            f'<span class="sg-survey__mark" aria-hidden="true"></span><span class="sg-survey__rule"></span>'
            f'<span class="sg-survey__label">§ {index:02d} · {_E(text)}</span></div>')


def custody_timeline_html(entries: Sequence[dict[str, Any]]) -> str:
    """Chain of custody as a vertical timeline with survey-mark nodes.

    Each entry is a custody_log row: seq, action, created_utc,
    manifest_sha256, previous_sha256. The arrow "<-" shows the hash link to
    the previous package (the hash chain).
    """
    if not entries:
        return '<p class="sg-card__body">No custody events yet.</p>'
    items = []
    for e in entries:
        action = str(e.get("action", "")).replace("_", " ").upper()
        items.append(
            f'<li class="sg-timeline__item"><span class="sg-timeline__node" aria-hidden="true"></span>'
            f'<div><p class="sg-timeline__head"><b>#{_E(str(e.get("seq", "")))} · {_E(action)}</b>'
            f'<span class="sg-timeline__time">{_E(str(e.get("created_utc", "")))}</span></p>'
            f'<p class="sg-hash">manifest {_E(short_hash(e.get("manifest_sha256", ""), 8))} '
            f'&larr; previous {_E(short_hash(e.get("previous_sha256", ""), 8))}</p></div></li>'
        )
    return f'<ol class="sg-timeline">{"".join(items)}</ol>'


# --------------------------------------------------------------------------
# Streamlit wrappers
# --------------------------------------------------------------------------
def html_block(markup: str) -> None:
    """Put an HTML string on the page."""
    st.html(markup)


def eyebrow(text: str) -> None:
    """Eyebrow label."""
    html_block(eyebrow_html(text))


def page_header(title: str, italic: str | None = None, eyebrow: str | None = None,
                caption: str | None = None) -> None:
    """Serif page title block."""
    html_block(page_header_html(title, italic, eyebrow, caption))


def section(title: str, caption: str | None = None, eyebrow: str | None = None,
            anchor: str | None = None) -> None:
    """Section heading."""
    html_block(section_html(title, caption, eyebrow, anchor))


def status_pill(status: str, label: str | None = None) -> None:
    """Status pill on its own line."""
    html_block(status_pill_html(status, label))


def kpi_card(label: str, value: int | float | str, status: str | None = None, delta: str | None = None) -> None:
    """One KPI card."""
    html_block(kpi_card_html(label, value, status, delta))


def kpi_row(items: Sequence[KPI]) -> None:
    """A row of KPI cards (stacks on narrow screens)."""
    html_block(kpi_row_html(items))


def navbar(active_page: str) -> None:
    """Context bar with the run status of the current pipeline (if any)."""
    pipe = st.session_state.get("pipeline")
    if pipe is None:
        status, text = "none", "No run yet"
    elif pipe.fusion.alarms().any():
        status, text = "alarm", "Fused alarm"
    elif any(r.flags().any() for r in pipe.runs):
        status, text = "suspicious", "Detector alarm"
    else:
        status, text = "normal", "No anomaly detected"
    scenario = st.session_state.get("scenario_key")
    seed = st.session_state.get("seed")
    tag = f"{scenario} · seed {seed}" if scenario else None
    html_block(navbar_html(active_page, status, text, tag))


def console_panel(title: str, body_fn: Callable[[], None], key: str | None = None,
                  live: str | None = None) -> None:
    """Charcoal instrument panel (the only dark surface) that can hold real widgets and charts.

    body_fn draws the content; it is called inside a keyed container that
    style.css paints as an instrument panel with crosshair corners.
    The name is kept from Phase 5 ("console") so every page keeps working;
    instrument_panel is the same function under its Ground Truth name.
    """
    with st.container(key=f"sg-panel-{key or _slug(title)}"):
        live_html = f'<span class="sg-panel__live">{_E(live)}</span>' if live else ""
        html_block(f'<div class="sg-panel__head"><p class="sg-panel__title">{_E(title)}</p>{live_html}</div>')
        body_fn()


instrument_panel = console_panel


def event_feed(events: Sequence[Event], key: str = "feed", limit: int = 12) -> None:
    """Event feed inside a instrument panel."""
    console_panel("Event feed", lambda: html_block(event_feed_html(events, limit)), key=key,
                  live=f"{len(events)} events")


def ribbon(events: Sequence[Event], seconds: float = 90.0) -> None:
    """Slow scrolling ribbon of recent events."""
    html_block(ribbon_html(events, seconds))


def hash_ribbon(slim: bool = False, limit: int = 8) -> None:
    """Motif M5: the scrolling ribbon of real evidence-manifest hashes (newest first)."""
    html_block(hash_ribbon_html(hash_ribbon_entries(limit), slim=slim))


def pattern_banner(title: str, body: str, tone: str = "alarm") -> None:
    """Pattern banner (tone "alarm" = brick bar, anything else = amber bar)."""
    html_block(pattern_banner_html(title, body, tone))


def drone_table(rows: pd.DataFrame, key: str | None = None) -> None:
    """Per-drone table: scores as progress bars or formatted numbers, status as text."""
    config = {
        "drone": st.column_config.TextColumn("Drone", width="small"),
        "A": st.column_config.TextColumn("A status"),
        "A_score": st.column_config.ProgressColumn("A disagreeing pairs", min_value=0.0, max_value=1.0,
                                                   format="percent"),
        "B": st.column_config.TextColumn("B status"),
        "B_cusum": st.column_config.NumberColumn("B CUSUM", format="%.1f"),
        "B_nis": st.column_config.NumberColumn("B NIS", format="%.1f"),
        "C": st.column_config.TextColumn("C status"),
        "C_residual_m": st.column_config.TextColumn("C residual (m)"),
        "verdict": st.column_config.TextColumn("Fused verdict"),
        "belief": st.column_config.ProgressColumn("Belief", min_value=0.0, max_value=1.0, format="%.2f"),
    }
    # WHY text for C: a drone without an anchor reading has no residual; the
    # table says "no reading" instead of a blank or "None" (rule R4).
    shown = rows.copy()
    if "C_residual_m" in shown:
        shown["C_residual_m"] = shown["C_residual_m"].map(lambda v: "no reading" if pd.isna(v) else f"{v:.2f}")
    st.dataframe(shown, hide_index=True, column_config=config, width="stretch", key=key)


def drone_inspector(data: dict[str, Any]) -> None:
    """Drone Inspector card."""
    html_block(drone_inspector_html(data))


def scenario_card(key: str, title: str, description: str, expected: dict[str, str],
                  sticker_name: str, button_label: str = "Run scenario") -> bool:
    """Feature card with a sticker, title, description, expected-outcome pills and a Run button.

    Returns True when its button was pressed.
    """
    with st.container(key=f"sg-card-{_slug(key)}"):
        pills = "".join(detector_pill_html(d, f"{d} · {o}") for d, o in sorted(expected.items()))
        pills = pills or status_pill_html("none", "No attack")
        html_block(f'{sticker_img(sticker_name, size=56)}<p class="sg-card__title">{_E(title)}</p>'
                   f'<p class="sg-card__body">{_E(description)}</p>'
                   f'<div class="sg-pills">{pills}</div>')
        return st.button(button_label, key=f"run-{_slug(key)}", width="stretch")


def verification_table(checks: Sequence[Any], key: str = "verify", overall: str | None = None,
                       case_id: str | None = None, utc: str | None = None) -> None:
    """Verification results inside an instrument panel (draw-on checks, shake on FAIL).

    When `overall` is given, the matching rubber stamp lands after the rows
    (VERIFIED on PASS, TAMPERED on FAIL), carrying the case number and time.
    """
    def body() -> None:
        markup = verification_table_html(checks)
        if overall is not None and overall.upper() in ("PASS", "FAIL"):
            kind = "verified" if overall.upper() == "PASS" else "tampered"
            markup += f'<div class="sg-stamp-slot">{stamp_html(kind, case_id or "", utc)}</div>'
        html_block(markup)

    console_panel("Verification", body, key=key)


def empty_state(message: str, action_label: str | None = None, sticker_name: str = "radar",
                key: str = "empty") -> bool:
    """Sticker + one sentence + (optionally) one button. Returns True if the button was pressed."""
    html_block(empty_state_html(message, sticker_name))
    if action_label is None:
        return False
    left, mid, right = st.columns([1, 1, 1])
    with mid:
        return st.button(action_label, key=f"sg-empty-{key}", type="primary", width="stretch")


def explain_box(text: str, title: str = "How to read this") -> None:
    """Collapsed "How to read this" expander with plain-language guidance."""
    with st.expander(title):
        st.markdown(text)


PLOTLY_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
# Maps (docs/phases/phase-7.md A1): the wheel zooms, dragging pans (the figure
# sets dragmode="pan"), and the box-zoom / selection tools are removed so a
# drag can never draw a zoom box. "Reset view" stays in the mode bar.
MAP_PLOTLY_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "zoom2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
CAMERA_OPTIONS: dict[str, str] = {"Follow swarm": "follow", "Whole flight": "whole"}


def chart(fig: go.Figure, key: str | None = None) -> None:
    """Show a Plotly figure with OUR template.

    WHY theme=None: Streamlit otherwise re-themes Plotly charts with its own
    defaults, which would override the Ground Truth templates.
    WHY two configs: a figure that pans when dragged (maps) also gets wheel
    zoom. Ordinary charts do not, because a wheel that zooms a chart would
    stop the page from scrolling past it.
    """
    is_map = fig.layout.dragmode == "pan"
    st.plotly_chart(fig, theme=None, width="stretch", config=MAP_PLOTLY_CONFIG if is_map else PLOTLY_CONFIG,
                    key=key)


def camera_control(key: str) -> str:
    """Segmented control "Follow swarm" / "Whole flight" above a map; returns "follow" or "whole".

    Follow swarm (default): the camera stays on the swarm's GPS centroid.
    Whole flight: one fixed camera showing every position of the run.
    If the user clicks the selected option again (which clears a segmented
    control), we fall back to the default instead of failing.
    """
    choice = st.segmented_control("Camera", list(CAMERA_OPTIONS), default="Follow swarm",
                                  key=f"sg-camera-{_slug(key)}", label_visibility="collapsed")
    return CAMERA_OPTIONS.get(choice or "Follow swarm", "follow")


def danger_button(label: str, key: str) -> bool:
    """Destructive-style (red) button; used ONLY for the tamper test."""
    with st.container(key=f"sg-danger-{_slug(key)}"):
        return st.button(label, key=key)


def require_run(page_key: str, message: str = "No simulation has run yet. Start with the quick start "
                "scenario, or pick one on the Home page.") -> PipelineResult | None:
    """The current run, or an empty state whose button runs the quick-start scenario.

    WHY one helper: every analysis page needs a run; this keeps the empty
    state (rule: never a blank page) identical everywhere.
    """
    pipe = get_or_run_pipeline()
    if pipe is not None:
        return pipe
    if empty_state(message, "Run quick start", sticker_name="radar", key=page_key):
        with st.spinner("Simulating, detecting, fusing and recovering..."):
            run_scenario(QUICK_START)
        st.rerun()
    return None


def require_ml_model(page_key: str) -> Any:
    """The trained experimental model, or an empty state with a "train quick model" button.

    A fresh clone (or a Streamlit Cloud deployment) has no model, because
    data/ is never committed. The button trains a small model in the app;
    `python cli.py train-ml --runs 300` makes the full one.
    """
    try:
        trained = get_ml_model()
    except ValueError as exc:
        st.error(f"The saved model was refused: {exc}")
        return None
    if trained is not None:
        return trained
    message = (f"No experimental model has been trained yet. Train a quick one here "
               f"({QUICK_MODEL_RUNS} simulated runs, about a minute), or run "
               "python cli.py train-ml --runs 300 for the full model.")
    if empty_state(message, "Train quick model", sticker_name="constellation", key=f"ml-{page_key}"):
        bar = st.progress(0.0, text="Simulating training runs...")
        train_quick_model(progress=lambda f: bar.progress(min(max(float(f), 0.0), 1.0),
                                                          text="Simulating training runs..."))
        st.rerun()
    return None


def paper_card(key: str):
    """A flat outlined paper card (no shadow) to put widgets, charts or equations in.

    Use as: with paper_card("equation"): st.latex(...)
    """
    return st.container(key=f"sg-paper-{_slug(key)}")
