"""Reusable UI components (docs/phases/phase-5.md section 5.9).

Every component comes in two parts:
    *_html(...)   a PURE function that returns an HTML string (unit-tested)
    wrapper(...)  a thin function that puts that HTML on the page with st.html,
                  or combines it with real Streamlit widgets

WHY split them: the HTML can be tested without starting Streamlit, and the
wrappers stay tiny. All text is passed through html.escape(), so a drone
name or a file path can never inject markup.

WHY st.html and not st.markdown: markdown treats indentation and blank
lines specially and can break raw HTML; st.html inserts it as written.

Widgets cannot live inside a raw HTML <div>, so cards and console panels
that contain buttons, sliders or charts use st.container(key="sg-...").
Streamlit adds the class "st-key-<key>" to such a container, and
assets/style.css styles every "st-key-sg-console*" / "st-key-sg-card*".

Status is always shown as text (rule from CLAUDE.md), never colour alone.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.stickers import drone_mark, sticker
from analysis.experiments import PipelineResult
from ui.state import QUICK_START, Event, get_or_run_pipeline, run_scenario
from ui.theme import DETECTOR_NAMES, status_label, status_tone

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
    """Line-art drone + "SWARM" (Inter 300, spaced) + "GUARD" (Bricolage 800)."""
    return (f'<span class="sg-wordmark" aria-label="SwarmGuard">{drone_mark(size)}'
            f'<span class="sg-wordmark__swarm">SWARM</span><span class="sg-wordmark__guard">GUARD</span></span>')


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
    """Section heading (Bricolage 700) with optional eyebrow, caption and anchor id."""
    anchor_attr = f' id="{_E(anchor)}"' if anchor else ""
    parts = [f'<div class="sg-section"{anchor_attr}>']
    if eyebrow:
        parts.append(eyebrow_html(eyebrow))
    parts.append(f'<h2 class="sg-section__title">{_E(title)}</h2>')
    if caption:
        parts.append(f'<p class="sg-section__caption">{_E(caption)}</p>')
    parts.append("</div>")
    return "".join(parts)


def navbar_html(active_page: str, status: str | None, status_text: str | None,
                scenario: str | None) -> str:
    """Context bar under the top navigation: wordmark, page name, status pill, scenario."""
    right = []
    if status is not None:
        right.append(status_pill_html(status, status_text))
    if scenario:
        right.append(f'<span class="sg-scenario-tag">{_E(scenario)}</span>')
    return (f'<div class="sg-topbar"><div>{wordmark_html()}</div>'
            f'<div class="sg-topbar__right"><span class="sg-eyebrow">{_E(active_page)}</span>'
            f'{"".join(right)}</div></div>')


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
    """KPI card with Bricolage numerals; whole numbers count up (CSS @property).

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
    """Black band of scrolling events. Content is doubled so the loop is seamless."""
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
    """Outlined properties card for one drone (data from ui.state.inspector_data)."""
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
        f'<div class="sg-inspector"><div class="sg-inspector__head">'
        f'<div>{head}'
        f'<p class="sg-inspector__name">{_E(data["label"])}</p></div>'
        f'{status_pill_html(data["verdict"])}</div>'
        f'<table class="sg-props">{"".join(rows)}</table>'
        f'{eyebrow_html("Why")}<ul class="sg-reasons">{reasons}</ul>'
        f'<div class="sg-action"><b>Recommended action.</b> {_E(data["action"])}</div></div>'
    )


_CHECK_SVG = '<svg class="sg-check" width="22" height="22" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5 L10 17.5 L19 7"/></svg>'
_CROSS_SVG = '<svg class="sg-cross" width="22" height="22" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6 L18 18 M18 6 L6 18"/></svg>'
_DASH_SVG = '<svg class="sg-dash" width="22" height="22" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 12 H18"/></svg>'


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


def legend_html(items: Sequence[tuple[str, str]]) -> str:
    """Colour legend; items are (label, CSS colour expression such as "var(--sg-mint)")."""
    spans = "".join(f'<span><span class="sg-legend__swatch" style="background:{_E(colour)}"></span>{_E(label)}</span>'
                    for label, colour in items)
    return f'<div class="sg-legend">{spans}</div>'


def feature_card_html(sticker_name: str, title: str, body: str, rotate: float = 0.0) -> str:
    """Feature card (hard shadow, hover lift) with a sticker, title and two lines of copy."""
    return (f'<div class="sg-card sg-card--feature"><span class="sg-card__sticker">'
            f'{sticker(sticker_name, size=64, rotate=rotate)}</span>'
            f'<p class="sg-card__title">{_E(title)}</p><p class="sg-card__body">{_E(body)}</p></div>')


def empty_state_html(message: str, sticker_name: str = "radar") -> str:
    """Sticker + one sentence (the button is added by empty_state)."""
    return (f'<div class="sg-empty">{sticker(sticker_name, size=88, rotate=-3)}'
            f'<p class="sg-empty__text">{_E(message)}</p></div>')


# --------------------------------------------------------------------------
# Streamlit wrappers
# --------------------------------------------------------------------------
def html_block(markup: str) -> None:
    """Put an HTML string on the page."""
    st.html(markup)


def eyebrow(text: str) -> None:
    """Eyebrow label."""
    st.html(eyebrow_html(text))


def page_header(title: str, italic: str | None = None, eyebrow: str | None = None,
                caption: str | None = None) -> None:
    """Serif page title block."""
    st.html(page_header_html(title, italic, eyebrow, caption))


def section(title: str, caption: str | None = None, eyebrow: str | None = None,
            anchor: str | None = None) -> None:
    """Section heading."""
    st.html(section_html(title, caption, eyebrow, anchor))


def status_pill(status: str, label: str | None = None) -> None:
    """Status pill on its own line."""
    st.html(status_pill_html(status, label))


def kpi_card(label: str, value: int | float | str, status: str | None = None, delta: str | None = None) -> None:
    """One KPI card."""
    st.html(kpi_card_html(label, value, status, delta))


def kpi_row(items: Sequence[KPI]) -> None:
    """A row of KPI cards (stacks on narrow screens)."""
    st.html(kpi_row_html(items))


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
    st.html(navbar_html(active_page, status, text, tag))


def console_panel(title: str, body_fn: Callable[[], None], key: str | None = None,
                  live: str | None = None) -> None:
    """Dark console panel that can hold real widgets and charts.

    body_fn draws the content; it is called inside a keyed container that
    style.css paints as a console panel.
    """
    with st.container(key=f"sg-console-{key or _slug(title)}"):
        live_html = f'<span class="sg-console__live">{_E(live)}</span>' if live else ""
        st.html(f'<div class="sg-console__head"><p class="sg-console__title">{_E(title)}</p>{live_html}</div>')
        body_fn()


def event_feed(events: Sequence[Event], key: str = "feed", limit: int = 12) -> None:
    """Event feed inside a console panel."""
    console_panel("Event feed", lambda: st.html(event_feed_html(events, limit)), key=key,
                  live=f"{len(events)} events")


def ribbon(events: Sequence[Event], seconds: float = 90.0) -> None:
    """Slow scrolling ribbon of recent events."""
    st.html(ribbon_html(events, seconds))


def pattern_banner(title: str, body: str, tone: str = "alarm") -> None:
    """Pattern banner (tone "alarm" = red bar, anything else = forest bar)."""
    st.html(pattern_banner_html(title, body, tone))


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
        "C_residual_m": st.column_config.NumberColumn("C residual (m)", format="%.2f"),
        "verdict": st.column_config.TextColumn("Fused verdict"),
        "belief": st.column_config.ProgressColumn("Belief", min_value=0.0, max_value=1.0, format="%.2f"),
    }
    st.dataframe(rows, hide_index=True, column_config=config, width="stretch", key=key)


def drone_inspector(data: dict[str, Any]) -> None:
    """Drone Inspector card."""
    st.html(drone_inspector_html(data))


def scenario_card(key: str, title: str, description: str, expected: dict[str, str],
                  sticker_name: str, button_label: str = "Run scenario") -> bool:
    """Feature card with a sticker, title, description, expected-outcome pills and a Run button.

    Returns True when its button was pressed.
    """
    with st.container(key=f"sg-card-{_slug(key)}"):
        pills = "".join(detector_pill_html(d, f"{d} · {o}") for d, o in sorted(expected.items()))
        pills = pills or status_pill_html("none", "No attack")
        st.html(f'{sticker(sticker_name, size=56, rotate=-2)}<p class="sg-card__title">{_E(title)}</p>'
                f'<p class="sg-card__body">{_E(description)}</p>'
                f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px">{pills}</div>')
        return st.button(button_label, key=f"run-{_slug(key)}", width="stretch")


def verification_table(checks: Sequence[Any], key: str = "verify") -> None:
    """Verification results inside a console panel (draw-on checks, shake on FAIL)."""
    console_panel("Verification", lambda: st.html(verification_table_html(checks)), key=key)


def empty_state(message: str, action_label: str | None = None, sticker_name: str = "radar",
                key: str = "empty") -> bool:
    """Sticker + one sentence + (optionally) one button. Returns True if the button was pressed."""
    st.html(empty_state_html(message, sticker_name))
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


def chart(fig: go.Figure, key: str | None = None) -> None:
    """Show a Plotly figure with OUR template.

    WHY theme=None: Streamlit otherwise re-themes Plotly charts with its own
    defaults, which would override the swarmguard templates.
    """
    st.plotly_chart(fig, theme=None, width="stretch", config=PLOTLY_CONFIG, key=key)


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


def paper_card(key: str):
    """A flat outlined paper card (no shadow) to put widgets, charts or equations in.

    Use as: with paper_card("equation"): st.latex(...)
    """
    return st.container(key=f"sg-paper-{_slug(key)}")
