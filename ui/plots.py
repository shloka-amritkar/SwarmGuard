"""Plotly figures and templates for the app (docs/phases/phase-5.md section 5.10).

TWO TEMPLATES
    swarmguard_light    charts on paper cards: transparent background, Inter,
                        1px ink axes, light dashed gridlines, muted ticks,
                        2.25px lines, horizontal legend above, no titles.
    swarmguard_console  charts inside dark console panels: console
                        background, console_ln gridlines, console_tx labels,
                        mint as the neutral trace.
Every colour comes from core/palette.py. Detectors always use their fixed
colours (A blue, B teal, C ochre); red is used only for alarms.

THE SWARM MAP
Always the dark "carto-darkmatter" basemap over Pune, inside a console
panel. Positions are converted from metres with enu_to_latlon (rule R3,
display only). Animated maps use one frame every 5 steps.
WHY every 5 steps: 600 frames x 20 drones makes the browser lag; 120
frames at 0.5 s spacing still look smooth.
Map markers cannot have outlines or custom shapes, so outlines are drawn as
a slightly larger marker underneath, and anchored drones get a mint ring.
The "offline" grid version (no internet tiles) can use real outlines,
diamonds for anchored drones and dashed truth lines.

EVALUATION VIEW (rule R1): true positions are shown only when the caller
asks for it, labelled "Evaluation view: ground truth".
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from analysis.experiments import PipelineResult
from core.schema import ObservedFrame, SimResult
from core.utils import LAT0, drone_label, enu_to_latlon
from detection.base import DetectionRun
from detection.detector_a_geometry import geometry_residuals
from detection.fusion import BELIEF_COMPROMISED, PLAUSIBILITY_NORMAL
from ui.theme import DETECTOR_NAMES, FONT_MONO, FONT_SANS, PALETTE, rgba

logger = logging.getLogger(__name__)

P = PALETTE
LIGHT = "swarmguard_light"
CONSOLE = "swarmguard_console"
LINE_W = 2.25
MAP_STYLE = "carto-darkmatter"
FRAME_EVERY = 5
TRAIL_STEPS = 20
TRANSPARENT = "rgba(0,0,0,0)"


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------
def _axis(line: str, grid: str, tick: str, dashed: bool) -> dict[str, Any]:
    """Axis style shared by both templates."""
    return {"showline": True, "linecolor": line, "linewidth": 1, "gridcolor": grid,
            "griddash": "dash" if dashed else "solid", "zeroline": False, "ticks": "outside",
            "tickcolor": line, "ticklen": 4, "tickfont": {"color": tick, "size": 12},
            "title": {"font": {"color": tick, "size": 12}}, "automargin": True}


def _template(bg: str, text: str, line: str, grid: str, tick: str, neutral: str, dashed: bool) -> go.layout.Template:
    """Build one Plotly template."""
    colorway = [neutral, P.det_a, P.det_b, P.det_c, P.det_d, P.forest_2, P.muted]
    layout = go.Layout(
        paper_bgcolor=bg, plot_bgcolor=bg, colorway=colorway,
        font={"family": FONT_SANS, "color": text, "size": 13},
        xaxis=_axis(line, grid, tick, dashed), yaxis=_axis(line, grid, tick, dashed),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
                "bgcolor": TRANSPARENT, "font": {"color": text, "size": 12}},
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        hoverlabel={"bgcolor": P.paper, "bordercolor": P.ink,
                    "font": {"family": FONT_SANS, "color": P.ink, "size": 12}},
        transition={"duration": 300, "easing": "cubic-in-out"},
        title={"text": ""},
    )
    data = {"scatter": [go.Scatter(line={"width": LINE_W})]}
    return go.layout.Template(layout=layout, data=data)


def register_templates() -> None:
    """Register both templates with Plotly (safe to call many times)."""
    pio.templates[LIGHT] = _template(TRANSPARENT, P.ink, P.ink, P.line, P.muted, P.forest, True)
    pio.templates[CONSOLE] = _template(P.console, P.console_tx, P.console_ln, P.console_ln,
                                       P.console_mu, P.mint, False)


register_templates()


def _threshold(fig: go.Figure, y: float, label: str, row: int | None = None, col: int | None = None) -> None:
    """Dashed ink threshold line with a right-aligned label."""
    kwargs: dict[str, Any] = {"row": row, "col": col} if row is not None else {}
    fig.add_hline(y=y, line={"color": P.ink, "dash": "dash", "width": 1},
                  annotation_text=label, annotation_position="top right",
                  annotation_font={"size": 11, "color": P.ink_2}, **kwargs)


def _onset(fig: go.Figure, onset: float | None, label: str = "simulation onset") -> None:
    """Vertical dotted line at the attack onset (EVALUATION: from the simulation truth)."""
    if onset is None:
        return
    fig.add_vline(x=onset, line={"color": P.muted, "dash": "dot", "width": 1.2},
                  annotation_text=label, annotation_position="top left",
                  annotation_font={"size": 11, "color": P.muted})


def status_colour(verdict: str) -> str:
    """Fill colour for a fused verdict / status string."""
    return P.status_colour(verdict)


# --------------------------------------------------------------------------
# Swarm map
# --------------------------------------------------------------------------
def _zoom_for(extent_m: float, lat: float) -> float:
    """Map zoom so that extent_m (plus margin) fits about 640 px.

        metres per pixel at zoom z = 156543.03 * cos(lat) / 2^z
    """
    mpp = max(extent_m, 30.0) * 1.5 / 640.0
    return float(np.clip(math.log2(156543.03 * math.cos(math.radians(lat)) / mpp), 10.0, 20.0))


def _nearest_links(pos: np.ndarray, k: int = 2) -> tuple[list[float | None], list[float | None]]:
    """Line segments from each drone to its k nearest neighbours, None-separated."""
    n = len(pos)
    xs: list[float | None] = []
    ys: list[float | None] = []
    if n < 2:
        return xs, ys
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        for j in np.argsort(d[i])[: min(k, n - 1)]:
            pair = (min(i, int(j)), max(i, int(j)))
            if pair in seen:
                continue
            seen.add(pair)
            xs += [pos[i, 0], pos[j, 0], None]
            ys += [pos[i, 1], pos[j, 1], None]
    return xs, ys


def _frame_layers(result: SimResult, statuses: np.ndarray, k: int, links: bool, evaluation: bool,
                  light_show: bool, trail: int) -> list[dict[str, Any]]:
    """Plain-data layers for step k: dicts with x, y (metres), colour, size, kind.

    The same layers feed the map (lat/lon) and the offline grid figure.
    """
    frame = result.frames[k]
    gps = frame.observed.gps_pos
    n = result.n_drones
    colours = [status_colour(s) for s in statuses[k]]
    labels = [f"{drone_label(i)} · {str(statuses[k][i]).replace('_', ' ').upper()}" for i in range(n)]
    layers: list[dict[str, Any]] = []
    lx, ly = _nearest_links(gps) if links else ([], [])
    layers.append({"kind": "links", "x": lx, "y": ly})
    lo = max(0, k - trail)
    hist = np.stack([f.observed.gps_pos for f in result.frames[lo:k + 1]])        # (L, N, 2)
    ages = np.repeat(np.arange(hist.shape[0])[:, None], n, axis=1).reshape(-1)
    alpha = 0.08 + 0.55 * (ages + 1) / hist.shape[0]
    layers.append({"kind": "trails", "x": hist[:, :, 0].reshape(-1), "y": hist[:, :, 1].reshape(-1),
                   "colour": [rgba(P.mint, float(a)) for a in alpha]})
    if light_show:
        layers.append({"kind": "halo", "x": gps[:, 0], "y": gps[:, 1], "colour": [rgba(c, 0.22) for c in colours]})
    if evaluation:
        truth = frame.true_pos
        cx: list[float | None] = []
        cy: list[float | None] = []
        for i in range(n):
            cx += [gps[i, 0], truth[i, 0], None]
            cy += [gps[i, 1], truth[i, 1], None]
        layers.append({"kind": "truth_links", "x": cx, "y": cy})
        layers.append({"kind": "truth", "x": truth[:, 0], "y": truth[:, 1],
                       "text": [f"{drone_label(i)} · true position (evaluation)" for i in range(n)]})
    anchored = np.flatnonzero(frame.anchored)
    layers.append({"kind": "anchors", "x": gps[anchored, 0], "y": gps[anchored, 1]})
    layers.append({"kind": "drones", "x": gps[:, 0], "y": gps[:, 1], "colour": colours, "text": labels,
                   "anchored": frame.anchored})
    return layers


def _map_traces(layers: list[dict[str, Any]]) -> list[go.Scattermap]:
    """Turn layers into map traces (lat/lon). Fixed trace order across frames."""
    def ll(xs: Sequence[float | None], ys: Sequence[float | None]) -> tuple[list[Any], list[Any]]:
        lats, lons = [], []
        for x, y in zip(xs, ys):
            if x is None or y is None:
                lats.append(None)
                lons.append(None)
            else:
                lat, lon = enu_to_latlon(float(x), float(y))
                lats.append(float(lat))
                lons.append(float(lon))
        return lats, lons

    traces: list[go.Scattermap] = []
    for layer in layers:
        lat, lon = ll(list(layer["x"]), list(layer["y"]))
        kind = layer["kind"]
        if kind in ("links", "truth_links"):
            colour = rgba(P.mint, 0.2) if kind == "links" else rgba(P.console_tx, 0.45)
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="lines", line={"color": colour, "width": 1},
                                        hoverinfo="skip", name=kind))
        elif kind == "trails":
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", hoverinfo="skip", name=kind,
                                        marker={"size": 4, "color": layer["colour"]}))
        elif kind == "halo":
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", hoverinfo="skip", name=kind,
                                        marker={"size": 30, "color": layer["colour"]}))
        elif kind == "truth":
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", hoverinfo="skip", name="truth ring",
                                        marker={"size": 13, "color": P.console_tx}))
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", name="truth", text=layer["text"],
                                        hovertemplate="%{text}<extra></extra>",
                                        marker={"size": 9, "color": P.console}))
        elif kind == "anchors":
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", hoverinfo="skip", name="anchor ring",
                                        marker={"size": 21, "color": P.mint}))
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", hoverinfo="skip", name="anchor gap",
                                        marker={"size": 17, "color": P.console}))
        elif kind == "drones":
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", hoverinfo="skip", name="outline",
                                        marker={"size": 13, "color": P.ivory}))
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", name="drones", text=layer["text"],
                                        hovertemplate="%{text}<extra></extra>",
                                        marker={"size": 11, "color": layer["colour"]}))
    return traces


def _grid_traces(layers: list[dict[str, Any]]) -> list[go.Scatter]:
    """Offline version on a metric grid: real outlines, diamonds for anchors, dashed truth lines."""
    traces: list[go.Scatter] = []
    for layer in layers:
        kind, x, y = layer["kind"], list(layer["x"]), list(layer["y"])
        if kind in ("links", "truth_links"):
            traces.append(go.Scatter(x=x, y=y, mode="lines", hoverinfo="skip", name=kind,
                                     line={"color": rgba(P.mint, 0.2) if kind == "links" else rgba(P.console_tx, 0.5),
                                           "width": 1, "dash": "solid" if kind == "links" else "dash"}))
        elif kind == "trails":
            traces.append(go.Scatter(x=x, y=y, mode="markers", hoverinfo="skip", name=kind,
                                     marker={"size": 4, "color": layer["colour"]}))
        elif kind == "halo":
            traces.append(go.Scatter(x=x, y=y, mode="markers", hoverinfo="skip", name=kind,
                                     marker={"size": 30, "color": layer["colour"]}))
        elif kind == "truth":
            traces.append(go.Scatter(x=x, y=y, mode="markers", name="truth", text=layer["text"],
                                     hovertemplate="%{text}<extra></extra>",
                                     marker={"size": 11, "symbol": "circle-open", "color": P.console_tx,
                                             "line": {"width": 1.5}}))
        elif kind == "drones":
            symbols = ["diamond" if a else "circle" for a in layer["anchored"]]
            traces.append(go.Scatter(x=x, y=y, mode="markers", name="drones", text=layer["text"],
                                     hovertemplate="%{text}<extra></extra>",
                                     marker={"size": 13, "color": layer["colour"], "symbol": symbols,
                                             "line": {"color": P.ivory, "width": 1}}))
    return traces


def _animation_controls(labels: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Play/pause buttons and a styled time slider for the console panel."""
    frame_args = {"frame": {"duration": 160, "redraw": True}, "mode": "immediate",
                  "transition": {"duration": 0}, "fromcurrent": True}
    buttons = [{
        "type": "buttons", "direction": "left", "x": 0.0, "y": -0.04, "xanchor": "left", "yanchor": "top",
        "pad": {"r": 8, "t": 8}, "showactive": False, "bgcolor": P.console_2, "bordercolor": P.console_ln,
        "font": {"color": P.console_tx, "family": FONT_SANS, "size": 12},
        "buttons": [{"label": "Play", "method": "animate", "args": [None, frame_args]},
                    {"label": "Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}],
    }]
    slider = [{
        "active": 0, "x": 0.16, "y": -0.02, "len": 0.84, "xanchor": "left", "yanchor": "top",
        "pad": {"t": 4, "b": 4}, "bgcolor": P.console_2, "activebgcolor": P.mint, "bordercolor": P.console_ln,
        "tickcolor": P.console_ln, "font": {"color": P.console_mu, "family": FONT_MONO, "size": 10},
        "currentvalue": {"prefix": "t = ", "suffix": " s", "font": {"color": P.mint, "family": FONT_MONO, "size": 12}},
        "steps": [{"label": lab, "method": "animate",
                   "args": [[lab], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                  for lab in labels],
    }]
    return buttons, slider


def swarm_map(
    result: SimResult,
    statuses: np.ndarray,
    step: int | None = None,
    every: int = FRAME_EVERY,
    links: bool = True,
    evaluation: bool = False,
    offline: bool = False,
    light_show: bool = False,
    height: int = 520,
    trail: int = TRAIL_STEPS,
) -> go.Figure:
    """Swarm map in console style.

    statuses  (T, N) status/verdict strings that colour the drones
    step      draw only this step (no animation); None = animate every `every` steps
    evaluation  also draw true positions (EVALUATION VIEW, rule R1)
    offline   plain metric grid instead of map tiles
    light_show larger low-opacity layer under each drone (night lights)
    """
    if statuses.shape != (result.n_steps, result.n_drones):
        raise ValueError(f"statuses must have shape {(result.n_steps, result.n_drones)}, got {statuses.shape}")
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    make = _grid_traces if offline else _map_traces
    steps = [step] if step is not None else list(range(0, result.n_steps, every))
    if step is None and steps[-1] != result.n_steps - 1:
        steps.append(result.n_steps - 1)
    first = make(_frame_layers(result, statuses, steps[0], links, evaluation, light_show, trail))
    fig = go.Figure(data=first)
    times = result.times()
    if len(steps) > 1:
        labels = [f"{times[k]:.1f}" for k in steps]
        fig.frames = [go.Frame(name=lab, data=make(_frame_layers(result, statuses, k, links, evaluation,
                                                                  light_show, trail)))
                      for k, lab in zip(steps, labels)]
        buttons, slider = _animation_controls(labels)
        fig.update_layout(updatemenus=buttons, sliders=slider)
    gps = result.stack("gps_pos")
    centre = gps.reshape(-1, 2).mean(axis=0)
    extent = float(np.ptp(gps.reshape(-1, 2), axis=0).max())
    fig.update_layout(template=CONSOLE, height=height, showlegend=False,
                      margin={"l": 0, "r": 0, "t": 0, "b": 64 if len(steps) > 1 else 0})
    if offline:
        fig.update_xaxes(title="East (m)", scaleanchor="y", scaleratio=1)
        fig.update_yaxes(title="North (m)")
    else:
        lat, lon = enu_to_latlon(float(centre[0]), float(centre[1]))
        fig.update_layout(map={"style": MAP_STYLE, "center": {"lat": float(lat), "lon": float(lon)},
                               "zoom": _zoom_for(extent, LAT0)})
    return fig


# --------------------------------------------------------------------------
# Detection analysis
# --------------------------------------------------------------------------
def _runs(pipe: PipelineResult) -> dict[str, DetectionRun]:
    """Detector runs by letter."""
    return {r.short: r for r in pipe.runs}


def detector_timeseries(pipe: PipelineResult, drone: int, onset: float | None = None) -> go.Figure:
    """Four stacked panels for one drone: A vote share, B NIS (log), B CUSUM, C residual.

    Each panel carries its threshold as a dashed line. onset (optional) is
    the simulation's true attack start, labelled "simulation onset".
    """
    runs, det = _runs(pipe), pipe.det_cfg
    t = pipe.sim_result.times()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    a = runs["A"].scores()[:, drone]
    fig.add_trace(go.Scatter(x=t, y=a, name="A vote share", line={"color": P.det_a}), row=1, col=1)
    _threshold(fig, det.vote_fraction, f"vote {det.vote_fraction:.2f}", 1, 1)
    nis = np.array([s.diagnostics.get("nis", np.full(pipe.sim_result.n_drones, np.nan))[drone]
                    for s in runs["B"].steps], dtype=float)
    nis = np.where(nis > 0, nis, np.nan)
    fig.add_trace(go.Scatter(x=t, y=nis, name="B NIS", line={"color": P.det_b, "width": 1.5}), row=2, col=1)
    _threshold(fig, det.nis_threshold, f"NIS {det.nis_threshold:.2f}", 2, 1)
    fig.add_trace(go.Scatter(x=t, y=runs["B"].scores()[:, drone], name="B CUSUM",
                             line={"color": P.det_b}), row=3, col=1)
    _threshold(fig, det.cusum_h, f"h {det.cusum_h:.1f}", 3, 1)
    c = runs["C"].scores()[:, drone]
    fig.add_trace(go.Scatter(x=t, y=c, name="C residual", line={"color": P.det_c}), row=4, col=1)
    _threshold(fig, det.tau_c, f"tau_c {det.tau_c:.1f} m", 4, 1)
    if np.isnan(c).all():
        fig.add_annotation(text="NO EVIDENCE: no anchor readings for this drone", x=0.5, y=0.5,
                           xref="x4 domain", yref="y4 domain", showarrow=False,
                           font={"color": P.muted, "size": 12})
    _onset(fig, onset)
    fig.update_yaxes(title_text="A vote share", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="B NIS", type="log", row=2, col=1)
    fig.update_yaxes(title_text="B CUSUM", row=3, col=1)
    fig.update_yaxes(title_text="C residual (m)", row=4, col=1)
    fig.update_xaxes(title_text="time (s)", row=4, col=1)
    fig.update_layout(template=LIGHT, height=720, showlegend=False)
    return fig


def residual_heatmap(obs: ObservedFrame, tau_a: float) -> go.Figure:
    """Detector A's pair residuals |GPS distance - radio range| at one step (paper -> blush -> red)."""
    residual, _ = geometry_residuals(obs)
    names = [drone_label(i) for i in range(obs.n)]
    zmax = max(2.0 * tau_a, float(np.nanmax(residual)) if np.isfinite(residual).any() else 0.0)
    fig = go.Figure(go.Heatmap(
        z=residual, x=names, y=names, zmin=0, zmax=zmax,
        colorscale=[[0.0, P.paper], [min(tau_a / zmax, 0.99), P.blush], [1.0, P.status_alarm]],
        colorbar={"title": {"text": "residual (m)", "side": "right"}, "outlinewidth": 1, "outlinecolor": P.ink},
        hovertemplate="%{y} to %{x}: %{z:.2f} m<extra></extra>", xgap=1, ygap=1,
    ))
    fig.update_layout(template=LIGHT, height=460)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return fig


def fusion_chart(pipe: PipelineResult, drone: int, onset: float | None = None) -> go.Figure:
    """Belief and plausibility over time for one drone, with the verdict thresholds."""
    t = pipe.fusion.times
    bel, pl = pipe.fusion.belief[:, drone], pipe.fusion.plausibility[:, drone]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=pl, name="plausibility", line={"color": P.forest, "width": 1.5}))
    fig.add_trace(go.Scatter(x=t, y=bel, name="belief", fill="tonexty", fillcolor=rgba(P.forest, 0.10),
                             line={"color": P.ink}))
    _threshold(fig, BELIEF_COMPROMISED, f"compromised if belief >= {BELIEF_COMPROMISED}")
    _threshold(fig, PLAUSIBILITY_NORMAL, f"normal if plausibility < {PLAUSIBILITY_NORMAL}")
    _onset(fig, onset)
    fig.update_yaxes(range=[0, 1.02], title="mass")
    fig.update_xaxes(title="time (s)")
    fig.update_layout(template=LIGHT, height=320)
    return fig


def alarm_timeline(runs: Sequence[DetectionRun], onset: float | None = None,
                   onset_label: str = "estimated onset") -> go.Figure:
    """One row per (detector, alarmed drone): thick bars while in ALARM."""
    rows = [(r, i) for r in runs for i in sorted(r.alarm_drones())]
    fig = go.Figure()
    for y, (run, i) in enumerate(rows):
        times, flags = run.times(), run.flags()[:, i]
        x = np.where(flags, times, np.nan)
        fig.add_trace(go.Scatter(x=x, y=[y] * len(x), mode="lines", connectgaps=False,
                                 line={"color": P.detector_colour(run.short), "width": 11},
                                 name=DETECTOR_NAMES.get(run.short, run.short), hoverinfo="skip",
                                 showlegend=False))
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=[f"{r.short} · {drone_label(i)}" for r, i in rows],
                     autorange="reversed", showgrid=False)
    if not rows:
        fig.add_annotation(text="No detector alarm in this run", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font={"color": P.muted})
    _onset(fig, onset, onset_label)
    fig.update_xaxes(title="time (s)")
    fig.update_layout(template=LIGHT, height=max(220, 60 + 24 * len(rows)))
    return fig


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def comparison_bar(summary: pd.DataFrame, value_col: str = "detection_rate") -> go.Figure:
    """Rate per detector with asymmetric Wilson 95 % error bars.

    summary needs columns detector, <value_col>, and (if present)
    detection_ci_low / detection_ci_high (or false_alarm_ci_*).
    """
    prefix = "detection" if value_col == "detection_rate" else "false_alarm"
    lo, hi = summary.get(f"{prefix}_ci_low"), summary.get(f"{prefix}_ci_high")
    y = summary[value_col]
    err = None
    if lo is not None and hi is not None:
        err = {"type": "data", "symmetric": False, "array": (hi - y).tolist(), "arrayminus": (y - lo).tolist(),
               "color": P.ink, "thickness": 1.2, "width": 6}
    colours = [P.detector_colour(d) for d in summary["detector"]]
    fig = go.Figure(go.Bar(x=[DETECTOR_NAMES.get(d, d) for d in summary["detector"]], y=y, error_y=err,
                           marker={"color": colours, "line": {"color": P.ink, "width": 1.5}},
                           hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_layout(template=LIGHT, height=340, bargap=0.45)
    return fig


def detection_curve(summary: pd.DataFrame, param: str, detectors: Sequence[str] = ("A", "B", "C", "F")) -> go.Figure:
    """Detection rate vs parameter value per detector, with its 95 % Wilson band."""
    fig = go.Figure()
    for d in detectors:
        g = summary[summary["detector"] == d].sort_values("value")
        if g.empty:
            continue
        colour = P.detector_colour(d)
        x = g["value"].tolist()
        fig.add_trace(go.Scatter(x=x + x[::-1], y=g["detection_ci_high"].tolist() + g["detection_ci_low"].tolist()[::-1],
                                 fill="toself", fillcolor=rgba(colour, 0.12), line={"width": 0},
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=g["detection_rate"], name=f"{d} · {DETECTOR_NAMES.get(d, d)}",
                                 mode="lines+markers", line={"color": colour}, marker={"size": 7}))
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", title="detection rate")
    fig.update_xaxes(title=param)
    fig.update_layout(template=LIGHT, height=380)
    return fig


def delay_box(df: pd.DataFrame) -> go.Figure:
    """Detection delay distribution per detector (detected runs only)."""
    fig = go.Figure()
    for d in ("A", "B", "C", "F"):
        delays = df[(df["detector"] == d) & df["delay"].notna()]["delay"]
        if delays.empty:
            continue
        colour = P.detector_colour(d)
        fig.add_trace(go.Box(y=delays, name=DETECTOR_NAMES.get(d, d), marker={"color": colour},
                             line={"color": colour}, fillcolor=rgba(colour, 0.15), boxpoints="outliers"))
    fig.update_yaxes(title="delay after onset (s)")
    fig.update_layout(template=LIGHT, height=340, showlegend=False)
    return fig


def stealth_envelope_chart(env: pd.DataFrame) -> go.Figure:
    """Smallest attack each detector catches in 50 % / 90 % of runs (grouped bars)."""
    fig = go.Figure()
    for target, alpha in ((0.5, 0.45), (0.9, 1.0)):
        g = env[env["target_rate"] == target]
        colours = [rgba(P.detector_colour(d), alpha) for d in g["detector"]]
        text = ["not reliably detected" if pd.isna(v) else f"{v:.2f}" for v in g["threshold"]]
        fig.add_trace(go.Bar(x=[DETECTOR_NAMES.get(d, d) for d in g["detector"]], y=g["threshold"].fillna(0),
                             name=f"{target:.0%} detection", text=text, textposition="outside",
                             marker={"color": colours, "line": {"color": P.ink, "width": 1.5}}))
    param = env["param"].iloc[0] if not env.empty else "parameter"
    fig.update_yaxes(title=f"smallest {param}")
    fig.update_layout(template=LIGHT, height=380, barmode="group", bargap=0.3)
    return fig


def calibration_curve(curves: dict[str, dict[str, list[float]]], target: float,
                      chosen: dict[str, float]) -> go.Figure:
    """False-alarm rate vs threshold for detectors A and C, with the target and chosen values."""
    fig = go.Figure()
    for d in ("A", "C"):
        if d not in curves:
            continue
        fig.add_trace(go.Scatter(x=curves[d]["threshold"], y=curves[d]["fa_rate"], name=f"{d} · {DETECTOR_NAMES[d]}",
                                 line={"color": P.detector_colour(d)}))
        if d in chosen:
            fig.add_vline(x=chosen[d], line={"color": P.detector_colour(d), "dash": "dot", "width": 1})
    _threshold(fig, target, f"target {target:.1%}")
    fig.update_yaxes(title="false-alarm rate per run", tickformat=".0%")
    fig.update_xaxes(title="threshold (m)")
    fig.update_layout(template=LIGHT, height=340)
    return fig


def propagation_heatmap(matrix: np.ndarray) -> go.Figure:
    """M[i, j] = mean extra displacement of drone j when drone i is spoofed (forest scale)."""
    names = [drone_label(i) for i in range(len(matrix))]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=names, y=names, colorscale=[[0.0, P.paper], [0.5, P.lavender_2], [1.0, P.forest]],
        colorbar={"title": {"text": "metres", "side": "right"}, "outlinewidth": 1, "outlinecolor": P.ink},
        hovertemplate="spoof %{y} -> %{x} moved %{z:.2f} m<extra></extra>", xgap=1, ygap=1))
    fig.update_yaxes(title="spoofed drone", autorange="reversed", showgrid=False)
    fig.update_xaxes(title="displaced drone", showgrid=False, tickangle=-45)
    fig.update_layout(template=LIGHT, height=440)
    return fig


def pagerank_bar(influence: np.ndarray) -> go.Figure:
    """Influence (PageRank) per drone, highest first."""
    order = np.argsort(influence)[::-1]
    fig = go.Figure(go.Bar(x=[drone_label(int(i)) for i in order], y=influence[order],
                           marker={"color": P.forest, "line": {"color": P.ink, "width": 1.5}},
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title="influence (PageRank)")
    fig.update_layout(template=LIGHT, height=320, bargap=0.35)
    return fig


def min_separation_chart(times: np.ndarray, separation: np.ndarray, limit: float = 1.0) -> go.Figure:
    """Smallest true distance between any two drones over time, with the near-collision limit."""
    fig = go.Figure(go.Scatter(x=times, y=separation, name="min separation", line={"color": P.forest}))
    below = separation < limit
    if below.any():
        fig.add_trace(go.Scatter(x=times[below], y=separation[below], mode="markers", name="below limit",
                                 marker={"color": P.status_alarm, "size": 6}))
    _threshold(fig, limit, f"{limit:g} m near-collision limit")
    fig.update_yaxes(title="metres", rangemode="tozero")
    fig.update_xaxes(title="time (s)")
    fig.update_layout(template=LIGHT, height=300)
    return fig


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------
def recovery_map(obs: ObservedFrame, positions: np.ndarray | None, inlier_mask: np.ndarray | None,
                 outlier_anchors: Sequence[int], repaired: dict[int, np.ndarray] | None = None,
                 truth: np.ndarray | None = None) -> go.Figure:
    """Reported GPS vs recovered positions, anchors (inlier / RANSAC outlier), repairs and (optional) truth."""
    fig = go.Figure()
    gps = obs.gps_pos
    names = [drone_label(i) for i in range(obs.n)]
    if positions is not None:
        lx: list[float | None] = []
        ly: list[float | None] = []
        for i in range(obs.n):
            lx += [gps[i, 0], positions[i, 0], None]
            ly += [gps[i, 1], positions[i, 1], None]
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line={"color": P.muted, "width": 1},
                                 hoverinfo="skip", name="GPS to recovered"))
    fig.add_trace(go.Scatter(x=gps[:, 0], y=gps[:, 1], mode="markers", name="reported GPS", text=names,
                             marker={"symbol": "circle-open", "size": 11, "color": P.ink, "line": {"width": 1.5}},
                             hovertemplate="%{text} GPS<extra></extra>"))
    if positions is not None:
        fig.add_trace(go.Scatter(x=positions[:, 0], y=positions[:, 1], mode="markers", name="recovered",
                                 text=names, hovertemplate="%{text} recovered<extra></extra>",
                                 marker={"size": 10, "color": P.forest, "line": {"color": P.ink, "width": 1}}))
    anchored = np.flatnonzero(obs.anchor_mask)
    if anchored.size:
        inl = [i for i in anchored if i not in set(outlier_anchors)]
        if inl:
            fig.add_trace(go.Scatter(x=obs.anchor_pos[inl, 0], y=obs.anchor_pos[inl, 1], mode="markers",
                                     name="anchor (inlier)", marker={"symbol": "diamond", "size": 12, "color": P.det_c,
                                                                     "line": {"color": P.ink, "width": 1}}))
        if outlier_anchors:
            out = list(outlier_anchors)
            fig.add_trace(go.Scatter(x=obs.anchor_pos[out, 0], y=obs.anchor_pos[out, 1], mode="markers",
                                     name="anchor rejected by RANSAC",
                                     marker={"symbol": "x", "size": 12, "color": P.status_alarm}))
    if repaired:
        ids = sorted(repaired)
        pts = np.array([repaired[i] for i in ids])
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name="multilateration repair",
                                 text=[drone_label(i) for i in ids], hovertemplate="%{text} repaired<extra></extra>",
                                 marker={"symbol": "square", "size": 10, "color": P.lavender_2,
                                         "line": {"color": P.ink, "width": 1.5}}))
    if truth is not None:
        fig.add_trace(go.Scatter(x=truth[:, 0], y=truth[:, 1], mode="markers", name="true position (evaluation)",
                                 marker={"symbol": "cross-thin-open", "size": 10, "color": P.ink_2,
                                         "line": {"width": 1.5}}))
    fig.update_xaxes(title="East (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="North (m)")
    fig.update_layout(template=LIGHT, height=480)
    return fig


def attack_vector_chart(times: np.ndarray, vectors: np.ndarray, onset: float | None = None) -> go.Figure:
    """Estimated swarm-wide offset over time: east and north parts and its length.

    vectors is (T, 2) from RecoveryRun.attack_vector_series(): mean of
    (GPS - recovered position) over all drones; NaN where recovery failed.
    """
    if vectors.shape != (len(times), 2):
        raise ValueError(f"vectors must have shape ({len(times)}, 2), got {vectors.shape}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=np.linalg.norm(vectors, axis=1), name="size",
                             line={"color": P.forest}))
    fig.add_trace(go.Scatter(x=times, y=vectors[:, 0], name="east", line={"color": P.ink_2, "width": 1.5}))
    fig.add_trace(go.Scatter(x=times, y=vectors[:, 1], name="north",
                             line={"color": P.muted, "width": 1.5, "dash": "dot"}))
    _onset(fig, onset)
    fig.update_yaxes(title="metres")
    fig.update_xaxes(title="time (s)")
    fig.update_layout(template=LIGHT, height=300)
    return fig


# --------------------------------------------------------------------------
# Architecture diagram (Graphviz DOT, palette colours)
# --------------------------------------------------------------------------
def architecture_dot() -> str:
    """The processing pipeline as a Graphviz DOT string in palette colours."""
    node = (f'node [shape=box, style="rounded,filled", fillcolor="{P.paper}", color="{P.ink}", '
            f'penwidth=1.5, fontname="Inter", fontsize=11, fontcolor="{P.ink}", margin="0.18,0.08"];')

    def det(name: str, label: str, colour: str) -> str:
        """A detector node outlined in its fixed colour."""
        return f'{name} [label="{label}", color="{colour}", penwidth=2];'

    lines = [
        "digraph G {", "rankdir=LR; bgcolor=\"transparent\"; splines=true; nodesep=0.35; ranksep=0.45;",
        node, f'edge [color="{P.ink_2}", arrowsize=0.7, penwidth=1.2];',
        'sim [label="Swarm simulator\\n(truth + sensors)"];',
        'atk [label="Simulated GPS spoof\\n(software offsets)"];',
        f'obs [label="ObservedFrame\\n(no truth fields)", fillcolor="{P.lavender}"];',
        det("A", "A  relative geometry", P.det_a), det("B", "B  motion consistency", P.det_b),
        det("C", "C  absolute anchors", P.det_c),
        'rec [label="Recovery\\nMDS + RANSAC"];',
        f'fus [label="Dempster-Shafer\\nfusion", fillcolor="{P.lavender}"];',
        'inc [label="Incident +\\nexplanation"];',
        f'pkg [label="Signed evidence\\npackage", color="{P.forest}", penwidth=2];',
        "sim -> obs; atk -> obs; obs -> A; obs -> B; obs -> C; obs -> rec;",
        "A -> fus; B -> fus; C -> fus; rec -> fus; fus -> inc; inc -> pkg;",
        "}",
    ]
    return "\n".join(lines)
