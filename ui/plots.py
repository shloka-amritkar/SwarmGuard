"""Plotly figures and templates for the app (docs/phases/phase-7.md section B10).

TWO TEMPLATES
    groundtruth_light   charts on cards: transparent background, IBM Plex
                        Sans, 1px espresso axes, faint dashed gridlines in
                        "rule", stone ticks, 2.25px lines, horizontal legend
                        above, no titles, 300 ms transitions.
    groundtruth_panel   charts inside charcoal instrument panels: charcoal
                        background, char_line gridlines, char_text labels,
                        amber as the neutral trace.
Every colour comes from core/palette.py. Detectors always use their fixed
colours (A steel, B teal, C ochre, D terracotta); brick red only for alarms.

THE SWARM MAP
Always the dark "carto-darkmatter" basemap over Pune, inside an instrument
panel. Positions are converted from metres with enu_to_latlon (rule R3,
display only). Animated maps use one frame every 5 steps.
WHY every 5 steps: 600 frames x 20 drones makes the browser lag; 120
frames at 0.5 s spacing still look smooth.
Dragging pans the map (dragmode "pan") and the wheel zooms. The camera
either follows the swarm's GPS centroid or shows the whole flight.
Map markers cannot have outlines or custom shapes, so outlines are drawn as
a slightly larger marker underneath, and anchored drones get an amber
diamond drawn as a small line. The "offline" grid version (no internet
tiles) can use real outlines, diamond markers and dashed truth lines.

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
LIGHT = "groundtruth_light"
PANEL = "groundtruth_panel"
CONSOLE = PANEL     # WHY kept: older code and tests may still say CONSOLE
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
    colorway = [neutral, P.det_a, P.det_b, P.det_c, P.det_d, P.olive_2, P.stone_text]
    layout = go.Layout(
        paper_bgcolor=bg, plot_bgcolor=bg, colorway=colorway,
        font={"family": FONT_SANS, "color": text, "size": 13},
        xaxis=_axis(line, grid, tick, dashed), yaxis=_axis(line, grid, tick, dashed),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
                "bgcolor": TRANSPARENT, "font": {"color": text, "size": 12}},
        margin={"l": 16, "r": 20, "t": 40, "b": 16},
        hoverlabel={"bgcolor": P.card, "bordercolor": P.espresso,
                    "font": {"family": FONT_SANS, "color": P.espresso, "size": 12}},
        transition={"duration": 300, "easing": "cubic-in-out"},
        title={"text": ""},
    )
    data = {"scatter": [go.Scatter(line={"width": LINE_W})]}
    return go.layout.Template(layout=layout, data=data)


def register_templates() -> None:
    """Register both templates with Plotly (safe to call many times)."""
    pio.templates[LIGHT] = _template(TRANSPARENT, P.espresso, P.espresso, P.rule, P.stone_text, P.olive, True)
    pio.templates[PANEL] = _template(P.charcoal, P.char_text, P.char_line, P.char_line,
                                       P.char_mute, P.amber, False)


register_templates()


def _threshold(fig: go.Figure, y: float, label: str, row: int | None = None, col: int | None = None) -> None:
    """Dashed espresso threshold line with a right-aligned label."""
    kwargs: dict[str, Any] = {"row": row, "col": col} if row is not None else {}
    fig.add_hline(y=y, line={"color": P.espresso, "dash": "dash", "width": 1},
                  annotation_text=label, annotation_position="top right",
                  annotation_font={"size": 11, "color": P.ink_2}, **kwargs)


def _onset(fig: go.Figure, onset: float | None, label: str = "simulation onset") -> None:
    """Vertical dotted line at the attack onset (EVALUATION: from the simulation truth)."""
    if onset is None:
        return
    fig.add_vline(x=onset, line={"color": P.stone_text, "dash": "dot", "width": 1.2},
                  annotation_text=label, annotation_position="top left",
                  annotation_font={"size": 11, "color": P.stone_text})


def _heat_scale(brass_at: float) -> list[list[Any]]:
    """Heatmap colour scale card -> vellum -> brass -> brick (B10).

    brass_at is where (0..1) the scale reaches brass; vellum sits halfway
    before it. WHY no rainbow: one warm ramp reads as "more" without
    suggesting categories that do not exist.
    """
    b = float(np.clip(brass_at, 0.02, 0.99))
    return [[0.0, P.card], [b / 2.0, P.vellum], [b, P.brass], [1.0, P.status_alarm]]


def status_colour(verdict: str) -> str:
    """Fill colour for a fused verdict / status string."""
    return P.status_colour(verdict)


# --------------------------------------------------------------------------
# Swarm map
# --------------------------------------------------------------------------
def _zoom_for(extent_m: float, lat: float, pixels: float = 440.0, fill: float = 1 / 1.3,
              min_extent: float = 30.0) -> float:
    """Map zoom so that a distance of extent_m covers `fill` of `pixels` screen pixels.

        metres per pixel at zoom z = 78271.52 * cos(lat) / 2^z
        we want   extent_m = fill * pixels * mpp
        so        z = log2(78271.52 * cos(lat) * fill * pixels / extent_m)

    WHY 78271.52 (not the familiar 156543.03): Plotly's map uses MapLibre,
    whose tiles are 512 px wide, so zoom 0 shows the equator at half the
    metres per pixel of the classic 256 px tiles.
    The default fill 1/1.3 leaves a 30 % margin. Zoom is clipped to 10..21:
    Carto tiles stop at about zoom 20; MapLibre enlarges them beyond that.
    """
    if pixels <= 0:
        raise ValueError(f"pixels must be > 0, got {pixels}")
    if not 0.0 < fill <= 1.0:
        raise ValueError(f"fill must be in (0, 1], got {fill}")
    mpp = max(extent_m, min_extent) / (fill * pixels)
    return float(np.clip(math.log2(78271.52 * math.cos(math.radians(lat)) / mpp), 10.0, 21.0))


def metres_per_pixel(zoom: float, lat: float = LAT0) -> float:
    """Inverse of _zoom_for: ground metres covered by one screen pixel at this zoom."""
    return 78271.52 * math.cos(math.radians(lat)) / 2.0 ** zoom


def camera_whole(points: np.ndarray, pixels: float, padding: float = 0.15) -> tuple[np.ndarray, float]:
    """"Whole flight" camera: (centre in metres, zoom) showing every point of the run.

        centre = middle of the bounding box  (min + max) / 2
        extent = longest side of the bounding box
        zoom   chosen so extent * (1 + padding) fits in `pixels`

    WHY the bounding-box middle (not the mean): the camera must show the
    whole flight, and a mean is pulled towards where the swarm lingered.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    extent = float(np.ptp(pts, axis=0).max())
    return centre, _zoom_for(extent, LAT0, pixels=pixels, fill=1.0 / (1.0 + padding))


def camera_follow(gps: np.ndarray, pixels: float, fill: float) -> tuple[np.ndarray, float]:
    """"Follow swarm" camera: (per-step centres (T, 2) in metres, one fixed zoom).

    gps  (T, N, 2) GPS-REPORTED positions (never the truth: the camera shows
         what the operator would see, rule R1).
    centre_k = mean_i gps[k, i]                  (swarm centroid at step k)
    r_k      = max_i |gps[k, i] - centre_k|      (radius of the swarm at step k)
    Two zooms are computed and the smaller (wider view) is used:
      * comfort: the typical diameter 2 * median(r_k) fills `fill` of the view
        (about half for a 10-drone swarm, two thirds for the light show);
      * safety:  the largest radius ever, 2 * max(r_k), still fits with a
        15 % margin, so no drone ever leaves the view, even a spoofed one
        that the GPS reports 10 m away.
    WHY one fixed zoom: a zoom that changed every frame would make the map
    "breathe" and distract from the drones' own motion.
    """
    gps = np.asarray(gps, dtype=float)
    centres = gps.mean(axis=1)
    radius = np.linalg.norm(gps - centres[:, None, :], axis=-1).max(axis=1)
    comfort = _zoom_for(2.0 * float(np.median(radius)), LAT0, pixels=pixels, fill=fill, min_extent=4.0)
    safety = _zoom_for(2.0 * float(radius.max()), LAT0, pixels=pixels, fill=1.0 / 1.15, min_extent=4.0)
    return centres, min(comfort, safety)


def _ring_points(centre: np.ndarray, radius: float, n: int = 48) -> tuple[list[float | None], list[float | None]]:
    """A closed circle of `radius` metres around centre, as a polyline ending in None."""
    angles = np.linspace(0.0, 2.0 * math.pi, n + 1)
    xs: list[float | None] = list(centre[0] + radius * np.cos(angles))
    ys: list[float | None] = list(centre[1] + radius * np.sin(angles))
    return xs + [None], ys + [None]


def _diamond_points(pos: np.ndarray, half: float) -> tuple[list[float | None], list[float | None]]:
    """Diamond outlines (half-diagonal `half` metres) around each position, None-separated.

    WHY drawn as lines: map markers cannot be diamonds, so the survey symbol
    for an anchored drone is drawn as a small closed polyline.
    """
    xs: list[float | None] = []
    ys: list[float | None] = []
    for x, y in pos:
        xs += [x, x + half, x, x - half, x, None]
        ys += [y + half, y, y - half, y, y + half, None]
    return xs, ys


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


RING_RADII_M = (10.0, 20.0)    # faint range rings around the swarm centroid


def _frame_layers(result: SimResult, statuses: np.ndarray, k: int, links: bool, evaluation: bool,
                  light_show: bool, trail: int, diamond_m: float = 1.2) -> list[dict[str, Any]]:
    """Plain-data layers for step k: dicts with x, y (metres), colour, size, kind.

    The same layers feed the map (lat/lon) and the offline grid figure.
    diamond_m is the half-diagonal of the anchor diamond in metres (chosen
    by the caller from the zoom so the diamond is about 9 px on screen).
    """
    frame = result.frames[k]
    gps = frame.observed.gps_pos
    n = result.n_drones
    colours = [status_colour(s) for s in statuses[k]]
    labels = [f"{drone_label(i)} · {str(statuses[k][i]).replace('_', ' ').upper()}" for i in range(n)]
    layers: list[dict[str, Any]] = []
    rx: list[float | None] = []
    ry: list[float | None] = []
    for radius in RING_RADII_M:
        xs, ys = _ring_points(gps.mean(axis=0), radius)
        rx += xs
        ry += ys
    layers.append({"kind": "rings", "x": rx, "y": ry})
    lx, ly = _nearest_links(gps) if links else ([], [])
    layers.append({"kind": "links", "x": lx, "y": ly})
    lo = max(0, k - trail)
    hist = np.stack([f.observed.gps_pos for f in result.frames[lo:k + 1]])        # (L, N, 2)
    ages = np.repeat(np.arange(hist.shape[0])[:, None], n, axis=1).reshape(-1)
    alpha = 0.08 + 0.55 * (ages + 1) / hist.shape[0]
    layers.append({"kind": "trails", "x": hist[:, :, 0].reshape(-1), "y": hist[:, :, 1].reshape(-1),
                   "colour": [rgba(P.amber, float(a)) for a in alpha]})
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
    ax_, ay_ = _diamond_points(gps[anchored], diamond_m)
    layers.append({"kind": "anchors", "x": ax_, "y": ay_})
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
        if kind == "rings":
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="lines", line={"color": P.char_line, "width": 1},
                                        hoverinfo="skip", name="range rings"))
        elif kind in ("links", "truth_links"):
            colour = rgba(P.amber, 0.15) if kind == "links" else rgba(P.char_text, 0.45)
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
                                        marker={"size": 13, "color": P.char_text}))
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", name="truth", text=layer["text"],
                                        hovertemplate="%{text}<extra></extra>",
                                        marker={"size": 9, "color": P.charcoal}))
        elif kind == "anchors":
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="lines", hoverinfo="skip", name="anchor diamond",
                                        line={"color": P.amber, "width": 1.5}))
        elif kind == "drones":
            # WHY two markers: map markers have no outline, so a 1 px card-colour
            # ring is a marker 2 px larger drawn underneath.
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", hoverinfo="skip", name="outline",
                                        marker={"size": 13, "color": P.card}))
            traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers", name="drones", text=layer["text"],
                                        hovertemplate="%{text}<extra></extra>",
                                        marker={"size": 11, "color": layer["colour"]}))
    return traces


def _grid_traces(layers: list[dict[str, Any]]) -> list[go.Scatter]:
    """Offline version on a metric grid: real outlines, diamonds for anchors, dashed truth lines."""
    traces: list[go.Scatter] = []
    for layer in layers:
        kind, x, y = layer["kind"], list(layer["x"]), list(layer["y"])
        if kind == "rings":
            traces.append(go.Scatter(x=x, y=y, mode="lines", hoverinfo="skip", name="range rings",
                                     line={"color": P.char_line, "width": 1}))
        elif kind in ("links", "truth_links"):
            traces.append(go.Scatter(x=x, y=y, mode="lines", hoverinfo="skip", name=kind,
                                     line={"color": rgba(P.amber, 0.15) if kind == "links" else rgba(P.char_text, 0.5),
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
                                     marker={"size": 11, "symbol": "circle-open", "color": P.char_text,
                                             "line": {"width": 1.5}}))
        elif kind == "drones":
            symbols = ["diamond" if a else "circle" for a in layer["anchored"]]
            traces.append(go.Scatter(x=x, y=y, mode="markers", name="drones", text=layer["text"],
                                     hovertemplate="%{text}<extra></extra>",
                                     marker={"size": 13, "color": layer["colour"], "symbol": symbols,
                                             "line": {"color": P.card, "width": 1}}))
    return traces


def _animation_controls(labels: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Play/pause buttons and a styled time slider for the instrument panel."""
    frame_args = {"frame": {"duration": 160, "redraw": True}, "mode": "immediate",
                  "transition": {"duration": 0}, "fromcurrent": True}
    buttons = [{
        "type": "buttons", "direction": "left", "x": 0.0, "y": -0.04, "xanchor": "left", "yanchor": "top",
        "pad": {"r": 8, "t": 8}, "showactive": False, "bgcolor": P.charcoal_2, "bordercolor": P.char_line,
        "font": {"color": P.char_text, "family": FONT_SANS, "size": 12},
        "buttons": [{"label": "Play", "method": "animate", "args": [None, frame_args]},
                    {"label": "Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}],
    }]
    slider = [{
        "active": 0, "x": 0.16, "y": -0.02, "len": 0.84, "xanchor": "left", "yanchor": "top",
        "pad": {"t": 4, "b": 4}, "bgcolor": P.charcoal_2, "activebgcolor": P.amber, "bordercolor": P.char_line,
        "tickcolor": P.char_line, "font": {"color": P.char_mute, "family": FONT_MONO, "size": 10},
        "currentvalue": {"prefix": "t = ", "suffix": " s", "font": {"color": P.amber, "family": FONT_MONO, "size": 12}},
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
    camera: str | None = None,
) -> go.Figure:
    """Swarm map in instrument-panel style.

    statuses  (T, N) status/verdict strings that colour the drones
    step      draw only this step (no animation); None = animate every `every` steps
    evaluation  also draw true positions (EVALUATION VIEW, rule R1)
    offline   plain metric grid instead of map tiles
    light_show larger low-opacity layer under each drone (night lights)
    camera    "follow" (centre on the swarm's GPS centroid every frame, at a
              fixed close zoom) or "whole" (one camera showing the entire
              flight). None = "follow" when animated, "whole" for a single
              step (docs/phases/phase-7.md A2, A3).
    Dragging pans the map (dragmode "pan"); the mouse wheel zooms when the
    figure is shown with MAP_PLOTLY_CONFIG (ui/components.chart does this).
    """
    if statuses.shape != (result.n_steps, result.n_drones):
        raise ValueError(f"statuses must have shape {(result.n_steps, result.n_drones)}, got {statuses.shape}")
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    camera = camera or ("whole" if step is not None else "follow")
    if camera not in CAMERA_MODES:
        raise ValueError(f"camera must be one of {CAMERA_MODES}, got {camera!r}")
    steps = [step] if step is not None else list(range(0, result.n_steps, every))
    if step is None and steps[-1] != result.n_steps - 1:
        steps.append(result.n_steps - 1)
    centres, zoom = _camera(result, camera, evaluation, light_show, height)
    # WHY 9 px: the anchor diamond should look the same size at any zoom.
    diamond_m = 9.0 * metres_per_pixel(zoom)
    make = _grid_traces if offline else _map_traces

    def layers(k: int) -> list[Any]:
        return make(_frame_layers(result, statuses, k, links, evaluation, light_show, trail, diamond_m))

    def centre_of(k: int) -> dict[str, float]:
        lat, lon = enu_to_latlon(float(centres[k, 0]), float(centres[k, 1]))
        return {"lat": float(lat), "lon": float(lon)}

    fig = go.Figure(data=layers(steps[0]))
    if len(steps) > 1:
        labels = [f"{result.times()[k]:.1f}" for k in steps]
        follow = camera == "follow" and not offline
        # WHY layout inside each frame (follow mode only): Plotly applies a
        # frame's layout while animating, so the camera moves with the swarm.
        fig.frames = [go.Frame(name=lab, data=layers(k),
                               layout={"map": {"center": centre_of(k)}} if follow else None)
                      for k, lab in zip(steps, labels)]
        buttons, slider = _animation_controls(labels)
        fig.update_layout(updatemenus=buttons, sliders=slider)
    # WHY explicit colours: the map always sits in an instrument panel, so its
    # paper (behind the play controls) must match the panel, not the page.
    fig.update_layout(template=PANEL, height=height, showlegend=False, dragmode="pan",
                      paper_bgcolor=P.charcoal, plot_bgcolor=P.charcoal,
                      margin={"l": 0, "r": 0, "t": 0, "b": 64 if len(steps) > 1 else 0})
    if offline:
        fig.update_xaxes(title="East (m)", scaleanchor="y", scaleratio=1)
        fig.update_yaxes(title="North (m)")
    else:
        fig.update_layout(map={"style": MAP_STYLE, "center": centre_of(steps[0]), "zoom": zoom})
    return fig


CAMERA_MODES = ("follow", "whole")
FOLLOW_FILL = 0.5           # a normal swarm fills about half of the view
FOLLOW_FILL_SHOW = 0.67     # the 20-drone light show about two thirds
MAX_VIEW_PX = 340           # WHY: the map's shorter side; a phone panel is about 340 px wide


def _camera(result: SimResult, camera: str, evaluation: bool, light_show: bool,
            height: int) -> tuple[np.ndarray, float]:
    """Per-step camera centres (T, 2) in metres and one zoom for the chosen mode.

    The view's shorter side is taken as min(height - 80, MAX_VIEW_PX) pixels
    (80 px are the play controls), so the swarm also fits on a narrow phone.
    In "whole" mode every step gets the same centre.
    """
    pixels = float(min(max(height - 80, 200), MAX_VIEW_PX))
    gps = result.stack("gps_pos")
    if camera == "follow":
        return camera_follow(gps, pixels, FOLLOW_FILL_SHOW if light_show else FOLLOW_FILL)
    points = gps.reshape(-1, 2)
    if evaluation:  # evaluation view also draws the truth, so keep it in frame
        points = np.vstack([points, result.stack("true_pos").reshape(-1, 2)])
    centre, zoom = camera_whole(points, pixels)
    return np.repeat(centre[None, :], result.n_steps, axis=0), zoom


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
                           font={"color": P.stone_text, "size": 12})
    _onset(fig, onset)
    fig.update_yaxes(title_text="A vote share", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="B NIS", type="log", row=2, col=1)
    fig.update_yaxes(title_text="B CUSUM", row=3, col=1)
    fig.update_yaxes(title_text="C residual (m)", row=4, col=1)
    fig.update_xaxes(title_text="time (s)", row=4, col=1)
    fig.update_layout(template=LIGHT, height=720, showlegend=False)
    return fig


def residual_heatmap(obs: ObservedFrame, tau_a: float) -> go.Figure:
    """Detector A's pair residuals |GPS distance - radio range| at one step.

    Colour scale card -> vellum -> brass -> brick, with brass at the
    threshold tau_a, so anything brick-coloured is above the alarm level.
    """
    residual, _ = geometry_residuals(obs)
    names = [drone_label(i) for i in range(obs.n)]
    zmax = max(2.0 * tau_a, float(np.nanmax(residual)) if np.isfinite(residual).any() else 0.0)
    fig = go.Figure(go.Heatmap(
        z=residual, x=names, y=names, zmin=0, zmax=zmax,
        colorscale=_heat_scale(min(tau_a / zmax, 0.99)),
        colorbar={"title": {"text": "residual (m)", "side": "right"}, "outlinewidth": 1, "outlinecolor": P.espresso},
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
    fig.add_trace(go.Scatter(x=t, y=pl, name="plausibility", line={"color": P.olive, "width": 1.5}))
    fig.add_trace(go.Scatter(x=t, y=bel, name="belief", fill="tonexty", fillcolor=rgba(P.olive, 0.10),
                             line={"color": P.espresso}))
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
                           showarrow=False, font={"color": P.stone_text})
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
               "color": P.espresso, "thickness": 1.2, "width": 6}
    colours = [P.detector_colour(d) for d in summary["detector"]]
    fig = go.Figure(go.Bar(x=[DETECTOR_NAMES.get(d, d) for d in summary["detector"]], y=y, error_y=err,
                           marker={"color": colours, "line": {"color": P.espresso, "width": 1.5}},
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
                             marker={"color": colours, "line": {"color": P.espresso, "width": 1.5}}))
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
    """M[i, j] = mean extra displacement of drone j when drone i is spoofed (card -> brick scale)."""
    names = [drone_label(i) for i in range(len(matrix))]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=names, y=names, colorscale=_heat_scale(0.66),
        colorbar={"title": {"text": "metres", "side": "right"}, "outlinewidth": 1, "outlinecolor": P.espresso},
        hovertemplate="spoof %{y} -> %{x} moved %{z:.2f} m<extra></extra>", xgap=1, ygap=1))
    fig.update_yaxes(title="spoofed drone", autorange="reversed", showgrid=False)
    fig.update_xaxes(title="displaced drone", showgrid=False, tickangle=-45)
    fig.update_layout(template=LIGHT, height=440)
    return fig


def pagerank_bar(influence: np.ndarray) -> go.Figure:
    """Influence (PageRank) per drone, highest first."""
    order = np.argsort(influence)[::-1]
    fig = go.Figure(go.Bar(x=[drone_label(int(i)) for i in order], y=influence[order],
                           marker={"color": P.olive, "line": {"color": P.espresso, "width": 1.5}},
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title="influence (PageRank)")
    fig.update_layout(template=LIGHT, height=320, bargap=0.35)
    return fig


def feature_importance_bar(importances: dict[str, float]) -> go.Figure:
    """Experimental Detector D: Random Forest feature importance, largest at the top.

    Importance = the average drop in Gini impurity that splits on that
    feature produce, over all trees (scikit-learn's feature_importances_).
    """
    items = sorted(importances.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker={"color": P.det_d, "line": {"color": P.espresso, "width": 1.5}},
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="importance (mean impurity decrease)", rangemode="tozero")
    fig.update_yaxes(showgrid=False, tickfont={"family": FONT_MONO, "size": 12})
    fig.update_layout(template=LIGHT, height=40 + 30 * len(items), bargap=0.3)
    return fig


def confusion_chart(confusion: Sequence[Sequence[int]]) -> go.Figure:
    """2 x 2 confusion matrix (rows = truth, columns = prediction) as an outlined grid."""
    z = np.asarray(confusion, dtype=float)
    labels = ["no attack", "attacked"]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"predicted {s}" for s in labels], y=[f"truly {s}" for s in labels],
        colorscale=[[0.0, P.card], [1.0, P.brass_2]], showscale=False, xgap=2, ygap=2,
        text=[[f"{int(v):,}" for v in row] for row in z], texttemplate="%{text}",
        textfont={"family": FONT_MONO, "size": 15, "color": P.espresso},
        hovertemplate="%{y}, %{x}: %{z:,.0f} rows<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(template=LIGHT, height=260)
    return fig


def ml_probability_chart(run: DetectionRun, drone: int, onset: float | None = None,
                         threshold: float = 0.5) -> go.Figure:
    """Experimental Detector D: P(attacked) for one drone over time, alarms marked."""
    t, p = run.times(), run.scores()[:, drone]
    flags = run.flags()[:, drone]
    fig = go.Figure(go.Scatter(x=t, y=p, name="P(attacked)", line={"color": P.det_d}))
    if flags.any():
        fig.add_trace(go.Scatter(x=t[flags], y=p[flags], mode="markers", name="alarm (after persistence)",
                                 marker={"color": P.status_alarm, "size": 5}))
    _threshold(fig, threshold, f"threshold {threshold:g}")
    _onset(fig, onset)
    fig.update_yaxes(range=[0, 1.02], title="probability")
    fig.update_xaxes(title="time (s)")
    fig.update_layout(template=LIGHT, height=300)
    return fig


def min_separation_chart(times: np.ndarray, separation: np.ndarray, limit: float = 1.0) -> go.Figure:
    """Smallest true distance between any two drones over time, with the near-collision limit."""
    fig = go.Figure(go.Scatter(x=times, y=separation, name="min separation", line={"color": P.olive}))
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
    """Reported GPS vs recovered positions, anchors (inlier / RANSAC outlier), repairs and (optional) truth.

    Drawn in the instrument-panel template (it sits in a charcoal panel);
    dragging pans, like the swarm map.
    """
    fig = go.Figure()
    gps = obs.gps_pos
    names = [drone_label(i) for i in range(obs.n)]
    if positions is not None:
        lx: list[float | None] = []
        ly: list[float | None] = []
        for i in range(obs.n):
            lx += [gps[i, 0], positions[i, 0], None]
            ly += [gps[i, 1], positions[i, 1], None]
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line={"color": P.char_mute, "width": 1},
                                 hoverinfo="skip", name="GPS to recovered"))
    fig.add_trace(go.Scatter(x=gps[:, 0], y=gps[:, 1], mode="markers", name="reported GPS", text=names,
                             marker={"symbol": "circle-open", "size": 11, "color": P.char_text, "line": {"width": 1.5}},
                             hovertemplate="%{text} GPS<extra></extra>"))
    if positions is not None:
        fig.add_trace(go.Scatter(x=positions[:, 0], y=positions[:, 1], mode="markers", name="recovered",
                                 text=names, hovertemplate="%{text} recovered<extra></extra>",
                                 marker={"size": 10, "color": P.amber, "line": {"color": P.charcoal, "width": 1}}))
    anchored = np.flatnonzero(obs.anchor_mask)
    if anchored.size:
        inl = [i for i in anchored if i not in set(outlier_anchors)]
        if inl:
            fig.add_trace(go.Scatter(x=obs.anchor_pos[inl, 0], y=obs.anchor_pos[inl, 1], mode="markers",
                                     name="anchor (inlier)", marker={"symbol": "diamond", "size": 12, "color": P.det_c,
                                                                     "line": {"color": P.card, "width": 1}}))
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
                                 marker={"symbol": "square", "size": 10, "color": P.brass,
                                         "line": {"color": P.card, "width": 1}}))
    if truth is not None:
        fig.add_trace(go.Scatter(x=truth[:, 0], y=truth[:, 1], mode="markers", name="true position (evaluation)",
                                 marker={"symbol": "cross-thin-open", "size": 10, "color": P.char_mute,
                                         "line": {"width": 1.5}}))
    fig.update_xaxes(title="East (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="North (m)")
    fig.update_layout(template=PANEL, height=480, dragmode="pan",
                      margin={"l": 16, "r": 20, "t": 48, "b": 16})
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
                             line={"color": P.olive}))
    fig.add_trace(go.Scatter(x=times, y=vectors[:, 0], name="east", line={"color": P.ink_2, "width": 1.5}))
    fig.add_trace(go.Scatter(x=times, y=vectors[:, 1], name="north",
                             line={"color": P.stone_text, "width": 1.5, "dash": "dot"}))
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
    node = (f'node [shape=box, style="rounded,filled", fillcolor="{P.card}", color="{P.espresso}", '
            f'penwidth=1.5, fontname="IBM Plex Sans", fontsize=11, fontcolor="{P.espresso}", margin="0.18,0.08"];')

    def det(name: str, label: str, colour: str) -> str:
        """A detector node outlined in its fixed colour."""
        return f'{name} [label="{label}", color="{colour}", penwidth=2];'

    lines = [
        "digraph G {", "rankdir=LR; bgcolor=\"transparent\"; splines=true; nodesep=0.35; ranksep=0.45;",
        node, f'edge [color="{P.ink_2}", arrowsize=0.7, penwidth=1.2];',
        'sim [label="Swarm simulator\\n(truth + sensors)"];',
        'atk [label="Simulated GPS spoof\\n(software offsets)"];',
        f'obs [label="ObservedFrame\\n(no truth fields)", fillcolor="{P.sage}"];',
        det("A", "A  relative geometry", P.det_a), det("B", "B  motion consistency", P.det_b),
        det("C", "C  absolute anchors", P.det_c),
        'rec [label="Recovery\\nMDS + RANSAC"];',
        f'fus [label="Dempster-Shafer\\nfusion", fillcolor="{P.vellum}"];',
        'inc [label="Incident +\\nexplanation"];',
        f'pkg [label="Signed evidence\\npackage", color="{P.olive}", penwidth=2];',
        "sim -> obs; atk -> obs; obs -> A; obs -> B; obs -> C; obs -> rec;",
        "A -> fus; B -> fus; C -> fus; rec -> fus; fus -> inc; inc -> pkg;",
        "}",
    ]
    return "\n".join(lines)
