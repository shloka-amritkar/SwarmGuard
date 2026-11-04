"""Formation shapes, shape morphing and legitimate manoeuvres.

A formation is an (N, 2) array of OFFSETS in metres, measured from the
swarm centre. Drone i's intended position ("slot") is centre + offsets[i].
Every built-in shape is centred on (0, 0) and scaled so that its farthest
point is exactly `r` metres from the centre.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import ManoeuvreEvent
from core.utils import pairwise_distances, rotation_matrix

logger = logging.getLogger(__name__)

# Time (s) taken to blend from one shape to the next during a shape change.
MORPH_BLEND_S: float = 2.0
# Number of points used to draw the heart curve before resampling it.
_DENSE_CURVE_POINTS = 2000


def _normalise(points: np.ndarray, r: float) -> np.ndarray:
    """Centre points on their mean and scale so the farthest one is r away.

    offsets = (points - mean) * r / max |points - mean|
    """
    centred = points - points.mean(axis=0)
    max_norm = float(np.max(np.linalg.norm(centred, axis=1)))
    if max_norm < 1e-12:
        return centred
    return centred * (r / max_norm)


def resample_closed_curve(vertices: np.ndarray, n: int) -> np.ndarray:
    """Place n points at equal arc-length spacing along a closed polyline.

    Steps:
      1. Append the first vertex at the end so the curve is closed.
      2. Cumulative length s_k = sum of segment lengths up to vertex k.
      3. Target lengths s = L * j / n for j = 0..n-1 (L = total length).
      4. Linear interpolation of x(s) and y(s) with np.interp.

    WHY equal arc length: sampling the heart curve at equal steps of its
    parameter t would bunch drones together where the curve moves slowly
    (near the top dip) and leave gaps elsewhere. Equal spacing along the
    curve gives an evenly spread, recognisable shape.
    """
    closed = np.vstack([vertices, vertices[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = cum[-1] * np.arange(n) / n
    x = np.interp(targets, cum, closed[:, 0])
    y = np.interp(targets, cum, closed[:, 1])
    return np.column_stack([x, y])


def circle(n: int, r: float) -> np.ndarray:
    """Drones evenly spaced on a circle: angle_k = 2 pi k / n, k = 0..n-1."""
    angles = 2.0 * np.pi * np.arange(n) / n
    return r * np.column_stack([np.cos(angles), np.sin(angles)])


def line(n: int, r: float) -> np.ndarray:
    """Drones evenly spaced on an east-west line from x = -r to x = +r."""
    x = np.linspace(-r, r, n)
    return np.column_stack([x, np.zeros(n)])


def grid(n: int, r: float) -> np.ndarray:
    """Drones on a near-square grid: cols = ceil(sqrt(n)), filled row by row.

    The last row may be partly filled; the result is re-centred on its mean.
    """
    cols = math.ceil(math.sqrt(n))
    k = np.arange(n)
    points = np.column_stack([k % cols, -(k // cols)]).astype(float)
    return _normalise(points, r)


def star(n: int, r: float) -> np.ndarray:
    """Drones evenly spaced along the outline of a 5-pointed star.

    The outline is a 10-vertex polygon alternating outer radius 1 and inner
    radius 0.4, starting with a point straight up (north):
        angle_k = pi/2 + k * pi / 5,  radius_k = 1 if k even else 0.4
    """
    k = np.arange(10)
    angles = np.pi / 2 + k * np.pi / 5
    radii = np.where(k % 2 == 0, 1.0, 0.4)
    vertices = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalise(resample_closed_curve(vertices, n), r)


def heart(n: int, r: float) -> np.ndarray:
    """Drones evenly spaced along the classic heart curve.

    x(t) = 16 sin^3 t
    y(t) = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t,   t in [0, 2 pi)
    """
    t = np.linspace(0.0, 2.0 * np.pi, _DENSE_CURVE_POINTS, endpoint=False)
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    vertices = np.column_stack([x, y])
    return _normalise(resample_closed_curve(vertices, n), r)


def custom(points: Sequence[tuple[float, float]]) -> np.ndarray:
    """User-defined offsets in metres, used exactly as given (no scaling)."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"custom points must have shape (N, 2), got {arr.shape}")
    return arr.copy()


def get_shape(
    name: str,
    n: int,
    r: float,
    custom_points: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return the (n, 2) offsets for a named formation."""
    builders = {"circle": circle, "line": line, "grid": grid, "star": star, "heart": heart}
    if name == "custom":
        if custom_points is None or len(custom_points) != n:
            raise ValueError(f"shape 'custom' needs exactly {n} custom points")
        return custom(custom_points)
    if name not in builders:
        raise ValueError(f"unknown formation {name!r}; choose from {sorted(builders) + ['custom']}")
    return builders[name](n, r)


def assign_slots(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reorder target slots so total travel from current positions is minimal.

    Cost matrix:  C[i, j] = |current_i - target_j|
    Find the one-to-one assignment j = a(i) minimising  sum_i C[i, a(i)].
    scipy.optimize.linear_sum_assignment solves this exactly (the
    "Hungarian algorithm"). Returns target rows in the new order, so row i
    is drone i's new slot.

    WHY: without it, drone 0 of the circle goes to point 0 of the star even
    if that point is on the far side; drones cross paths during light-show
    shape changes and near-collisions appear that no attack caused.
    """
    if current.shape != target.shape:
        raise ValueError(f"shape mismatch: {current.shape} vs {target.shape}")
    diff = current[:, None, :] - target[None, :, :]
    cost = np.linalg.norm(diff, axis=-1)
    _, cols = linear_sum_assignment(cost)
    return target[cols]


@dataclass(frozen=True)
class _ShapeChange:
    """One shape change: blend from `start_offsets` to `end_offsets` from time t0."""

    t0: float
    start_offsets: np.ndarray
    end_offsets: np.ndarray


class ShapeSchedule:
    """Pre-computed sequence of shape changes for a whole flight.

    offsets(t) during a change that starts at t0:
        alpha = clip((t - t0) / MORPH_BLEND_S, 0, 1)
        offsets = (1 - alpha) * start_offsets + alpha * end_offsets

    WHY pre-compute: the slot assignment for each change is solved once,
    not 600 times, and each change starts from wherever the previous blend
    actually was (so even closely spaced changes never jump).
    """

    def __init__(
        self,
        base_shape: str,
        n: int,
        r: float,
        shape_sequence: Sequence[tuple[float, str]] = (),
        custom_points: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        sequence = list(shape_sequence)
        initial = base_shape
        if sequence and sequence[0][0] <= 0.0:
            # WHY: a shape listed at t = 0 is the starting shape, not a change.
            initial = sequence.pop(0)[1]
        self.initial = get_shape(initial, n, r, custom_points)
        self.changes: list[_ShapeChange] = []
        for t0, name in sequence:
            start = self.offsets(t0)
            end = assign_slots(start, get_shape(name, n, r, custom_points))
            self.changes.append(_ShapeChange(float(t0), start, end))

    def offsets(self, t: float) -> np.ndarray:
        """Return the (N, 2) formation offsets at time t."""
        active = [c for c in self.changes if c.t0 <= t]
        if not active:
            return self.initial.copy()
        change = active[-1]
        alpha = min(max((t - change.t0) / MORPH_BLEND_S, 0.0), 1.0)
        return (1.0 - alpha) * change.start_offsets + alpha * change.end_offsets


def morph_offsets(
    t: float,
    shape_sequence: Sequence[tuple[float, str]],
    n: int,
    r: float,
    base_shape: str = "circle",
    custom_points: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Formation offsets at time t for a list of (time, shape) changes.

    Convenience wrapper around ShapeSchedule for a single time. The
    simulator builds one ShapeSchedule and reuses it for every step.
    """
    return ShapeSchedule(base_shape, n, r, shape_sequence, custom_points).offsets(t)


def _progress(t: float, start: float, end: float) -> float:
    """Linear progress through a window: clip((t - start) / (end - start), 0, 1)."""
    return min(max((t - start) / (end - start), 0.0), 1.0)


def apply_manoeuvres(
    offsets: np.ndarray,
    t: float,
    events: Sequence[ManoeuvreEvent],
) -> np.ndarray:
    """Apply cumulative rotations and scalings to formation offsets at time t.

    For each event, progress p = clip((t - start) / (end - start), 0, 1).
        rotation angle  theta = sum over rotate events of  value_deg * p
        scale factor    k     = product over scale events of 1 + (value - 1) * p
        new offsets     = k * offsets @ R(theta)^T

    The formation rotates about the swarm centre, and a uniform scale
    commutes with rotation, so the order of the two does not matter.
    """
    theta_deg = 0.0
    scale = 1.0
    for event in events:
        p = _progress(t, event.start, event.end)
        if event.kind == "rotate":
            theta_deg += event.value * p
        else:
            scale *= 1.0 + (event.value - 1.0) * p
    if theta_deg == 0.0 and scale == 1.0:
        return offsets
    rot = rotation_matrix(math.radians(theta_deg))
    return scale * (offsets @ rot.T)


def min_separation(offsets: np.ndarray) -> float:
    """Smallest distance between any two slots (useful for collision checks)."""
    dist = pairwise_distances(offsets)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())
