"""Shared mathematical helpers used across SwarmGuard.

Everything here is a small pure function: no randomness, no global state.

Coordinate convention (rule R3): a local East-North frame in metres,
x = east, y = north, with its origin at Pune (lat0, lon0). Detector maths is
always done in metres; latitude/longitude are used only for drawing maps.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Reference origin of the local frame: Pune, India.
LAT0: float = 18.5204
LON0: float = 73.8567
# Approximate metres in one degree of latitude.
METRES_PER_DEG: float = 111_320.0


def enu_to_latlon(
    x: np.ndarray | float,
    y: np.ndarray | float,
    lat0: float = LAT0,
    lon0: float = LON0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local East-North metres to latitude/longitude (display only).

    lat = lat0 + y / 111320
    lon = lon0 + x / (111320 * cos(lat0))

    WHY the cosine: lines of longitude meet at the poles, so one degree of
    longitude is shorter than one degree of latitude by a factor cos(lat).
    At Pune (18.5 deg N) that factor is 0.948. This "flat earth" formula is
    accurate to millimetres over the few hundred metres a swarm flies.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    lat = lat0 + y_arr / METRES_PER_DEG
    lon = lon0 + x_arr / (METRES_PER_DEG * math.cos(math.radians(lat0)))
    return lat, lon


def latlon_to_enu(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    lat0: float = LAT0,
    lon0: float = LON0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of enu_to_latlon: latitude/longitude to East-North metres.

    y = (lat - lat0) * 111320
    x = (lon - lon0) * 111320 * cos(lat0)
    """
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    y = (lat_arr - lat0) * METRES_PER_DEG
    x = (lon_arr - lon0) * METRES_PER_DEG * math.cos(math.radians(lat0))
    return x, y


def rotation_matrix(theta: float) -> np.ndarray:
    """Return the 2x2 counter-clockwise rotation matrix for angle theta (radians).

    R(theta) = [[cos theta, -sin theta],
                [sin theta,  cos theta]]

    A point p (column vector) rotated about the origin is R @ p. For many
    points stored as rows of an (N, 2) array P, the same rotation is P @ R.T.
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Return the (N, N) matrix of Euclidean distances between N 2D points.

    D[i, j] = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)

    WHY broadcasting: points[:, None, :] has shape (N, 1, 2) and
    points[None, :, :] has shape (1, N, 2); subtracting them gives all N*N
    difference vectors at once, with no Python loop.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def clip_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    """Shorten any row vector longer than max_norm, keeping its direction.

    v_clipped = v * min(1, max_norm / |v|)

    WHY: a real drone cannot fly faster than its motors allow. Clipping the
    commanded velocity stops the simulation producing physically impossible
    true motion, for example when a large spoofed jump makes the controller
    ask for a huge correction.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")
    v = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    # WHY np.maximum: avoids dividing by zero for a zero vector (scale stays 1).
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return v * scale


def chi2_2dof_quantile(p: float) -> float:
    """Return x such that P(X <= x) = p for X ~ chi-square with 2 dof.

    For 2 degrees of freedom the tail is P(X > x) = exp(-x / 2), so
        1 - p = exp(-x / 2)   =>   x = -2 ln(1 - p).
    Example: p = 0.99 gives x = 9.21.
    """
    if not (0.0 <= p < 1.0):
        raise ValueError(f"p must be in [0, 1), got {p}")
    return -2.0 * math.log(1.0 - p)


def rayleigh_quantile(sigma: float, p: float) -> float:
    """Return the p-quantile of the length of a 2D Gaussian error vector.

    If e_x, e_y ~ N(0, sigma^2) independently, |e| is Rayleigh(sigma) with
    P(|e| > r) = exp(-r^2 / (2 sigma^2)). Solving for r:
        r = sigma * sqrt(-2 ln(1 - p)).
    Example: sigma = 1.118, p = 0.999 gives r = 4.16 m.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if not (0.0 <= p < 1.0):
        raise ValueError(f"p must be in [0, 1), got {p}")
    return sigma * math.sqrt(-2.0 * math.log(1.0 - p))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score confidence interval for a proportion k / n.

    p      = k / n
    centre = (p + z^2 / (2n)) / (1 + z^2 / n)
    half   = z * sqrt(p (1 - p) / n + z^2 / (4 n^2)) / (1 + z^2 / n)
    interval = [centre - half, centre + half]

    WHY Wilson instead of p +/- z*sqrt(p(1-p)/n): the simple formula gives
    an interval of width zero when p is 0 or 1, so detecting 50 out of 50
    attacks would be reported as "100 %, certain". Wilson still gives a
    sensible range (about 93 % to 100 % for 50/50). z = 1.96 means 95 %.

    With n = 0 there is no data, so the honest answer is the full range [0, 1].
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def round_half_up(x: float) -> int:
    """Round to the nearest integer, with .5 always rounding up.

    WHY: Python's built-in round() uses "banker's rounding" (round(2.5) = 2),
    which is surprising when choosing e.g. how many drones carry anchors.
    """
    return int(math.floor(x + 0.5))


def drone_label(index: int) -> str:
    """0-based drone index -> display name, e.g. 3 -> "Drone 04".

    WHY here: the CLI, the explainer and the PDF report all name drones the
    same way; one shared helper keeps them consistent.
    """
    return f"Drone {int(index) + 1:02d}"


def format_drones(drones: Iterable[int]) -> str:
    """Indices -> "Drone 02, Drone 06" (sorted), or "none" if empty."""
    return ", ".join(drone_label(i) for i in sorted(drones)) or "none"
