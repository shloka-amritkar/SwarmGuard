"""Recovering true positions from radio ranges and absolute anchors.

IDEA
A GPS spoofer controls the GPS readings, but not the radio ranges between
drones and not the anchor fixes. So we rebuild the swarm without GPS:
  1. SHAPE: the radio ranges tell us every pairwise distance. Classical
     multidimensional scaling (MDS) turns a distance matrix back into
     points. The result has the right shape, but an arbitrary position,
     rotation and possibly a mirror image.
  2. PLACE: drones with anchor fixes tell us where a few of those points
     really are. A rigid fit (Kabsch) moves and rotates the MDS shape onto
     the anchors.
  3. COMPARE: gps_i - recovered_i is each drone's apparent GPS error. Under
     a rigid swarm shift D, every drone shows about D, so the mean of
     (gps - recovered) estimates the attack vector.

ROBUSTNESS (fix F4)
A faulty anchor would pull the whole fit. RANSAC tries small subsets of 3
anchors, keeps the fit most anchors agree with, and reports the others as
outliers. Robust alignment is the default.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from core.schema import ObservedFrame

logger = logging.getLogger(__name__)

MIN_ANCHORS = 3
MIRROR = np.array([1.0, -1.0])


class RecoveryStatus(str, Enum):
    """Outcome of a recovery attempt."""

    OK = "ok"
    INSUFFICIENT_ANCHORS = "insufficient_anchors"
    INSUFFICIENT_NEIGHBOURS = "insufficient_neighbours"
    DISCONNECTED = "disconnected"


class RecoveryError(Exception):
    """Raised when positions cannot be recovered (e.g. the range graph is split)."""


# --------------------------------------------------------------------------
# Step 1: shape from distances
# --------------------------------------------------------------------------
def fill_missing_distances(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN (unmeasured) distances with shortest paths through other drones.

    Floyd-Warshall: for every drone k, a route i -> k -> j may be shorter
    than the best known i -> j:
        D[i, j] = min(D[i, j], D[i, k] + D[k, j])
    WHY: MDS needs every distance. A path through neighbours is an upper
    bound on the straight-line distance, which is a reasonable estimate
    when drones are spread out. The filled pairs are reported, not hidden.

    Returns (filled matrix, mask of pairs that were filled).
    Raises RecoveryError if some pair is still unreachable (graph split).
    """
    d = np.array(dist, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"distance matrix must be square, got {d.shape}")
    missing = np.isnan(d)
    d[missing] = np.inf
    np.fill_diagonal(d, 0.0)
    for k in range(d.shape[0]):
        d = np.minimum(d, d[:, k, None] + d[None, k, :])
    if np.isinf(d).any():
        raise RecoveryError("range graph is disconnected; some drones cannot be placed")
    return d, missing & ~np.eye(d.shape[0], dtype=bool)


def classical_mds(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild 2D points from a distance matrix (classical MDS).

        J = I - (1/n) 1 1^T            (centering matrix)
        B = -0.5 * J D^2 J             (Gram matrix of centred points, B = X X^T)
        B = V Lambda V^T               (np.linalg.eigh)
        X = V_2 sqrt(Lambda_2)         (top two eigenvalues, negatives -> 0)

    WHY double centering: for centred points x_i, |x_i - x_j|^2 =
    x_i.x_i + x_j.x_j - 2 x_i.x_j; centring rows and columns of D^2 removes
    the first two terms and leaves -2 x_i.x_j, i.e. -2B.
    The result is correct up to rotation, reflection and translation.

    Returns (X (n, 2), filled_mask).
    """
    full, filled = fill_missing_distances(dist)
    n = full.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (full**2) @ j
    eigvals, eigvecs = np.linalg.eigh(b)          # ascending order
    top = np.argsort(eigvals)[::-1][:2]
    lam = np.clip(eigvals[top], 0.0, None)
    return eigvecs[:, top] * np.sqrt(lam), filled


# --------------------------------------------------------------------------
# Step 2: place the shape on the anchors
# --------------------------------------------------------------------------
def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best rotation R and translation t with b_i ~ R a_i + t (least squares).

        a_c = a - mean(a),  b_c = b - mean(b)
        H = a_c^T b_c;  U S V^T = svd(H)
        d = sign(det(V U^T));  R = V diag(1, d) U^T
        t = mean(b) - R mean(a)

    WHY d: without it the SVD may return a reflection (det R = -1), which is
    not a physical rotation. Mirror images are handled separately by
    fit_with_reflection.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 2:
        raise ValueError(f"kabsch needs two (k, 2) arrays with k >= 2, got {a.shape} and {b.shape}")
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    v = vt.T
    d = 1.0 if np.linalg.det(v @ u.T) >= 0 else -1.0
    rot = v @ np.diag([1.0, d]) @ u.T
    return rot, cb - rot @ ca


@dataclass
class AlignFit:
    """A rigid fit of (possibly mirrored) MDS points onto anchor fixes."""

    rot: np.ndarray
    trans: np.ndarray
    mirrored: bool

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Transform MDS points: (mirror if needed) then R x + t."""
        pts = x * MIRROR if self.mirrored else x
        return pts @ self.rot.T + self.trans


def fit_with_reflection(x_sub: np.ndarray, targets: np.ndarray) -> tuple[AlignFit, np.ndarray]:
    """Fit both the MDS points and their mirror image; keep the better one.

    WHY: MDS cannot tell a shape from its mirror image (distances are the
    same), so both must be tried. Returns (fit, per-point residuals).
    """
    best: tuple[AlignFit, np.ndarray] | None = None
    for mirrored in (False, True):
        pts = x_sub * MIRROR if mirrored else x_sub
        rot, trans = kabsch(pts, targets)
        fit = AlignFit(rot, trans, mirrored)
        res = np.linalg.norm(fit.apply(x_sub) - targets, axis=1)
        if best is None or res.sum() < best[1].sum():
            best = (fit, res)
    assert best is not None
    return best


@dataclass
class RansacResult:
    """Outcome of ransac_align."""

    status: RecoveryStatus
    fit: AlignFit | None = None
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _triplets(k: int, max_iter: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """All 3-anchor subsets if there are at most max_iter, else max_iter random ones."""
    if math.comb(k, 3) <= max_iter:
        return list(itertools.combinations(range(k), 3))
    return [tuple(sorted(rng.choice(k, size=3, replace=False))) for _ in range(max_iter)]


def ransac_align(
    x: np.ndarray,
    anchor_idx: np.ndarray,
    anchor_pos: np.ndarray,
    inlier_thresh: float = 3.0,
    max_iter: int = 200,
    rng: np.random.Generator | None = None,
) -> RansacResult:
    """Robustly fit MDS points onto anchors, rejecting faulty anchors.

    For each 3-anchor subset: fit, transform ALL anchors, count anchors
    whose residual < inlier_thresh. Keep the subset with the most inliers
    (ties: smaller residual sum), then refit on all its inliers.
    WHY 3: two points fix a 2D rigid transform; a third checks it.
    rng is only used when there are too many subsets to try them all; the
    default is a fixed-seed generator so results stay reproducible (R2).
    """
    idx = np.asarray(anchor_idx, dtype=int)
    targets = np.asarray(anchor_pos, dtype=float)
    k = len(idx)
    if k < MIN_ANCHORS:
        return RansacResult(RecoveryStatus.INSUFFICIENT_ANCHORS)
    rng = rng if rng is not None else np.random.default_rng(0)
    x_anchor = x[idx]
    best_key: tuple[int, float] | None = None
    best_inliers = np.ones(k, dtype=bool)
    for trip in _triplets(k, max_iter, rng):
        sel = list(trip)
        fit, _ = fit_with_reflection(x_anchor[sel], targets[sel])
        res = np.linalg.norm(fit.apply(x_anchor) - targets, axis=1)
        inliers = res < inlier_thresh
        key = (int(inliers.sum()), -float(res[inliers].sum()))
        if best_key is None or key > best_key:
            best_key, best_inliers = key, inliers
    if best_inliers.sum() < MIN_ANCHORS:
        # WHY fall back to all anchors: no consensus exists; a fit on all of
        # them is the least-bad answer, and the residuals expose the problem.
        best_inliers = np.ones(k, dtype=bool)
    fit, _ = fit_with_reflection(x_anchor[best_inliers], targets[best_inliers])
    residuals = np.linalg.norm(fit.apply(x_anchor) - targets, axis=1)
    return RansacResult(RecoveryStatus.OK, fit, best_inliers, residuals)


# --------------------------------------------------------------------------
# Step 3: one frame, and a whole run
# --------------------------------------------------------------------------
@dataclass
class RecoveryResult:
    """Recovered positions for one frame.

    positions        (N, 2) estimated true positions, or None if not possible
    inlier_mask      (N,) True for anchors used in the final fit
    outlier_anchors  drone indices whose anchor RANSAC rejected
    residual_to_gps  (N, 2) gps - recovered: each drone's apparent GPS error
    attack_vector    (2,) mean of residual_to_gps: estimated swarm-wide offset
    """

    step: int
    t: float
    status: RecoveryStatus
    method: str
    positions: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    outlier_anchors: list[int] = field(default_factory=list)
    residual_to_gps: np.ndarray | None = None
    attack_vector: np.ndarray | None = None
    filled_pairs: int = 0


def _fit_all_anchors(x: np.ndarray, idx: np.ndarray, pos: np.ndarray) -> RansacResult:
    """Non-robust alignment: one fit on every anchor (for comparison with F4)."""
    fit, res = fit_with_reflection(x[idx], pos)
    return RansacResult(RecoveryStatus.OK, fit, np.ones(len(idx), dtype=bool), res)


def recover_positions(
    obs: ObservedFrame,
    robust: bool = True,
    rng: np.random.Generator | None = None,
) -> RecoveryResult:
    """Estimate true positions from radio ranges + anchors (no GPS used)."""
    method = "mds_ransac" if robust else "mds_all_anchors"
    idx = np.flatnonzero(obs.anchor_mask)
    if len(idx) < MIN_ANCHORS:
        return RecoveryResult(obs.step, obs.t, RecoveryStatus.INSUFFICIENT_ANCHORS, method)
    dist = np.where(obs.range_mask, obs.ranges, np.nan)
    try:
        x, filled = classical_mds(dist)
    except RecoveryError:
        return RecoveryResult(obs.step, obs.t, RecoveryStatus.DISCONNECTED, method)
    pos = obs.anchor_pos[idx]
    align = ransac_align(x, idx, pos, rng=rng) if robust else _fit_all_anchors(x, idx, pos)
    assert align.fit is not None
    positions = align.fit.apply(x)
    inlier_mask = np.zeros(obs.n, dtype=bool)
    inlier_mask[idx[align.inliers]] = True
    residual = obs.gps_pos - positions
    return RecoveryResult(
        step=obs.step, t=obs.t, status=RecoveryStatus.OK, method=method,
        positions=positions, inlier_mask=inlier_mask,
        outlier_anchors=[int(i) for i in idx[~align.inliers]],
        residual_to_gps=residual, attack_vector=residual.mean(axis=0),
        filled_pairs=int(np.triu(filled, k=1).sum()),
    )


@dataclass
class RecoveryRun:
    """Recovery results for a whole run (None at steps that were skipped)."""

    times: np.ndarray
    results: list[RecoveryResult | None]

    def attack_vector_series(self) -> np.ndarray:
        """(T, 2) estimated attack vector per step; NaN where unavailable."""
        out = np.full((len(self.results), 2), np.nan)
        for k, r in enumerate(self.results):
            if r is not None and r.attack_vector is not None:
                out[k] = r.attack_vector
        return out

    def outlier_flags(self) -> np.ndarray:
        """(T,) True where the latest recovery so far reported outlier anchors.

        Forward-filled: skipped steps reuse the last computed answer.
        """
        flags = np.zeros(len(self.results), dtype=bool)
        last = False
        for k, r in enumerate(self.results):
            if r is not None:
                last = bool(r.outlier_anchors)
            flags[k] = last
        return flags

    def summary(self, window_s: float = 5.0) -> dict[str, object]:
        """Run-level summary: median attack vector over the last window_s seconds.

        WHY median over a window: single frames are noisy (GPS sigma ~1 m);
        the median of many frames is steadier and ignores odd frames.
        """
        done = [r for r in self.results if r is not None]
        ok = [r for r in done if r.status == RecoveryStatus.OK]
        if not ok:
            status = done[-1].status.value if done else "not_run"
            return {"method": done[-1].method if done else "-", "status": status,
                    "attack_vector": None, "attack_magnitude": None, "n_steps": 0,
                    "outlier_anchors": []}
        end = ok[-1].t
        window = [r for r in ok if r.t >= end - window_s]
        vec = np.median(np.stack([r.attack_vector for r in window]), axis=0)
        # WHY "at least half the steps": noise can make RANSAC reject an
        # honest anchor on an odd frame; a faulty one is rejected every time.
        counts: dict[int, int] = {}
        for r in window:
            for i in r.outlier_anchors:
                counts[i] = counts.get(i, 0) + 1
        outliers = sorted(i for i, c in counts.items() if c >= len(window) / 2)
        return {"method": ok[-1].method, "status": RecoveryStatus.OK.value,
                "attack_vector": (float(vec[0]), float(vec[1])),
                "attack_magnitude": float(np.linalg.norm(vec)), "n_steps": len(window),
                "outlier_anchors": outliers}


def recover_run(
    frames: list[ObservedFrame],
    robust: bool = True,
    every: int = 1,
) -> RecoveryRun:
    """Run recover_positions on every `every`-th frame (and always the last one)."""
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    results: list[RecoveryResult | None] = []
    for k, obs in enumerate(frames):
        if not isinstance(obs, ObservedFrame):
            raise TypeError(f"recovery must not see truth: expected ObservedFrame, got {type(obs).__name__}")
        if k % every == 0 or k == len(frames) - 1:
            results.append(recover_positions(obs, robust=robust))
        else:
            results.append(None)
    return RecoveryRun(times=np.array([f.t for f in frames]), results=results)
