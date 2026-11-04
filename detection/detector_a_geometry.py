"""Detector A: relative geometry (GPS distances vs. radio ranges).

IDEA
Every pair of drones (i, j) can measure the distance between them in two
independent ways:
    from GPS:    G_ij = |gps_i - gps_j|
    from radio:  r_ij (UWB-style ranging, cannot be spoofed by a GPS attacker)
If one drone's GPS is spoofed, its GPS distances to every partner become
wrong, while the radio ranges stay correct. The residual
    R_ij = |G_ij - r_ij|
is then large on every pair that involves the spoofed drone.

MAJORITY VOTE
    pair (i, j) is "suspicious"  if  R_ij > tau_a
    score_i = (number of suspicious pairs of i) / (number of valid pairs of i)
    raw_flag_i = score_i > vote_fraction     (strictly more than half)
WHY: a spoofed drone disagrees with every partner, so score ~ 1. An honest
drone has only one bad pair (the one with the liar) out of nine, so score
~ 0.11. With two liars an honest drone has 2/9. Voting only breaks down
when close to half the swarm is attacked; Phase 3 experiments show this.

THE BLIND SPOT (proof)
If every drone's GPS is shifted by the same vector D (a swarm-wide spoof):
    (g_i + D) - (g_j + D) = g_i - g_j
so every G_ij is unchanged and every residual is unchanged. If every GPS is
rotated by the same rotation Q about a point c:
    |Q(g_i - c) - Q(g_j - c)| = |Q (g_i - g_j)| = |g_i - g_j|
because rotations preserve length. Therefore Detector A is mathematically
unable to see a rigid swarm shift or rotation. This is expected behaviour,
not a bug; it is the "RigidShift" blind spot formalised by Park & Yoo.
"""

from __future__ import annotations

import logging

import numpy as np

from config import DetectorConfig
from core.schema import ObservedFrame
from core.utils import pairwise_distances
from detection.base import Detector, PersistenceFilter, StepResult, status_from_flags

logger = logging.getLogger(__name__)

# A drone needs at least this many valid pairs for its vote to mean anything.
MIN_VALID_PAIRS = 2


def geometry_residuals(obs: ObservedFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, valid): the (N, N) residual matrix and its validity mask.

    R_ij = |G_ij - r_ij| where G = pairwise GPS distances and r = ranges.
    valid_ij = range_mask_ij (and i != j). R is NaN where not valid.
    """
    gps_dist = pairwise_distances(obs.gps_pos)
    valid = obs.range_mask.copy()
    np.fill_diagonal(valid, False)
    residual = np.where(valid, np.abs(gps_dist - obs.ranges), np.nan)
    return residual, valid


def vote(
    residual: np.ndarray,
    valid: np.ndarray,
    tau_a: float,
    vote_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Majority vote for one step (or many steps at once).

    Works on arrays shaped (..., N, N): the last two axes are the pair
    matrix, any leading axes (e.g. time) are kept.

    Returns (scores, raw_flags, has_evidence, bad_counts):
        bad_i      = #{j : valid_ij and R_ij > tau_a}
        valid_i    = #{j : valid_ij}
        score_i    = bad_i / valid_i          (NaN if valid_i < 2)
        raw_flag_i = score_i > vote_fraction  (False if no evidence)
    """
    # WHY fill NaN with 0 before comparing: NaN > tau gives False anyway,
    # but filling first avoids NumPy "invalid value" warnings.
    suspicious = valid & (np.nan_to_num(residual, nan=0.0) > tau_a)
    bad = suspicious.sum(axis=-1)
    n_valid = valid.sum(axis=-1)
    has_evidence = n_valid >= MIN_VALID_PAIRS
    scores = np.where(has_evidence, bad / np.maximum(n_valid, 1), np.nan)
    raw = has_evidence & (np.nan_to_num(scores, nan=0.0) > vote_fraction)
    return scores, raw, has_evidence, bad


def swarm_residual_score(residual: np.ndarray, valid: np.ndarray) -> float:
    """95th percentile of the valid residuals (each pair counted once).

    WHY 95th percentile and not the mean: with one liar among ten drones,
    only 9 of 45 pairs are bad. The mean would dilute those 9 large values
    among 36 small ones; the 95th percentile still shows them.
    """
    upper = np.triu(valid, k=1)
    values = residual[upper]
    return float(np.percentile(values, 95)) if values.size else float("nan")


class GeometryDetector(Detector):
    """Detector A: GPS-derived distances vs. radio ranges, with neighbour voting."""

    name = "Relative geometry"
    short = "A"

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__()
        self.config = config

    def _reset_state(self, n: int) -> None:
        self._persist = PersistenceFilter(self.config.persist_m, self.config.persist_n, n)

    def _step(self, obs: ObservedFrame) -> StepResult:
        cfg = self.config
        residual, valid = geometry_residuals(obs)
        scores, raw, has_evidence, bad = vote(residual, valid, cfg.tau_a, cfg.vote_fraction)
        flags = self._persist.update(raw)
        return StepResult(
            step=obs.step,
            t=obs.t,
            flags=flags,
            raw_flags=raw,
            scores=scores,
            status=status_from_flags(flags, raw, has_evidence),
            swarm_flag=bool(flags.any()),
            swarm_score=swarm_residual_score(residual, valid),
            diagnostics={"residuals": residual, "bad_counts": bad, "valid_counts": valid.sum(axis=1)},
        )
