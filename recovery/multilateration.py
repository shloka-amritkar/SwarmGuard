"""Repairing one drone's position from its neighbours' ranges (multilateration).

IDEA
If drone i's GPS is untrustworthy but its neighbours j are trusted, the
radio ranges r_ij say how far i is from each neighbour. The point p that
best fits all those distances is i's estimated position:
    e_j(p) = |p - gps_j| - r_ij        (one residual per neighbour)
    p* = argmin_p  sum_j rho(e_j(p)^2)

FIX F1: neighbours that are themselves flagged are NEVER used.
WHY: a spoofed neighbour's GPS is wrong, so ranging to it would import the
attacker's offset into the repair. An earlier project used every neighbour.

ROBUST LOSS (soft_l1)
rho(z) = 2 (sqrt(1 + z) - 1): behaves like least squares for small errors
but grows only linearly for large ones, so one bad range cannot dominate.

LIMITATION
Under a swarm-wide shift every drone is spoofed (and flagged), so no
trusted neighbours remain and the status is INSUFFICIENT_NEIGHBOURS. That
case is handled by MDS + anchors (recovery/mds_alignment.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from core.schema import Frame, ObservedFrame
from recovery.mds_alignment import RecoveryStatus

logger = logging.getLogger(__name__)

MIN_NEIGHBOURS = 3


@dataclass
class RepairResult:
    """Multilateration result for one drone.

    position     (2,) estimate, or None when there were too few neighbours
    covariance   (2, 2) approximate uncertainty of the estimate (m^2)
    neighbours   indices of the trusted neighbours actually used
    residual_rms root-mean-square of the final range residuals (m)
    """

    drone: int
    status: RecoveryStatus
    position: np.ndarray | None = None
    covariance: np.ndarray | None = None
    neighbours: list[int] = field(default_factory=list)
    residual_rms: float = float("nan")


def trusted_neighbours(i: int, obs: ObservedFrame, flagged: np.ndarray) -> np.ndarray:
    """Indices j != i with a valid range to i and NOT flagged (fix F1)."""
    ok = obs.range_mask[i] & ~np.asarray(flagged, dtype=bool)
    ok[i] = False
    return np.flatnonzero(ok)


def repair_drone(
    i: int,
    obs: ObservedFrame,
    flagged: np.ndarray,
    init: np.ndarray | None = None,
) -> RepairResult:
    """Estimate drone i's position from trusted neighbours' GPS and ranges.

    Covariance (Gauss-Newton approximation at the solution):
        cov ~ inv(J^T J) * sum(res^2) / (m - 2)
    where J is the Jacobian of the residuals, m the number of neighbours
    and 2 the number of unknowns (x, y).
    """
    if not isinstance(obs, ObservedFrame):
        raise TypeError("recovery must not see truth: expected ObservedFrame")
    if not (0 <= i < obs.n):
        raise ValueError(f"drone index {i} out of range for {obs.n} drones")
    nbrs = trusted_neighbours(i, obs, flagged)
    if len(nbrs) < MIN_NEIGHBOURS:
        return RepairResult(i, RecoveryStatus.INSUFFICIENT_NEIGHBOURS, neighbours=nbrs.tolist())
    anchors = obs.gps_pos[nbrs]
    ranges = obs.ranges[i, nbrs]

    def residuals(p: np.ndarray) -> np.ndarray:
        return np.linalg.norm(anchors - p, axis=1) - ranges

    x0 = np.asarray(init, dtype=float) if init is not None else anchors.mean(axis=0)
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=1.0)
    res = sol.fun
    m = len(nbrs)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * float(res @ res) / max(m - 2, 1)
    except np.linalg.LinAlgError:
        # WHY: neighbours exactly on a line give a singular J^T J; the
        # position is then ambiguous and the uncertainty unbounded.
        cov = np.full((2, 2), np.inf)
    return RepairResult(i, RecoveryStatus.OK, sol.x, cov, nbrs.tolist(),
                        float(np.sqrt(np.mean(res**2))))


def repair_all(obs: ObservedFrame, flagged: np.ndarray) -> dict[int, RepairResult]:
    """Repair every flagged drone. Returns {drone index: RepairResult}."""
    flags = np.asarray(flagged, dtype=bool)
    if flags.shape != (obs.n,):
        raise ValueError(f"flagged must have shape ({obs.n},), got {flags.shape}")
    return {int(i): repair_drone(int(i), obs, flags) for i in np.flatnonzero(flags)}


def recovery_error(frame: Frame, repaired: dict[int, RepairResult]) -> pd.DataFrame:
    """EVALUATION ONLY: compare GPS error and repaired error against the truth.

    Columns: drone, status, gps_error_m, repaired_error_m (NaN if no repair).
    This reads Frame.true_pos and must never be called by a detector.
    """
    rows = []
    for i, rep in sorted(repaired.items()):
        gps_err = float(np.linalg.norm(frame.observed.gps_pos[i] - frame.true_pos[i]))
        rep_err = (float(np.linalg.norm(rep.position - frame.true_pos[i]))
                   if rep.position is not None else float("nan"))
        rows.append({"drone": i, "status": rep.status.value,
                     "gps_error_m": gps_err, "repaired_error_m": rep_err})
    return pd.DataFrame(rows, columns=["drone", "status", "gps_error_m", "repaired_error_m"])
