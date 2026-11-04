"""How one spoofed drone disturbs the rest of a closed-loop swarm.

IDEA (inspired by SwarmFuzz, Yao et al., DSN 2023)
In closed loop every drone steers using the swarm's GPS-estimated centre
(simulation/swarm.py). A false GPS reading on drone i shifts that estimate,
so the controller nudges OTHER drones too: with centre_gain = 0, a spoof D
on one of N drones permanently displaces every other drone by about D / N.
This module measures that spread and ranks which drones matter most.

SENSITIVITY MATRIX (truth used: this is an evaluation tool)
    baseline = clean run;  run_i = same seed, only drone i spoofed
    M[i, j]  = mean over t >= attack start of |true_j(run_i) - true_j(baseline)|
WHY the same seed: both runs draw identical sensor noise, so any
difference in true positions is caused by the attack alone.
WHY average only after the attack starts: before it the runs are identical
by construction, and including those zeros would just dilute M.
The diagonal (how far the spoofed drone itself is dragged) is reported
separately as self_displacement and set to 0 in M.

PAGERANK (implemented here; no networkx)
Graph convention: W[j, i] = weight of the edge i -> j.
Column-normalise W (a column of zeros becomes uniform), then iterate
    r <- d * W_norm r + (1 - d) / n     until the change is < tol.
A node scores high when high-scoring nodes point to it.
    influence      = pagerank(M):   column j of M lists how much each drone
                     i displaces j, so j "votes" for the drones that move it.
                     A drone ranks high if spoofing it displaces many drones.
    susceptibility = pagerank(M^T): ranks drones that get pushed around.

DESIGN DECISION D1: results depend on centre_gain, so propagation_study
always runs BOTH 0.0 and 0.5 and labels every result with its value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from config import AttackConfig, SimConfig
from core.schema import SimResult
from core.utils import pairwise_distances
from simulation.swarm import simulate

logger = logging.getLogger(__name__)

NEAR_COLLISION_M = 1.0


def _require_closed_loop(sim: SimConfig) -> None:
    """Propagation only exists when drones steer by GPS (closed loop)."""
    if sim.loop_mode != "closed":
        raise ValueError("propagation analysis needs loop_mode 'closed'; in open loop "
                         "drones steer by true position, so a spoof cannot spread")


def pagerank(w: np.ndarray, damping: float = 0.85, tol: float = 1e-9, max_iter: int = 1000) -> np.ndarray:
    """PageRank by power iteration. W[j, i] = weight of edge i -> j (see module docstring)."""
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"W must be square, got {w.shape}")
    if (w < 0).any():
        raise ValueError("PageRank weights must be >= 0")
    if not (0.0 < damping < 1.0):
        raise ValueError(f"damping must be in (0, 1), got {damping}")
    n = w.shape[0]
    col_sums = w.sum(axis=0)
    # WHY uniform for empty columns: a node with no out-links would
    # otherwise leak its score; spreading it evenly keeps the total at 1.
    norm = np.where(col_sums > 0, w / np.where(col_sums > 0, col_sums, 1.0), 1.0 / n)
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = damping * norm @ r + (1.0 - damping) / n
        if np.abs(new - r).sum() < tol:
            return new
        r = new
    logger.warning("PageRank did not converge in %d iterations", max_iter)
    return r


@dataclass
class NearCollision:
    """Two drones closer than NEAR_COLLISION_M (true positions)."""

    t: float
    i: int
    j: int
    distance: float


def min_separation(result: SimResult, limit: float = NEAR_COLLISION_M) -> tuple[np.ndarray, list[NearCollision]]:
    """Minimum pairwise TRUE distance per step, plus near-collision events.

    An event is recorded when a pair drops below `limit`. The pair is
    counted again only after it has separated beyond 2 x limit.
    WHY this hysteresis: noise makes a close pass flicker around the limit;
    without it one pass would be counted dozens of times.
    """
    series = np.zeros(result.n_steps)
    events: list[NearCollision] = []
    active: set[tuple[int, int]] = set()
    for k, frame in enumerate(result.frames):
        dist = pairwise_distances(frame.true_pos)
        np.fill_diagonal(dist, np.inf)
        series[k] = dist.min()
        ii, jj = np.nonzero(np.triu(dist < limit, k=1))
        for pair in sorted({(int(i), int(j)) for i, j in zip(ii, jj)} - active):
            events.append(NearCollision(frame.t, pair[0], pair[1], float(dist[pair])))
            active.add(pair)
        active = {p for p in active if dist[p] < 2.0 * limit}
    return series, events


@dataclass
class PropagationResult:
    """Sensitivity matrix and rankings for one centre_gain value."""

    centre_gain: float
    magnitude: float
    seed: int
    start_time: float
    matrix: np.ndarray                       # (N, N), zero diagonal
    self_displacement: np.ndarray            # (N,)
    influence: np.ndarray                    # (N,) PageRank of M
    susceptibility: np.ndarray               # (N,) PageRank of M^T
    min_separation_baseline: float
    min_separation_attacked: np.ndarray      # (N,) min separation when drone i is spoofed
    near_collisions: dict[int, list[NearCollision]] = field(default_factory=dict)


def sensitivity_matrix(
    base: SimConfig,
    magnitude: float = 6.0,
    seed: int = 5000,
    start_time: float = 20.0,
    direction_deg: float = 0.0,
) -> PropagationResult:
    """Spoof each drone alone and measure how far every other drone is displaced."""
    _require_closed_loop(base)
    if not (0.0 <= start_time < base.duration):
        raise ValueError(f"start_time must be in [0, duration), got {start_time}")
    sim = replace(base, seed=int(seed))
    baseline = simulate(sim, AttackConfig())
    base_true = baseline.stack("true_pos")
    after = baseline.times() >= start_time
    n = sim.n_drones
    m = np.zeros((n, n))
    min_sep = np.zeros(n)
    near: dict[int, list[NearCollision]] = {}
    for i in range(n):
        atk = AttackConfig(type="constant_offset", targets=[i], magnitude_m=magnitude,
                           direction_deg=direction_deg, start_time=start_time, seed=int(seed))
        attacked = simulate(sim, atk)
        disp = np.linalg.norm(attacked.stack("true_pos") - base_true, axis=-1)   # (T, N)
        m[i] = disp[after].mean(axis=0)
        series, events = min_separation(attacked)
        min_sep[i] = series.min()
        if events:
            near[i] = events
    self_disp = np.diag(m).copy()
    np.fill_diagonal(m, 0.0)
    return PropagationResult(
        centre_gain=sim.centre_gain, magnitude=magnitude, seed=int(seed), start_time=start_time,
        matrix=m, self_displacement=self_disp, influence=pagerank(m), susceptibility=pagerank(m.T),
        min_separation_baseline=float(min_separation(baseline)[0].min()),
        min_separation_attacked=min_sep, near_collisions=near,
    )


def propagation_study(
    base: SimConfig,
    centre_gains: tuple[float, ...] = (0.0, 0.5),
    magnitude: float = 6.0,
    seed: int = 5000,
    start_time: float = 20.0,
) -> list[PropagationResult]:
    """Run sensitivity_matrix once per centre_gain (D1: both 0 and 0.5 by default)."""
    return [sensitivity_matrix(replace(base, centre_gain=g), magnitude, seed, start_time)
            for g in centre_gains]
