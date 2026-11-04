"""Simulated sensors: GPS, radio ranging, velocity and absolute anchors.

Noise model (all Gaussian, each axis independent):

    GPS        gps      = true_pos + attack_offset + N(0, gps_sigma^2 I)
    velocity   vel_meas = true_vel + N(0, vel_sigma^2 I)
    ranging    r_ij     = |p_i - p_j| + N(0, range_sigma^2)
    anchor     a_i      = true_pos_i + N(0, anchor_sigma^2 I)  (+ fault bias)

WHY only GPS can be spoofed: ranging, velocity and anchors come from
independent physical channels (UWB radio between the drones, IMU/optical
flow on board, camera recognition of known landmarks). A GPS spoofer does
not control them. SwarmGuard's whole detection strategy is to cross-check
GPS against these channels. This assumption is stated openly in the
Methodology page; if an attacker could also corrupt them, the detectors
would have to be re-evaluated.

WHY each sensor has its own random generator: changing one sensor's
settings (e.g. turning on anchor dropout) must not change the random noise
drawn by any other sensor. With one shared generator, every later random
number would shift and two "identical" runs could not be compared.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config import SimConfig
from core.utils import pairwise_distances

logger = logging.getLogger(__name__)


@dataclass
class RngBundle:
    """One independent NumPy random generator per sensor."""

    gps: np.random.Generator
    range: np.random.Generator
    vel: np.random.Generator
    anchor: np.random.Generator

    @classmethod
    def from_seed_sequence(cls, seed_seq: np.random.SeedSequence) -> RngBundle:
        """Spawn four statistically independent child generators from one seed."""
        gps, rng_range, vel, anchor = (np.random.default_rng(s) for s in seed_seq.spawn(4))
        return cls(gps=gps, range=rng_range, vel=vel, anchor=anchor)


class SensorSuite:
    """All sensors of an N-drone swarm, with independent noise streams."""

    def __init__(self, config: SimConfig, rngs: RngBundle) -> None:
        self.config = config
        self.rngs = rngs
        self.n = config.n_drones
        # WHY drawn once here: a faulty anchor is biased in one FIXED
        # direction for the whole flight (like a mis-surveyed landmark),
        # not a new random direction every step.
        angles = rngs.anchor.uniform(0.0, 2.0 * np.pi, size=self.n)
        self.fault_directions = np.column_stack([np.cos(angles), np.sin(angles)])

    def gps_noise(self) -> np.ndarray:
        """Draw one step of GPS receiver noise, shape (N, 2): N(0, gps_sigma^2 I)."""
        return self.rngs.gps.normal(0.0, self.config.gps_sigma, size=(self.n, 2))

    def gps(
        self,
        true_pos: np.ndarray,
        offset: np.ndarray,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reported GPS: true_pos + offset + noise.

        `offset` is the (possibly zero) false offset from the attack model.
        `noise` may be passed in when the caller already drew it (the
        rigid_rotation attack needs the clean noisy reading first).
        """
        if noise is None:
            noise = self.gps_noise()
        return true_pos + offset + noise

    def velocity(self, true_vel: np.ndarray) -> np.ndarray:
        """Velocity sensor: true_vel + N(0, vel_sigma^2 I). Not spoofable."""
        noise = self.rngs.vel.normal(0.0, self.config.vel_sigma, size=(self.n, 2))
        return true_vel + noise

    def ranges(self, true_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inter-drone radio ranges and their validity mask.

        r_ij = max(0, |p_i - p_j| + e_ij),  e_ij ~ N(0, range_sigma^2)

        - Noise is drawn for the upper triangle (i < j) and mirrored, so
          r_ij = r_ji: both drones see one shared measurement.
        - Pairs farther apart than comm_range cannot hear each other:
          value NaN, mask False. The diagonal is 0 with mask False.
        - Values are clipped at 0 because a distance cannot be negative.
        - The NOISY values are what is stored and returned (fix F2: an
          earlier project computed noise but stored the clean distances).

        WHY noise for every pair, even out-of-range ones: the number of
        random draws per step stays constant, so changing comm_range never
        changes the noise on the pairs that remain in range.
        """
        n = self.n
        true_dist = pairwise_distances(true_pos)
        iu = np.triu_indices(n, k=1)
        noisy_upper = true_dist[iu] + self.rngs.range.normal(0.0, self.config.range_sigma, size=len(iu[0]))
        noisy = np.zeros((n, n))
        noisy[iu] = np.maximum(noisy_upper, 0.0)
        noisy = noisy + noisy.T
        mask = true_dist <= self.config.comm_range
        np.fill_diagonal(mask, False)
        ranges = np.where(mask, noisy, np.nan)
        np.fill_diagonal(ranges, 0.0)
        return ranges, mask

    def anchors(
        self,
        true_pos: np.ndarray,
        anchored: np.ndarray,
        faulty: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Absolute anchor fixes for anchored drones, and their availability mask.

        a_i = true_pos_i + N(0, anchor_sigma^2 I)
        faulty anchors:  a_i += anchor_fault_bias * d_i   (d_i fixed unit vector)
        dropout: each anchored drone misses its reading with probability
                 anchor_dropout -> row is NaN and mask False.
        Drones without an anchor always have NaN and mask False.
        """
        noise = self.rngs.anchor.normal(0.0, self.config.anchor_sigma, size=(self.n, 2))
        dropout_draw = self.rngs.anchor.uniform(0.0, 1.0, size=self.n)
        readings = true_pos + noise
        bias = self.config.anchor_fault_bias * self.fault_directions
        readings = np.where(faulty[:, None], readings + bias, readings)
        mask = anchored & (dropout_draw >= self.config.anchor_dropout)
        readings = np.where(mask[:, None], readings, np.nan)
        return readings, mask
