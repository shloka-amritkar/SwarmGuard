"""Simulated GPS spoofing: false offsets added to REPORTED GPS positions.

An attack is modelled as an offset vector per drone:

    reported_gps_i(t) = true_pos_i(t) + noise_i(t) + offset_i(t)

Only the GPS reading changes. Ranging, velocity and anchors are untouched
(see simulation/sensors.py for why). This is software arithmetic on
simulated numbers; there is no signal generation and no real-world use.

Notation used below:
    u          = (cos d, sin d), the unit direction for direction_deg = d
    p          = clip((t - start) / ramp_duration, 0, 1), ramp progress
    s(p)       = ramp shape in [0, 1] (see ramp_shape)
    M          = magnitude_m

Labels (fix F5): a drone is labelled attacked for the WHOLE active window
start <= t < end, even when its momentary offset is small (start of a
ramp) or zero (the "off" half of an intermittent attack). WHY: the drone is
under attack during that whole time; an earlier project labelled only the
"on" samples, which taught its ML model that a paused attack is normal.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from config import AttackConfig
from core.schema import AttackType
from core.utils import rotation_matrix

logger = logging.getLogger(__name__)

# Steepness constants of the ramp shapes (fixed by the specification).
_EXP_RATE = 5.0
_SIGMOID_STEEPNESS = 12.0


def ramp_progress(t: float, start: float, ramp_duration: float) -> float:
    """p = clip((t - start) / ramp_duration, 0, 1)."""
    return min(max((t - start) / ramp_duration, 0.0), 1.0)


def _logistic(x: float) -> float:
    """Logistic function L(x) = 1 / (1 + e^-x)."""
    return 1.0 / (1.0 + math.exp(-x))


def ramp_shape(p: float, profile: str) -> float:
    """How much of the full offset is applied at ramp progress p (0..1).

    step:         s = 1 (full size as soon as the attack starts)
    linear:       s = p
    exponential:  s = (1 - e^{-5p}) / (1 - e^{-5})     fast start, slow finish
    polynomial:   s = p^3                              slow start, fast finish
    sigmoid:      s = (L(12(p - 0.5)) - L(-6)) / (L(6) - L(-6))   S-curve
    Every non-step profile satisfies s(0) = 0 and s(1) = 1.

    WHY several profiles: a real spoofer ramps slowly to avoid triggering
    jump detectors; the shapes let experiments test how detection depends
    on how gradually an attack grows.
    """
    if profile == "step":
        return 1.0
    if profile == "linear":
        return p
    if profile == "exponential":
        return (1.0 - math.exp(-_EXP_RATE * p)) / (1.0 - math.exp(-_EXP_RATE))
    if profile == "polynomial":
        return p**3
    if profile == "sigmoid":
        half = _SIGMOID_STEEPNESS / 2.0
        low, high = _logistic(-half), _logistic(half)
        return (_logistic(_SIGMOID_STEEPNESS * (p - 0.5)) - low) / (high - low)
    raise ValueError(f"unknown ramp profile {profile!r}")


class AttackModel:
    """Computes the per-drone GPS offsets and labels for one AttackConfig.

    Usage:  offset, labels = model.offsets(t, clean_reported)
        offset  (N, 2) metres to ADD to each drone's reported GPS
        labels  (N,) True for every drone affected by an active attack
    """

    def __init__(self, config: AttackConfig, n_drones: int) -> None:
        self.config = config
        self.n = n_drones
        bad = [t for t in config.targets if t >= n_drones]
        if bad:
            raise ValueError(f"attack targets {bad} do not exist in a swarm of {n_drones} drones")
        self.type = AttackType(config.type)
        d = math.radians(config.direction_deg)
        self.direction = np.array([math.cos(d), math.sin(d)])
        self.targets = np.array(config.targets, dtype=int)
        self.affected = self._affected_drones()
        self._multi_vectors = self._draw_multi_vectors()

    def _affected_drones(self) -> np.ndarray:
        """Indices of drones whose GPS this attack changes."""
        if self.type == AttackType.NONE:
            return np.array([], dtype=int)
        if self.type in (AttackType.SWARM_WIDE, AttackType.RIGID_ROTATION):
            return np.arange(self.n)
        return self.targets

    def _draw_multi_vectors(self) -> np.ndarray:
        """Seeded offset vectors for multi_drone: own direction and size per target.

        direction ~ Uniform(0, 2 pi), size ~ Uniform(0.7, 1.3) * M.
        WHY seeded from AttackConfig.seed: the same attack config always
        produces the same coordinated attack (rule R2).
        """
        rng = np.random.default_rng(self.config.seed)
        k = len(self.targets)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=k)
        sizes = rng.uniform(0.7, 1.3, size=k) * self.config.magnitude_m
        return sizes[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])

    def is_active(self, t: float) -> bool:
        """Active window: start <= t and (end is None or t < end)."""
        cfg = self.config
        if self.type == AttackType.NONE or t < cfg.start_time:
            return False
        return cfg.end_time is None or t < cfg.end_time

    def _shape(self, t: float) -> float:
        """Ramp shape s(p) at time t."""
        cfg = self.config
        return ramp_shape(ramp_progress(t, cfg.start_time, cfg.ramp_duration), cfg.ramp_profile)

    def _drift_distance(self, t: float) -> float:
        """Drift length drift_rate * (t - start), capped at M if drift_cap is set."""
        cfg = self.config
        dist = cfg.drift_rate * (t - cfg.start_time)
        if cfg.drift_cap and cfg.magnitude_m > 0:
            dist = min(dist, cfg.magnitude_m)
        return dist

    def offsets(
        self,
        t: float,
        clean_reported: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (offset (N, 2), labels (N,)) at time t.

        clean_reported: the GPS readings this step BEFORE the attack (true
        position + receiver noise). Needed only by rigid_rotation.
        """
        offset = np.zeros((self.n, 2))
        labels = np.zeros(self.n, dtype=bool)
        if not self.is_active(t):
            return offset, labels
        labels[self.affected] = True
        handler = {
            AttackType.CONSTANT_OFFSET: self._constant,
            AttackType.GRADUAL_DRIFT: self._gradual_drift,
            AttackType.MULTI_DRONE: self._multi_drone,
            AttackType.SWARM_WIDE: self._swarm_wide,
            AttackType.SUBSET_RIGID: self._subset_rigid,
            AttackType.INTERMITTENT: self._intermittent,
        }.get(self.type)
        if handler is not None:
            handler(t, offset)
        else:
            offset[:] = self._rigid_rotation(t, clean_reported)
        return offset, labels

    # ---- one method per attack type; each writes into `offset` ----------
    def _constant(self, t: float, offset: np.ndarray) -> None:
        """constant_offset: one target gets M * u * s(p)."""
        offset[self.targets] = self.config.magnitude_m * self._shape(t) * self.direction

    def _gradual_drift(self, t: float, offset: np.ndarray) -> None:
        """gradual_drift: one target gets drift_rate * (t - start) * u."""
        offset[self.targets] = self._drift_distance(t) * self.direction

    def _multi_drone(self, t: float, offset: np.ndarray) -> None:
        """multi_drone: each target gets its own seeded vector, times s(p)."""
        offset[self.targets] = self._multi_vectors * self._shape(t)

    def _swarm_wide(self, t: float, offset: np.ndarray) -> None:
        """swarm_wide: the SAME offset D on all drones.

        D = drift_rate * (t - start) * u   if drift_rate > 0 and profile is linear
        D = M * s(p) * u                   otherwise

        Detector A cannot see this, because the offset cancels in every
        pairwise difference:  (g_i + D) - (g_j + D) = g_i - g_j.
        """
        cfg = self.config
        if cfg.drift_rate > 0 and cfg.ramp_profile == "linear":
            size = self._drift_distance(t)
        else:
            size = cfg.magnitude_m * self._shape(t)
        offset[:] = size * self.direction

    def _subset_rigid(self, t: float, offset: np.ndarray) -> None:
        """subset_rigid: the same offset M * u * s(p) on the listed targets only."""
        offset[self.targets] = self.config.magnitude_m * self._shape(t) * self.direction

    def _intermittent(self, t: float, offset: np.ndarray) -> None:
        """intermittent: M * u * s(p) when ((t - start) mod period) < duty * period."""
        cfg = self.config
        phase = (t - cfg.start_time) % cfg.pulse_period
        if phase < cfg.pulse_duty * cfg.pulse_period:
            offset[self.targets] = cfg.magnitude_m * self._shape(t) * self.direction

    def _rigid_rotation(self, t: float, clean_reported: np.ndarray | None) -> np.ndarray:
        """rigid_rotation: rotate the whole reported formation about its centroid.

        c        = mean of clean reported positions q_i
        theta(t) = rotation_rate * (t - start)
        q_i'     = R(theta) (q_i - c) + c + M * s(p) * u
        offset_i = q_i' - q_i

        Proof that pairwise distances are unchanged (so Detector A is blind):
            q_i' - q_j' = R(q_i - c) - R(q_j - c) = R (q_i - q_j)
            |R v| = |v| for any rotation matrix R (R^T R = I), hence
            |q_i' - q_j'| = |q_i - q_j|.
        The shift term M * s(p) * u is the same for every drone and also
        cancels in the difference.
        """
        if clean_reported is None:
            raise ValueError("rigid_rotation needs clean_reported positions")
        cfg = self.config
        centroid = clean_reported.mean(axis=0)
        theta = math.radians(cfg.rotation_rate_deg_s * (t - cfg.start_time))
        rotated = (clean_reported - centroid) @ rotation_matrix(theta).T + centroid
        shift = cfg.magnitude_m * self._shape(t) * self.direction
        return rotated + shift - clean_reported
