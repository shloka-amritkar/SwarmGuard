"""Detector B: motion consistency (GPS vs. the drone's own velocity sensor).

IDEA
Each drone carries a velocity sensor (IMU / optical flow) that a GPS
spoofer does not control. If we know where a drone was and how fast it
flew, we can predict where it should be now. GPS that keeps disagreeing
with that prediction is suspicious. Because every drone is checked against
its OWN physics, B also sees swarm-wide attacks that Detector A cannot.

KALMAN FILTER (one per drone, 2D position state, metres; fix F7)
    state x = estimated position (2,), covariance P (2x2)
    init:       x = first GPS reading,  P = gps_sigma^2 I
    predict:    x_pred = x + u * dt        u = velocity measured on the
                                           previous step (timing rule D4)
                P_pred = P + Q,  Q = (vel_sigma * dt)^2 I + 1e-4 I
    innovation: y = z - x_pred              z = this step's GPS
                S = P_pred + R,  R = gps_sigma^2 I
                NIS = y^T S^-1 y            (normalised innovation squared)
    update:     K = P_pred S^-1
                x = x_pred + K y,   P = (I - K) P_pred
WHY velocity as a control input: an earlier project worked in degrees and
ignored the velocity reading, so its "prediction" was just the last GPS
fix. Here the prediction really comes from an independent sensor.

GATING
If NIS > gate_nis or the drone is already in ALARM, the GPS reading is NOT
used ("coasting": x = x_pred, P = P_pred).
WHY: an ungated filter slowly follows a drift and the attack is absorbed
into the estimate. Refusing suspicious readings keeps the innovation, and
therefore the evidence, visible.

COAST LIMIT
Coasting for longer than max_coast_s re-initialises the filter from GPS,
but the ALARM stays latched for the rest of the run (diagnostics "reinit").
WHY: integrating velocity alone slowly drifts, so the filter cannot coast
forever; latching stops the attacker being "forgiven" after the reset.

CUSUM (slow-drift detector)
    S_t = max(0, S_{t-1} + NIS_t - (2 + k)),   k = cusum_k = 1
    alarm when S_t > cusum_h
    change point = last step where S was 0 before the alarm
WHY: for a clean 2D innovation E[NIS] = 2, so NIS - 3 is negative on
average and S stays near 0. A small persistent error adds a little each
step. Example: NIS averaging 4.5 adds about 1.5 per step, so S crosses
h = 25 after about 17 steps = 1.7 s at dt = 0.1 s.

FLAGS
    raw_flag = (NIS > nis_threshold) OR (S > cusum_h)
    flag     = persistent(NIS > nis_threshold) OR (S > cusum_h) OR latched
CUSUM is passed through without the persistence filter because it is
already an accumulation over many steps. score = S.
"""

from __future__ import annotations

import logging

import numpy as np

from config import DetectorConfig
from core.schema import ObservedFrame
from detection.base import Detector, PersistenceFilter, StepResult, status_from_flags

logger = logging.getLogger(__name__)

# Small extra process noise (m^2) so P never collapses to exactly zero.
Q_FLOOR = 1e-4
# Smallest GPS sigma used, so R = sigma^2 I can always be inverted.
MIN_GPS_SIGMA = 1e-3
# Expected NIS of a clean 2D innovation (chi-square with 2 dof has mean 2).
CLEAN_NIS_MEAN = 2.0


def cusum_update(previous: np.ndarray, nis: np.ndarray, k: float) -> np.ndarray:
    """One CUSUM step: S_t = max(0, S_{t-1} + NIS_t - (2 + k)). NaN NIS adds 0."""
    # WHY: on the first step there is no prediction yet (NIS = NaN), so
    # nothing is added rather than subtracting the slack.
    increment = np.where(np.isnan(nis), 0.0, nis - (CLEAN_NIS_MEAN + k))
    return np.maximum(0.0, previous + increment)


class MotionDetector(Detector):
    """Detector B: per-drone Kalman filter with velocity input, NIS test and CUSUM."""

    name = "Motion consistency"
    short = "B"

    def __init__(self, config: DetectorConfig, gps_sigma: float, vel_sigma: float) -> None:
        """gps_sigma (m) and vel_sigma (m/s) are sensor datasheet values, not truth."""
        super().__init__()
        if gps_sigma < 0 or vel_sigma < 0:
            raise ValueError(f"sensor sigmas must be >= 0, got gps={gps_sigma}, vel={vel_sigma}")
        self.config = config
        self.gps_sigma = max(float(gps_sigma), MIN_GPS_SIGMA)
        self.vel_sigma = float(vel_sigma)

    def _reset_state(self, n: int) -> None:
        self._persist = PersistenceFilter(self.config.persist_m, self.config.persist_n, n)
        self.x: np.ndarray | None = None              # (N, 2) position estimates
        self.P: np.ndarray | None = None              # (N, 2, 2) covariances
        self.prev_vel: np.ndarray | None = None       # velocity flown since last step
        self.prev_t = 0.0
        self.cusum = np.zeros(n)
        self.cusum_alarm = np.zeros(n, dtype=bool)
        self.last_zero_step = np.zeros(n, dtype=int)
        self.change_points: dict[int, int] = {}
        self.coast_time = np.zeros(n)
        self.latched = np.zeros(n, dtype=bool)
        self.prev_flags = np.zeros(n, dtype=bool)

    def _initial_cov(self, count: int) -> np.ndarray:
        """P = gps_sigma^2 I for `count` drones, shape (count, 2, 2)."""
        return np.tile(np.eye(2) * self.gps_sigma**2, (count, 1, 1))

    def _kalman(self, z: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict + (gated) update for all drones. Returns (nis, innovation, gated)."""
        eye = np.eye(2)
        x_pred = self.x + self.prev_vel * dt
        q = (self.vel_sigma * dt) ** 2 + Q_FLOOR
        P_pred = self.P + q * eye
        y = z - x_pred
        S = P_pred + self.gps_sigma**2 * eye
        S_inv = np.linalg.inv(S)                                   # (N, 2, 2)
        nis = np.einsum("ni,nij,nj->n", y, S_inv, y)               # y^T S^-1 y
        gated = (nis > self.config.gate_nis) | self.prev_flags
        K = P_pred @ S_inv
        x_upd = x_pred + np.einsum("nij,nj->ni", K, y)
        P_upd = (eye - K) @ P_pred
        self.x = np.where(gated[:, None], x_pred, x_upd)
        self.P = np.where(gated[:, None, None], P_pred, P_upd)
        return nis, y, gated

    def _coast_limit(self, z: np.ndarray, gated: np.ndarray, dt: float) -> np.ndarray:
        """Track coasting time; re-initialise drones that coasted too long."""
        self.coast_time = np.where(gated, self.coast_time + dt, 0.0)
        reinit = self.coast_time > self.config.max_coast_s
        if reinit.any():
            self.x[reinit] = z[reinit]
            self.P[reinit] = self._initial_cov(int(reinit.sum()))
            self.coast_time[reinit] = 0.0
            # WHY latch: the drone was refused for max_coast_s because it
            # looked wrong; snapping back to GPS must not clear the alarm.
            self.latched |= reinit & (self.prev_flags | self.cusum_alarm)
            logger.info("Detector B re-initialised drones %s", np.flatnonzero(reinit).tolist())
        return reinit

    def _update_cusum(self, nis: np.ndarray, step: int) -> None:
        """Advance CUSUM, remember the last zero, and record change points."""
        self.cusum = cusum_update(self.cusum, nis, self.config.cusum_k)
        self.last_zero_step = np.where(self.cusum == 0.0, step, self.last_zero_step)
        alarm = self.cusum > self.config.cusum_h
        for i in np.flatnonzero(alarm & ~self.cusum_alarm):
            self.change_points[int(i)] = int(self.last_zero_step[i])
        self.cusum_alarm = alarm

    def _step(self, obs: ObservedFrame) -> StepResult:
        n = obs.n
        z = obs.gps_pos
        if self.x is None:
            self.x = z.copy()
            self.P = self._initial_cov(n)
            nis = np.full(n, np.nan)
            y = np.zeros((n, 2))
            gated = np.zeros(n, dtype=bool)
            reinit = np.zeros(n, dtype=bool)
        else:
            dt = obs.t - self.prev_t
            if dt <= 0:
                raise ValueError(f"frames must be in time order (dt = {dt})")
            nis, y, gated = self._kalman(z, dt)
            reinit = self._coast_limit(z, gated, dt)
        self._update_cusum(nis, obs.step)

        nis_raw = np.nan_to_num(nis, nan=0.0) > self.config.nis_threshold
        nis_persistent = self._persist.update(nis_raw)
        raw = nis_raw | self.cusum_alarm
        flags = nis_persistent | self.cusum_alarm | self.latched
        self.prev_vel = obs.vel_meas.copy()
        self.prev_t = obs.t
        self.prev_flags = flags
        return StepResult(
            step=obs.step,
            t=obs.t,
            flags=flags,
            raw_flags=raw,
            scores=self.cusum.copy(),
            status=status_from_flags(flags, raw, np.ones(n, dtype=bool)),
            swarm_flag=bool(flags.any()),
            # WHY median: it is high only when MOST drones disagree with
            # their own physics, the fingerprint of a swarm-wide attack.
            swarm_score=float(np.median(self.cusum)),
            diagnostics={
                "nis": nis, "cusum": self.cusum.copy(), "innovation": y,
                "coasting": gated, "reinit": reinit, "latched": self.latched.copy(),
                "change_points": dict(self.change_points),
            },
        )
