"""Shared building blocks for every detector.

    StepResult         what one detector says about the swarm at one step
    PersistenceFilter  "m alarms within the last n steps" noise suppression
    Detector           abstract base class: reset / update / run
    DetectionRun       a whole run of StepResults, with handy queries

Rule R1 is enforced here: Detector.update() accepts ONLY an ObservedFrame.
Passing a Frame (which carries the hidden truth) raises TypeError, so a
detector can never accidentally read true positions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.schema import DroneStatus, ObservedFrame

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# One step of output
# --------------------------------------------------------------------------
@dataclass
class StepResult:
    """One detector's output for one time step (N drones).

    flags        (N,) bool  ALARM after the persistence filter
    raw_flags    (N,) bool  the single-step test, before persistence
    scores       (N,) float detector-specific score (NaN = no evidence)
    status       list of N DroneStatus values
    swarm_flag   True if the detector raises a swarm-level alarm
    swarm_score  one number summarising the whole swarm (NaN = no evidence)
    diagnostics  detector-specific extra arrays for the UI / explainer
    """

    step: int
    t: float
    flags: np.ndarray
    raw_flags: np.ndarray
    scores: np.ndarray
    status: list[DroneStatus]
    swarm_flag: bool
    swarm_score: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def status_from_flags(
    flags: np.ndarray,
    raw_flags: np.ndarray,
    has_evidence: np.ndarray,
) -> list[DroneStatus]:
    """Turn flag arrays into one DroneStatus per drone.

    Order of the rules (first match wins):
        no usable data            -> NO_EVIDENCE   (R4: never NORMAL)
        persistent flag           -> ALARM
        single-step flag only     -> SUSPICIOUS
        otherwise                 -> NORMAL
    """
    out: list[DroneStatus] = []
    for flag, raw, ok in zip(flags, raw_flags, has_evidence):
        if not ok:
            out.append(DroneStatus.NO_EVIDENCE)
        elif flag:
            out.append(DroneStatus.ALARM)
        elif raw:
            out.append(DroneStatus.SUSPICIOUS)
        else:
            out.append(DroneStatus.NORMAL)
    return out


# --------------------------------------------------------------------------
# Persistence ("m out of n")
# --------------------------------------------------------------------------
class PersistenceFilter:
    """Flag a drone only if at least m of its last n raw flags were True.

    WHY: 600 steps x 10 drones = 6,000 tests per run. Even a 1 % chance of
    a noise spike per test would give about 60 false alarms in every clean
    run. Isolated spikes rarely repeat 3 times in 5 steps, but a real
    attack persists, so it still passes the filter.

    Example (m = 3, n = 5): raw history F T F T T -> sum 3 -> flag.
    """

    def __init__(self, m: int, n: int, n_drones: int) -> None:
        if not (1 <= m <= n):
            raise ValueError(f"need 1 <= m <= n, got m={m}, n={n}")
        if n_drones < 1:
            raise ValueError(f"n_drones must be >= 1, got {n_drones}")
        self.m = m
        self.n = n
        # WHY deque(maxlen=n): appending to a full deque drops the oldest
        # item automatically, so it always holds exactly the last n flags.
        self._history: list[deque[bool]] = [deque(maxlen=n) for _ in range(n_drones)]

    def update(self, raw_flags: np.ndarray) -> np.ndarray:
        """Add this step's raw flags and return the persistent flags (N,)."""
        raw = np.asarray(raw_flags, dtype=bool)
        if raw.shape != (len(self._history),):
            raise ValueError(f"raw_flags must have shape ({len(self._history)},), got {raw.shape}")
        out = np.zeros(len(self._history), dtype=bool)
        for i, value in enumerate(raw):
            self._history[i].append(bool(value))
            out[i] = sum(self._history[i]) >= self.m
        return out


def persistence_over_time(raw: np.ndarray, m: int, n: int) -> np.ndarray:
    """Apply the m-of-n persistence rule to a whole run at once.

    raw has shape (T, N) (or (T,)). Output[k] is True when at least m of
    raw[k-n+1 .. k] are True (fewer steps at the start of the run). This
    gives exactly the same answer as feeding the rows one by one into a
    PersistenceFilter, but without a Python loop over time.

    Rolling sum trick: with c = cumulative sum of raw along time,
        window_sum[k] = c[k] - c[k - n]    (c[-1] taken as 0)
    WHY: calibration tries ~56 thresholds on 200 runs; recomputing the
    filter step by step for each would be slow for no benefit.
    """
    if not (1 <= m <= n):
        raise ValueError(f"need 1 <= m <= n, got m={m}, n={n}")
    arr = np.asarray(raw, dtype=np.int64)
    csum = np.cumsum(arr, axis=0)
    shifted = np.zeros_like(csum)
    if arr.shape[0] > n:
        shifted[n:] = csum[:-n]
    return (csum - shifted) >= m


# --------------------------------------------------------------------------
# Abstract detector
# --------------------------------------------------------------------------
class Detector(ABC):
    """Base class for all detectors.

    Subclasses set `name` and `short` and implement _reset_state(n) and
    _step(obs). Users call reset(n), update(obs) and run(frames).

    WHY a public update() that wraps a private _step(): the type check that
    keeps truth away from detectors is written once, here, and no subclass
    can forget it.
    """

    name: str = "detector"
    short: str = "?"

    def __init__(self) -> None:
        self.n: int | None = None

    def reset(self, n: int) -> None:
        """Forget all history and prepare for a swarm of n drones."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = int(n)
        self._reset_state(self.n)

    def update(self, obs: ObservedFrame) -> StepResult:
        """Process one ObservedFrame and return this step's StepResult."""
        if not isinstance(obs, ObservedFrame):
            raise TypeError(
                f"detectors must not see truth: expected ObservedFrame, got {type(obs).__name__}"
            )
        if self.n is None:
            # WHY auto-reset: lets a caller use update() directly on the
            # first frame without remembering to call reset() first.
            self.reset(obs.n)
        elif obs.n != self.n:
            raise ValueError(f"{self.name}: frame has {obs.n} drones, detector was reset for {self.n}")
        return self._step(obs)

    def run(self, frames: Sequence[ObservedFrame]) -> DetectionRun:
        """Reset, then process every frame in order. Returns a DetectionRun."""
        if not frames:
            raise ValueError("run() needs at least one frame")
        # WHY n = None: the first update() then type-checks the frame and
        # resets the detector for that frame's swarm size.
        self.n = None
        results = [self.update(obs) for obs in frames]
        return DetectionRun(name=self.name, short=self.short, steps=results)

    @abstractmethod
    def _reset_state(self, n: int) -> None:
        """Create fresh internal state for n drones."""

    @abstractmethod
    def _step(self, obs: ObservedFrame) -> StepResult:
        """Detector-specific work for one frame (input already type-checked)."""


# --------------------------------------------------------------------------
# A whole run
# --------------------------------------------------------------------------
@dataclass
class DetectionRun:
    """All StepResults of one detector over one simulation run."""

    name: str
    short: str
    steps: list[StepResult]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("DetectionRun needs at least one step")

    @property
    def n_drones(self) -> int:
        """Number of drones N."""
        return int(self.steps[0].flags.shape[0])

    def times(self) -> np.ndarray:
        """(T,) step times in seconds."""
        return np.array([s.t for s in self.steps], dtype=float)

    def flags(self) -> np.ndarray:
        """(T, N) persistent flags (ALARM)."""
        return np.stack([s.flags for s in self.steps])

    def raw_flags(self) -> np.ndarray:
        """(T, N) single-step flags."""
        return np.stack([s.raw_flags for s in self.steps])

    def scores(self) -> np.ndarray:
        """(T, N) scores (NaN = no evidence)."""
        return np.stack([s.scores for s in self.steps])

    def swarm_flags(self) -> np.ndarray:
        """(T,) swarm-level flags."""
        return np.array([s.swarm_flag for s in self.steps], dtype=bool)

    def swarm_scores(self) -> np.ndarray:
        """(T,) swarm-level scores."""
        return np.array([s.swarm_score for s in self.steps], dtype=float)

    def statuses(self) -> np.ndarray:
        """(T, N) array of status strings, e.g. "alarm"."""
        return np.array([[st.value for st in s.status] for s in self.steps], dtype=object)

    def to_dataframe(self) -> pd.DataFrame:
        """Long table, one row per (step, drone).

        Columns: step, t, drone, flag, raw_flag, score, status, swarm_flag,
        swarm_score. Swarm columns repeat on every drone row of a step.
        """
        t_count, n = len(self.steps), self.n_drones
        steps = np.array([s.step for s in self.steps])
        return pd.DataFrame({
            "step": np.repeat(steps, n),
            "t": np.repeat(self.times(), n),
            "drone": np.tile(np.arange(n), t_count),
            "flag": self.flags().reshape(-1),
            "raw_flag": self.raw_flags().reshape(-1),
            "score": self.scores().reshape(-1),
            "status": self.statuses().reshape(-1),
            "swarm_flag": np.repeat(self.swarm_flags(), n),
            "swarm_score": np.repeat(self.swarm_scores(), n),
        })

    def first_alarm_time(self, drone: int, after: float | None = None) -> float | None:
        """Time of the first ALARM for one drone (optionally at t >= after), or None."""
        if not (0 <= drone < self.n_drones):
            raise ValueError(f"drone must be in [0, {self.n_drones}), got {drone}")
        times = self.times()
        hits = self.flags()[:, drone]
        if after is not None:
            hits = hits & (times >= after)
        idx = np.flatnonzero(hits)
        return float(times[idx[0]]) if idx.size else None

    def alarm_drones(self, after: float | None = None) -> set[int]:
        """Set of drones that were in ALARM at least once (optionally at t >= after)."""
        flags = self.flags()
        if after is not None:
            flags = flags[self.times() >= after]
        return {int(i) for i in np.flatnonzero(flags.any(axis=0))}

    def first_swarm_alarm_time(self, after: float | None = None) -> float | None:
        """Time of the first swarm-level flag (optionally at t >= after), or None."""
        times = self.times()
        hits = self.swarm_flags()
        if after is not None:
            hits = hits & (times >= after)
        idx = np.flatnonzero(hits)
        return float(times[idx[0]]) if idx.size else None
