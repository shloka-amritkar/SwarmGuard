"""Detector C: absolute anchors (GPS vs. trusted absolute position fixes).

IDEA
Some drones carry an absolute position reference that a GPS spoofer cannot
change, e.g. a camera recognising surveyed landmarks. For each drone with
an anchor reading this step:
    r_i = |gps_i - anchor_i|
    raw_flag_i = r_i > tau_c
Drones without a reading get status NO_EVIDENCE and score NaN (rule R4).
This is the only detector that measures ABSOLUTE error directly, so it sees
a rigid swarm shift; but only on anchored drones, and only if the anchors
themselves are trustworthy.

SWARM LEVEL
    swarm_score = aggregate(r_i over drones with a reading)  (median default)
    swarm_flag  = persistent(swarm_score > tau_c)
With zero anchor readings the swarm status is NO_EVIDENCE and swarm_score is
NaN (fix F8: an earlier project returned a "clean" 0 when it had nothing to
check, which reads as "verified safe").

WHY the median (worked example)
Residuals [9.8, 10.1, 10.2, 100] m, where the last anchor is faulty:
    mean   = (9.8 + 10.1 + 10.2 + 100) / 4 = 32.5 m   (32.525; dragged by one fault)
    median = (10.1 + 10.2) / 2            = 10.15 m (the real swarm offset)
Limitation: if MOST anchors are faulty, the median is wrong too. Phase 3's
RANSAC alignment addresses that case.
"""

from __future__ import annotations

import logging

import numpy as np

from config import DetectorConfig
from core.schema import ObservedFrame
from detection.base import Detector, PersistenceFilter, StepResult, status_from_flags

logger = logging.getLogger(__name__)


def anchor_residuals(obs: ObservedFrame) -> np.ndarray:
    """Return (N,) r_i = |gps_i - anchor_i|, NaN where there is no anchor reading."""
    diff = obs.gps_pos - obs.anchor_pos
    r = np.linalg.norm(np.nan_to_num(diff, nan=0.0), axis=1)
    return np.where(obs.anchor_mask, r, np.nan)


def aggregate_residuals(residuals: np.ndarray, how: str) -> float:
    """Combine the available anchor residuals with median | mean | max.

    Returns NaN when there are no readings (F8) instead of a misleading 0.
    """
    values = residuals[~np.isnan(residuals)]
    if values.size == 0:
        return float("nan")
    if how == "median":
        return float(np.median(values))
    if how == "mean":
        return float(np.mean(values))
    if how == "max":
        return float(np.max(values))
    raise ValueError(f"unknown aggregate {how!r}; use median, mean or max")


class AnchorDetector(Detector):
    """Detector C: compare GPS with absolute anchor fixes on anchored drones."""

    name = "Absolute anchors"
    short = "C"

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__()
        self.config = config

    def _reset_state(self, n: int) -> None:
        cfg = self.config
        self._persist = PersistenceFilter(cfg.persist_m, cfg.persist_n, n)
        # WHY a 1-drone filter: the swarm flag gets the same m-of-n rule.
        self._swarm_persist = PersistenceFilter(cfg.persist_m, cfg.persist_n, 1)

    def _step(self, obs: ObservedFrame) -> StepResult:
        cfg = self.config
        residuals = anchor_residuals(obs)
        has_evidence = obs.anchor_mask.copy()
        raw = has_evidence & (np.nan_to_num(residuals, nan=0.0) > cfg.tau_c)
        flags = self._persist.update(raw)
        swarm_score = aggregate_residuals(residuals, cfg.c_aggregate)
        swarm_raw = bool(not np.isnan(swarm_score) and swarm_score > cfg.tau_c)
        swarm_flag = bool(self._swarm_persist.update(np.array([swarm_raw]))[0])
        return StepResult(
            step=obs.step,
            t=obs.t,
            flags=flags,
            raw_flags=raw,
            scores=residuals,
            status=status_from_flags(flags, raw, has_evidence),
            swarm_flag=swarm_flag,
            swarm_score=swarm_score,
            diagnostics={
                "residuals": residuals,
                "n_readings": int(has_evidence.sum()),
                "swarm_status": "no_evidence" if np.isnan(swarm_score) else "evaluated",
                "swarm_raw_flag": swarm_raw,
            },
        )
