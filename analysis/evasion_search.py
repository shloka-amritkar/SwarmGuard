"""Stealth envelopes: the smallest attack each detector reliably catches.

QUESTION
"Detector B detects attacks" is not a measurable claim. "Detector B
detects 90 % of single-drone drifts of at least 0.12 m/s within 10 s" is.
This module finds those numbers.

METHOD: BISECTION ON THE DETECTION RATE
For a parameter x (attack magnitude or drift rate) the detection rate
rate(x) = detected runs / n_seeds is measured on evaluation seeds. If
rate(x) >= target the threshold is at or below x, otherwise above it:
    lo, hi = range ends
    repeat until hi - lo < tol:
        mid = (lo + hi) / 2
        if rate(mid) >= target: hi = mid  else: lo = mid
    answer = hi
ASSUMPTION (stated, not proven): rate(x) rises monotonically with x.
Bigger or faster attacks are easier to see, which is true on average, but
with a finite number of seeds the measured rate can wobble, so the answer is
accurate to roughly tol plus the sampling noise of n_seeds runs.

A run counts as a detection only if the delay is at most max_delay_s (when
set): a drift caught after 40 s has already done its damage.

The upper end is checked first: if even `hi` is not detected often enough
the result is None ("not reliably detected in this range"), never a made-up
number. All numbers are simulation results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from analysis.calibration import EVALUATION_SEED_START, load_or_default
from analysis.experiments import DETECTOR_ORDER, ProgressFn, apply_param, run_tasks
from attacks.scenarios import Scenario, get_scenario
from config import DetectorConfig

logger = logging.getLogger(__name__)

# (low, high, tolerance) search range per parameter.
SEARCH_RANGES: dict[str, tuple[float, float, float]] = {
    "magnitude": (0.0, 20.0, 0.25),
    "drift_rate": (0.0, 1.0, 0.01),
}


@dataclass
class RateCache:
    """Remembers trial results per parameter value, shared by every detector.

    WHY: one trial scores A, B, C and fusion at once, so the bisections for
    different detectors and targets can reuse each other's evaluations.
    """

    scenario: Scenario
    param: str
    det: DetectorConfig
    n_seeds: int
    workers: int = 1
    seed_start: int = EVALUATION_SEED_START
    results: dict[float, list[dict]] = field(default_factory=dict)

    def trials(self, value: float) -> list[dict]:
        """All trials at this value (run once, then cached)."""
        key = round(float(value), 6)
        if key not in self.results:
            sim, atk = apply_param(self.scenario.sim, self.scenario.attack, self.param, key)
            kw = {"include_recovery": False}
            tasks = [(seed, sim, atk, self.det, seed, kw)
                     for seed in range(self.seed_start, self.seed_start + self.n_seeds)]
            self.results[key] = [trial for _, trial in run_tasks(tasks, self.workers)]
        return self.results[key]

    def rate(self, value: float, detector: str, max_delay_s: float | None) -> float:
        """Share of seeds where `detector` detected the attack (within max_delay_s)."""
        hits = 0
        for trial in self.trials(value):
            m = trial["metrics"][detector]
            if m.detected and (max_delay_s is None or m.detection_delay <= max_delay_s):
                hits += 1
        return hits / self.n_seeds


def bisect_threshold(
    rate_fn: Callable[[float], float], target: float, lo: float, hi: float, tol: float,
) -> float | None:
    """Smallest x in [lo, hi] (to within tol) with rate_fn(x) >= target.

    Returns lo if already detected there, None if not even at hi.
    Assumes rate_fn rises monotonically (see module docstring).
    """
    if not (0.0 < target <= 1.0):
        raise ValueError(f"target must be in (0, 1], got {target}")
    if not (lo < hi) or tol <= 0:
        raise ValueError(f"need lo < hi and tol > 0, got lo={lo}, hi={hi}, tol={tol}")
    if rate_fn(hi) < target:
        return None
    if rate_fn(lo) >= target:
        return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if rate_fn(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def min_detectable(
    param: str,
    detector: str,
    scenario: str,
    target_rate: float,
    lo: float | None = None,
    hi: float | None = None,
    n_seeds: int = 40,
    tol: float | None = None,
    max_delay_s: float | None = None,
    workers: int = 1,
    cache: RateCache | None = None,
) -> float | None:
    """Smallest value of `param` that `detector` catches in >= target_rate of runs."""
    if detector not in DETECTOR_ORDER:
        raise ValueError(f"detector must be one of {list(DETECTOR_ORDER)}, got {detector!r}")
    if param not in SEARCH_RANGES:
        raise ValueError(f"param must be one of {list(SEARCH_RANGES)}, got {param!r}")
    d_lo, d_hi, d_tol = SEARCH_RANGES[param]
    if cache is None:
        sc = get_scenario(scenario)
        cache = RateCache(sc, param, load_or_default(sc.sim), n_seeds, workers)
    return bisect_threshold(
        lambda x: cache.rate(x, detector, max_delay_s), target_rate,
        d_lo if lo is None else lo, d_hi if hi is None else hi, d_tol if tol is None else tol,
    )


def default_params(scenario: Scenario) -> tuple[str, ...]:
    """Which parameter describes this attack's strength.

    Drifting attacks (gradual_drift, or swarm_wide with a linear drift)
    grow over time, so their speed matters: drift_rate. Step attacks have a
    fixed size: magnitude.
    """
    atk = scenario.attack
    if atk.type == "none":
        raise ValueError(f"scenario {scenario.key!r} has no attack to evade detection with")
    drifting = atk.type == "gradual_drift" or (
        atk.type == "swarm_wide" and atk.drift_rate > 0 and atk.ramp_profile == "linear")
    return ("drift_rate",) if drifting else ("magnitude",)


def stealth_envelope(
    scenario: str,
    detectors: Sequence[str] = DETECTOR_ORDER,
    params: Sequence[str] | None = None,
    targets: Sequence[float] = (0.5, 0.9),
    n_seeds: int = 40,
    max_delay_s: float | None = 10.0,
    workers: int = 1,
    progress: ProgressFn = None,
) -> pd.DataFrame:
    """Per detector and parameter: the 50 % and 90 % detection thresholds.

    Columns: scenario, detector, param, target_rate, threshold (None = not
    reliably detected in range), max_delay_s, n_seeds.
    """
    sc = get_scenario(scenario)
    det = load_or_default(sc.sim)
    rows = []
    chosen = tuple(params) if params else default_params(sc)
    jobs = [(p, d, t) for p in chosen for d in detectors for t in targets]
    caches = {p: RateCache(sc, p, det, n_seeds, workers) for p in chosen}
    for done, (param, detector, target) in enumerate(jobs, start=1):
        value = min_detectable(param, detector, scenario, target, n_seeds=n_seeds,
                               max_delay_s=max_delay_s, cache=caches[param])
        rows.append({"scenario": scenario, "detector": detector, "param": param,
                     "target_rate": target,
                     # WHY round: bisection stops within tol; more digits would be false precision.
                     "threshold": None if value is None else round(value, 3),
                     "max_delay_s": max_delay_s, "n_seeds": n_seeds})
        if progress is not None:
            progress(done / len(jobs))
    logger.info("stealth envelope for %s used %d cached parameter values",
                scenario, sum(len(c.results) for c in caches.values()))
    return pd.DataFrame(rows)
