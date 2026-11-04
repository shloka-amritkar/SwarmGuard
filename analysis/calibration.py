"""Calibrating detector thresholds on clean simulated flights.

WHY calibrate at all
The default thresholds in config.py come from textbook statistics (3 sigma,
chi-square quantiles). Those formulas assume every test is independent.
Our real pipeline is not: persistence filters, neighbour voting, Kalman
gating and closed-loop control all correlate the tests. Calibration
therefore MEASURES the false-alarm rate of the complete pipeline on clean
flights and picks thresholds that achieve the target, which is exactly
what a claim like "fewer than 1 % of clean flights raise a false alarm"
needs.

PROCEDURE (target_fa = 0.01, n_seeds = 200 by default)
1. Simulate clean runs with seeds 1000..1199. Every odd seed also flies the
   legitimate rotate + scale manoeuvres of the manoeuvre_only scenario, so
   normal formation changes are part of "clean".
2. Detector A: try tau_a = 4.3, 4.4, ..., 8.0 through the full vote +
   persistence pipeline. Pick the SMALLEST tau_a whose per-run false-alarm
   rate is <= target (smallest = most sensitive while still acceptable).
   WHY the search starts at 4.3 m, the theoretical 3-sigma value
   (sqrt(1^2 + 1^2 + 0.3^2) = 1.45 m, x 3 = 4.3 m), and never goes below:
   clean-run false alarms alone do not constrain A. Majority voting hides
   per-pair noise, because an honest drone needs MORE THAN HALF of its pairs
   to look bad before it is flagged, so even a very low tau_a gives few
   clean false alarms. During a multi-drone attack, however, each honest
   drone already has one bad pair per liar (2 of 9 with two liars). With a
   low tau_a, ordinary noise on a few more pairs pushes it over half, and A
   wrongly blames an honest drone. (Measured: tau_a = 2.5 m flagged an
   honest drone in 8 of 20 coordinated_pair runs.) Calibration may RAISE
   tau_a above 4.3 m if clean runs need it, but never lowers it.
3. Detector B: run it with CUSUM alarms switched off (h = 1e12, effectively
   infinite) and record each run's maximum CUSUM value. Set
       cusum_h = (1 - target) quantile of those maxima x 1.05
   (5 % safety margin), then verify the complete detector. If the NIS path
   alone still gives too many false alarms, raise nis_threshold in steps of
   0.5 until it passes (this extension is reported in `notes`).
4. Detector C: try tau_c = 2.0, 2.1, ..., 8.0, same rule as A.
5. Report the achieved false-alarm rates with 95 % Wilson intervals.

SEED SEPARATION
Calibration uses seeds 1000+; evaluation experiments use 5000+. They never
overlap, because tuning and testing on the same random flights would make
the results look better than they are.

F3: thresholds are never changed silently. Results are saved to
data/calibration/<key>.json and only used when a caller explicitly loads
them with load_calibration() / load_or_default().
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attacks.scenarios import get_scenario
from config import AttackConfig, DetectorConfig, SimConfig
from core.schema import ObservedFrame
from core.utils import wilson_interval
from detection.base import persistence_over_time
from detection.detector_a_geometry import geometry_residuals, vote
from detection.detector_b_motion import MotionDetector
from detection.detector_c_anchor import aggregate_residuals, anchor_residuals
from simulation.swarm import simulate

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path("data") / "calibration"
CALIBRATION_SEED_START = 1000
EVALUATION_SEED_START = 5000
# WHY start at 4.3 m and not lower: see point 2 of the module docstring.
TAU_A_FLOOR = 4.3
TAU_A_GRID: tuple[float, ...] = tuple(round(TAU_A_FLOOR + 0.1 * i, 1) for i in range(38))  # 4.3..8.0
TAU_C_GRID: tuple[float, ...] = tuple(round(2.0 + 0.1 * i, 1) for i in range(61))   # 2.0..8.0
NIS_STEP = 0.5
NIS_MAX = 20.0
CUSUM_MARGIN = 1.05
# WHY 1e12 and not infinity: DetectorConfig rejects non-finite numbers, and
# no CUSUM value in a 60 s flight comes anywhere near 1e12.
CUSUM_OFF = 1e12

ProgressFn = Callable[[float], None] | None


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    """Everything a calibration produced, saved as JSON for later reuse.

    thresholds   the calibrated DetectorConfig
    achieved     per detector: false_alarms, runs, rate, ci_low, ci_high
    curves       false-alarm rate for every tried threshold (for plots)
    """

    thresholds: DetectorConfig
    config_key: str
    base_sim: dict[str, Any]
    seed_start: int
    seed_end: int
    target_fa: float
    include_manoeuvres: bool
    created_utc: str
    achieved: dict[str, dict[str, float]] = field(default_factory=dict)
    curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n_seeds(self) -> int:
        """Number of clean runs used."""
        return self.seed_end - self.seed_start + 1

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary (thresholds become a nested dict)."""
        data = asdict(self)
        data["thresholds"] = self.thresholds.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationResult:
        """Rebuild from to_dict() output (thresholds are re-validated)."""
        values = dict(data)
        values["thresholds"] = DetectorConfig.from_dict(values["thresholds"])
        return cls(**values)

    def table(self) -> pd.DataFrame:
        """One row per detector: threshold(s) and achieved false-alarm rate with CI."""
        t = self.thresholds
        settings = {
            "A": f"tau_a = {t.tau_a:.1f} m",
            "B": f"cusum_h = {t.cusum_h:.1f}, nis_threshold = {t.nis_threshold:.2f}",
            "C": f"tau_c = {t.tau_c:.1f} m",
        }
        rows = []
        for short in ("A", "B", "C"):
            a = self.achieved[short]
            rows.append({
                "detector": short, "thresholds": settings[short],
                "false_alarm_runs": int(a["false_alarms"]), "runs": int(a["runs"]),
                "fa_rate": a["rate"], "ci_low": a["ci_low"], "ci_high": a["ci_high"],
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Keys, seeds and clean flights
# --------------------------------------------------------------------------
def calibration_key(sim: SimConfig) -> str:
    """SHA-256 of the SimConfig without `seed` and `manoeuvres`.

    WHY drop them: calibration runs many seeds and adds its own manoeuvres,
    so neither describes the swarm being calibrated. This way the
    clean_baseline and manoeuvre_only scenarios share one calibration.
    """
    data = sim.to_dict()
    data.pop("seed")
    data.pop("manoeuvres")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def calibration_seeds(n_seeds: int) -> list[int]:
    """Seeds 1000 .. 1000 + n_seeds - 1; never reaching evaluation seeds (5000+)."""
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    if CALIBRATION_SEED_START + n_seeds > EVALUATION_SEED_START:
        raise ValueError(
            f"n_seeds={n_seeds} would overlap evaluation seeds starting at {EVALUATION_SEED_START}"
        )
    return list(range(CALIBRATION_SEED_START, CALIBRATION_SEED_START + n_seeds))


def clean_sim_for_seed(base_sim: SimConfig, seed: int, include_manoeuvres: bool) -> SimConfig:
    """Clean flight config for one seed (odd seeds fly manoeuvres if requested)."""
    events = get_scenario("manoeuvre_only").sim.manoeuvres if include_manoeuvres and seed % 2 else []
    return replace(base_sim, seed=seed, manoeuvres=list(events))


# --------------------------------------------------------------------------
# Per-run false-alarm checks for one threshold grid
# --------------------------------------------------------------------------
def fa_over_grid_a(frames: list[ObservedFrame], det: DetectorConfig) -> np.ndarray:
    """For one clean run: did Detector A alarm, for each tau_a in TAU_A_GRID? (bool array)

    Residuals are computed once; each threshold is then applied through the
    same vote + persistence steps the live detector uses.
    """
    pairs = [geometry_residuals(obs) for obs in frames]
    residual = np.stack([p[0] for p in pairs])          # (T, N, N)
    valid = np.stack([p[1] for p in pairs])
    out = np.zeros(len(TAU_A_GRID), dtype=bool)
    for g, tau in enumerate(TAU_A_GRID):
        _, raw, _, _ = vote(residual, valid, tau, det.vote_fraction)
        out[g] = persistence_over_time(raw, det.persist_m, det.persist_n).any()
    return out


def fa_over_grid_c(frames: list[ObservedFrame], det: DetectorConfig) -> np.ndarray:
    """For one clean run: did Detector C alarm (drone or swarm), for each tau_c?"""
    residual = np.stack([anchor_residuals(obs) for obs in frames])            # (T, N)
    swarm = np.array([aggregate_residuals(r, det.c_aggregate) for r in residual])
    safe_res = np.nan_to_num(residual, nan=0.0)
    has = ~np.isnan(residual)
    swarm_has = ~np.isnan(swarm)
    safe_swarm = np.nan_to_num(swarm, nan=0.0)
    out = np.zeros(len(TAU_C_GRID), dtype=bool)
    for g, tau in enumerate(TAU_C_GRID):
        drone_flags = persistence_over_time(has & (safe_res > tau), det.persist_m, det.persist_n)
        swarm_flags = persistence_over_time(swarm_has & (safe_swarm > tau), det.persist_m, det.persist_n)
        out[g] = drone_flags.any() or swarm_flags.any()
    return out


def run_motion(frames: list[ObservedFrame], det: DetectorConfig, sim: SimConfig) -> tuple[bool, float]:
    """Run Detector B once. Returns (any false alarm, maximum CUSUM value)."""
    run = MotionDetector(det, gps_sigma=sim.gps_sigma, vel_sigma=sim.vel_sigma).run(frames)
    return bool(run.flags().any()), float(np.nanmax(run.scores()))


# --------------------------------------------------------------------------
# Choosing thresholds
# --------------------------------------------------------------------------
def pick_smallest(grid: tuple[float, ...], fa_counts: np.ndarray, runs: int, target: float) -> tuple[float, bool]:
    """Smallest grid value whose FA rate <= target. Returns (value, target_met).

    If no value meets the target, the largest value is returned and
    target_met is False, so the caller can report it honestly.
    """
    ok = np.flatnonzero(fa_counts / runs <= target)
    if ok.size:
        return grid[int(ok[0])], True
    return grid[-1], False


def _achieved(false_alarms: int, runs: int) -> dict[str, float]:
    """Achieved false-alarm rate with its 95 % Wilson interval."""
    low, high = wilson_interval(false_alarms, runs)
    return {"false_alarms": float(false_alarms), "runs": float(runs),
            "rate": false_alarms / runs, "ci_low": low, "ci_high": high}


def _report(progress: ProgressFn, fraction: float) -> None:
    """Call the optional progress callback with a value in [0, 1]."""
    if progress is not None:
        progress(min(max(fraction, 0.0), 1.0))


def _first_pass(
    sims: list[SimConfig], det: DetectorConfig, progress: ProgressFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate every clean run once; collect A and C grid results and B's max CUSUM."""
    det_b_off = replace(det, cusum_h=CUSUM_OFF)
    fa_a = np.zeros((len(sims), len(TAU_A_GRID)), dtype=bool)
    fa_c = np.zeros((len(sims), len(TAU_C_GRID)), dtype=bool)
    max_cusum = np.zeros(len(sims))
    for r, sim in enumerate(sims):
        frames = simulate(sim, AttackConfig()).observed()
        fa_a[r] = fa_over_grid_a(frames, det)
        fa_c[r] = fa_over_grid_c(frames, det)
        _, max_cusum[r] = run_motion(frames, det_b_off, sim)
        _report(progress, 0.6 * (r + 1) / len(sims))
    return fa_a, fa_c, max_cusum


def _verify_b(
    sims: list[SimConfig], det: DetectorConfig, target: float, progress: ProgressFn,
) -> tuple[DetectorConfig, int, list[str]]:
    """Check the full Detector B; raise nis_threshold until it meets the target."""
    notes: list[str] = []
    nis = det.nis_threshold
    while True:
        candidate = replace(det, nis_threshold=nis)
        fas = 0
        for sim in sims:
            frames = simulate(sim, AttackConfig()).observed()
            fas += run_motion(frames, candidate, sim)[0]
        if fas / len(sims) <= target:
            break
        if nis + NIS_STEP > NIS_MAX:
            notes.append(f"Detector B did not reach the target even at nis_threshold = {nis:.2f}.")
            break
        notes.append(f"nis_threshold {nis:.2f} gave {fas}/{len(sims)} false-alarm runs; raised by {NIS_STEP}.")
        nis = round(nis + NIS_STEP, 2)
        _report(progress, 0.9)
    return candidate, fas, notes


def calibrate(
    base_sim: SimConfig,
    n_seeds: int = 200,
    include_manoeuvres: bool = True,
    target_fa: float = 0.01,
    progress: ProgressFn = None,
    base_detector: DetectorConfig | None = None,
) -> CalibrationResult:
    """Calibrate tau_a, cusum_h (and if needed nis_threshold) and tau_c.

    base_detector supplies every setting that is not calibrated (vote
    fraction, persistence m/n, gate, ...). Defaults to DetectorConfig().
    """
    if not (0.0 < target_fa < 1.0):
        raise ValueError(f"target_fa must be in (0, 1), got {target_fa}")
    det = base_detector if base_detector is not None else DetectorConfig()
    seeds = calibration_seeds(n_seeds)
    sims = [clean_sim_for_seed(base_sim, s, include_manoeuvres) for s in seeds]
    fa_a, fa_c, max_cusum = _first_pass(sims, det, progress)

    notes: list[str] = []
    tau_a, ok_a = pick_smallest(TAU_A_GRID, fa_a.sum(axis=0), n_seeds, target_fa)
    tau_c, ok_c = pick_smallest(TAU_C_GRID, fa_c.sum(axis=0), n_seeds, target_fa)
    for name, value, grid, ok in (("A", tau_a, TAU_A_GRID, ok_a), ("C", tau_c, TAU_C_GRID, ok_c)):
        if not ok:
            notes.append(f"Detector {name} did not reach the target; largest grid value used.")
        elif name == "A" and value == grid[0]:
            notes.append(f"Detector A uses its floor tau_a = {value} m (theoretical 3-sigma); "
                         "lower values are not tried because they blame honest drones in "
                         "multi-drone attacks.")
        elif value == grid[0]:
            notes.append(f"Detector {name} met the target at the lowest grid value ({value}); "
                         "a smaller threshold was not tried.")
    cusum_h = max(float(np.quantile(max_cusum, 1.0 - target_fa)) * CUSUM_MARGIN, 1.0)
    det_b, fas_b, notes_b = _verify_b(sims, replace(det, cusum_h=round(cusum_h, 2)), target_fa, progress)
    notes.extend(notes_b)

    thresholds = replace(det_b, tau_a=tau_a, tau_c=tau_c)
    result = CalibrationResult(
        thresholds=thresholds,
        config_key=calibration_key(base_sim),
        base_sim=base_sim.to_dict(),
        seed_start=seeds[0], seed_end=seeds[-1],
        target_fa=target_fa, include_manoeuvres=include_manoeuvres,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        achieved={
            "A": _achieved(int(fa_a[:, TAU_A_GRID.index(tau_a)].sum()), n_seeds),
            "B": _achieved(fas_b, n_seeds),
            "C": _achieved(int(fa_c[:, TAU_C_GRID.index(tau_c)].sum()), n_seeds),
        },
        curves={
            "A": {"threshold": list(TAU_A_GRID), "fa_rate": (fa_a.mean(axis=0)).tolist()},
            "C": {"threshold": list(TAU_C_GRID), "fa_rate": (fa_c.mean(axis=0)).tolist()},
            "B": {"max_cusum": max_cusum.tolist()},
        },
        notes=notes,
    )
    _report(progress, 1.0)
    logger.info("calibration finished: tau_a=%.1f cusum_h=%.1f tau_c=%.1f", tau_a, cusum_h, tau_c)
    return result


# --------------------------------------------------------------------------
# Saving and loading
# --------------------------------------------------------------------------
def calibration_path(sim: SimConfig, directory: Path = CALIBRATION_DIR) -> Path:
    """Where the calibration for this swarm configuration is stored."""
    return Path(directory) / f"{calibration_key(sim)}.json"


def save_calibration(result: CalibrationResult, directory: Path = CALIBRATION_DIR) -> Path:
    """Write the result as readable JSON; returns the file path."""
    path = Path(directory) / f"{result.config_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_calibration(sim: SimConfig, directory: Path = CALIBRATION_DIR) -> CalibrationResult | None:
    """Load the saved calibration for this swarm configuration, or None if absent."""
    path = calibration_path(sim, directory)
    if not path.exists():
        return None
    try:
        return CalibrationResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"calibration file {path} is invalid: {exc}") from exc


def load_or_default(sim: SimConfig, directory: Path = CALIBRATION_DIR) -> DetectorConfig:
    """Calibrated thresholds if a calibration exists, else the analytical defaults.

    Either way the choice is logged (F3: thresholds never change silently).
    """
    result = load_calibration(sim, directory)
    if result is None:
        logger.info("no calibration for this configuration; using analytical defaults")
        return DetectorConfig()
    logger.info("using calibration from %s (seeds %d-%d)", result.created_utc,
                result.seed_start, result.seed_end)
    return result.thresholds
