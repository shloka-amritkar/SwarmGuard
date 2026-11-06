"""The full pipeline, single trials, and parameter sweeps over many seeds.

    run_pipeline  simulate -> detect (A, B, C) -> recover -> fuse -> repair
    run_trial     one seed of one configuration, scored against the truth
    sweep         run_trial for every (parameter value, seed) pair
    summarise     detection rates with Wilson intervals per value and detector
    evaluate_scenarios  every preset scenario over evaluation seeds (the
                  README results tables, cli.py evaluate)

WHY sweeps: one run is an anecdote. "Detector A caught a 10 m hijack" says
little; "A detects 100 % (95 % CI 83-100 %) of hijacks >= 4 m over 20
seeds" is a measurable claim. Sweeps also show WHERE a detector stops
working, for example where majority voting collapses as more drones lie.

Seeds: evaluation seeds start at 5000 and never overlap the calibration
seeds (1000+). Thresholds come from the BASE scenario's calibration and are
held fixed across sweep values: the sweep measures how fixed thresholds
cope when conditions change, just as a deployed system would.
All numbers produced here are simulation results.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from analysis.calibration import EVALUATION_SEED_START, load_calibration, load_or_default
from analysis.metrics import RunMetrics, compute_metrics
from attacks.scenarios import get_scenario, list_scenarios
from config import MULTI_TARGET_ATTACKS, AttackConfig, DetectorConfig, SimConfig
from core.schema import SimResult, Verdict
from core.utils import wilson_interval
from detection import build_detectors
from detection.base import DetectionRun
from detection.fusion import FusionResult, fuse
from recovery.mds_alignment import RecoveryRun, RecoveryStatus, recover_run
from recovery.multilateration import RepairResult, repair_all
from simulation.swarm import simulate

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float], None] | None
DETECTOR_ORDER = ("A", "B", "C", "F")


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Everything one run of the platform produces."""

    sim_result: SimResult
    det_cfg: DetectorConfig
    runs: list[DetectionRun]
    recovery: RecoveryRun | None
    fusion: FusionResult

    def flagged_at(self, k: int) -> np.ndarray:
        """(N,) drones the fusion marks COMPROMISED or SUSPICIOUS at step k."""
        return np.array([v in (Verdict.COMPROMISED, Verdict.SUSPICIOUS)
                         for v in self.fusion.verdicts[k]])

    def repairs_at(self, k: int = -1) -> dict[int, RepairResult]:
        """Multilateration for the drones flagged at step k (default: last step)."""
        obs = self.sim_result.frames[k].observed
        return repair_all(obs, self.flagged_at(k))

    def all_runs(self) -> list[DetectionRun]:
        """Detector runs A, B, C plus the fused run F."""
        return [*self.runs, self.fusion.as_detection_run()]


def run_pipeline(
    sim: SimConfig,
    atk: AttackConfig,
    det: DetectorConfig,
    recovery_every: int = 1,
    include_recovery: bool = True,
) -> PipelineResult:
    """simulate -> detect -> recover -> fuse.

    WHY recover before fusing: fusion lowers Detector C's reliability while
    RANSAC reports outlier anchors, so it needs the recovery result first.
    Detectors and recovery receive only ObservedFrames (rule R1).
    """
    result = simulate(sim, atk)
    observed = result.observed()
    runs = [d.run(observed) for d in build_detectors(det, sim)]
    recovery = recover_run(observed, every=recovery_every) if include_recovery else None
    outliers = recovery.outlier_flags() if recovery is not None else None
    return PipelineResult(result, det, runs, recovery, fuse(runs, det, outliers))


# --------------------------------------------------------------------------
# One trial
# --------------------------------------------------------------------------
def _recovery_errors(result: SimResult, recovery: RecoveryRun) -> tuple[float, float]:
    """EVALUATION: median over recovered steps (after onset) of mean position error.

    Returns (recovery_error_m, gps_error_m); the second is the same statistic
    for raw GPS, so the two can be compared directly.
    """
    onset = result.attack_onset()
    rec, gps = [], []
    for frame, r in zip(result.frames, recovery.results):
        if r is None or r.status != RecoveryStatus.OK or (onset is not None and frame.t < onset):
            continue
        rec.append(np.linalg.norm(r.positions - frame.true_pos, axis=1).mean())
        gps.append(np.linalg.norm(frame.observed.gps_pos - frame.true_pos, axis=1).mean())
    if not rec:
        return float("nan"), float("nan")
    return float(np.median(rec)), float(np.median(gps))


def run_trial(
    sim: SimConfig,
    atk: AttackConfig,
    det: DetectorConfig,
    seed: int,
    recovery_every: int = 10,
    include_recovery: bool = True,
) -> dict[str, Any]:
    """Run one seed and score A, B, C and fusion F. Pure, so safe in a process pool.

    Returns {"seed", "metrics": {short: RunMetrics}, "recovery_error_m", "gps_error_m"}.
    """
    sim_s = replace(sim, seed=int(seed))
    atk_s = replace(atk, seed=int(seed))
    pipe = run_pipeline(sim_s, atk_s, det, recovery_every, include_recovery)
    metrics = {run.short: compute_metrics(pipe.sim_result, run) for run in pipe.all_runs()}
    rec_err, gps_err = (_recovery_errors(pipe.sim_result, pipe.recovery)
                        if pipe.recovery is not None else (float("nan"), float("nan")))
    return {"seed": int(seed), "metrics": metrics, "recovery_error_m": rec_err,
            "gps_error_m": gps_err}


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------
SWEEP_PARAMS: tuple[str, ...] = ("magnitude", "drift_rate", "n_targets", "anchor_ratio",
                                 "n_drones", "gps_sigma")


def apply_param(sim: SimConfig, atk: AttackConfig, param: str, value: float) -> tuple[SimConfig, AttackConfig]:
    """Return fresh, re-validated configs with one parameter changed.

    magnitude -> attack.magnitude_m      drift_rate -> attack.drift_rate
    n_targets -> attack targets 0..k-1   (multi-target attacks only)
    anchor_ratio, n_drones, gps_sigma -> the SimConfig field of that name
    """
    if param == "magnitude":
        return replace(sim), replace(atk, magnitude_m=float(value))
    if param == "drift_rate":
        return replace(sim), replace(atk, drift_rate=float(value))
    if param == "n_targets":
        if atk.type not in MULTI_TARGET_ATTACKS:
            raise ValueError(f"n_targets needs a multi-target attack {MULTI_TARGET_ATTACKS}, got {atk.type!r}")
        k = int(value)
        if not (1 <= k <= sim.n_drones):
            raise ValueError(f"n_targets must be in [1, {sim.n_drones}], got {k}")
        return replace(sim), replace(atk, targets=list(range(k)))
    if param == "n_drones":
        return replace(sim, n_drones=int(value)), replace(atk)
    if param in ("anchor_ratio", "gps_sigma"):
        return replace(sim, **{param: float(value)}), replace(atk)
    raise ValueError(f"unknown sweep parameter {param!r}; choose from {list(SWEEP_PARAMS)}")


def _rows(param: str, value: float, trial: dict[str, Any]) -> list[dict[str, Any]]:
    """Tidy rows (one per detector) from one trial."""
    rows = []
    for short in DETECTOR_ORDER:
        m: RunMetrics = trial["metrics"][short]
        rows.append({
            "param": param, "value": value, "detector": short, "seed": trial["seed"],
            "detected": m.detected, "delay": m.detection_delay,
            "false_alarm": bool(m.false_alarm_drones) or m.clean_false_alarm or m.early_alarms,
            "id_precision": m.id_precision, "id_recall": m.id_recall,
            "recovery_error_m": trial["recovery_error_m"],
        })
    return rows


def run_tasks(
    tasks: Sequence[tuple[Any, SimConfig, AttackConfig, DetectorConfig, int, dict[str, Any]]],
    workers: int = 1,
    progress: ProgressFn = None,
) -> list[tuple[Any, dict[str, Any]]]:
    """Run (key, sim, atk, det, seed, kwargs) tasks; returns (key, trial) in task order.

    WHY a process pool for workers > 1: trials are independent and
    CPU-bound, so separate processes use several cores at once. Results are
    put back into task order so the output never depends on timing (R2).
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    out: list[tuple[Any, dict[str, Any]] | None] = [None] * len(tasks)
    if workers == 1:
        for i, (key, sim, atk, det, seed, kw) in enumerate(tasks):
            out[i] = (key, run_trial(sim, atk, det, seed, **kw))
            if progress is not None:
                progress((i + 1) / len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_trial, sim, atk, det, seed, **kw): (i, key)
                       for i, (key, sim, atk, det, seed, kw) in enumerate(tasks)}
            for done, fut in enumerate(as_completed(futures), start=1):
                i, key = futures[fut]
                out[i] = (key, fut.result())
                if progress is not None:
                    progress(done / len(tasks))
    return [item for item in out if item is not None]


def sweep(
    param: str,
    values: Sequence[float],
    base_scenario: str,
    n_seeds: int,
    workers: int = 1,
    progress: ProgressFn = None,
    seed_start: int = EVALUATION_SEED_START,
    det: DetectorConfig | None = None,
    include_recovery: bool = True,
) -> pd.DataFrame:
    """Run every (value, seed) pair and return a tidy DataFrame (one row per detector).

    Columns: param, value, detector, seed, detected, delay, false_alarm,
    id_precision, id_recall, recovery_error_m.
    """
    if n_seeds < 1 or not values:
        raise ValueError("sweep needs n_seeds >= 1 and at least one value")
    if seed_start < EVALUATION_SEED_START:
        raise ValueError(f"sweeps use evaluation seeds >= {EVALUATION_SEED_START}, got {seed_start}")
    scenario = get_scenario(base_scenario)
    det_cfg = det if det is not None else load_or_default(scenario.sim)
    kw = {"include_recovery": include_recovery}
    tasks = []
    for value in values:
        sim_v, atk_v = apply_param(scenario.sim, scenario.attack, param, value)
        tasks += [(value, sim_v, atk_v, det_cfg, seed, kw)
                  for seed in range(seed_start, seed_start + n_seeds)]
    rows = []
    for value, trial in run_tasks(tasks, workers, progress):
        rows += _rows(param, value, trial)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (param, value, detector): detection and false-alarm rates with Wilson CIs.

    Also mean/median delay (detected runs only), mean id precision/recall
    (NaN values skipped) and mean recovery error.
    """
    rows = []
    for (param, value, detector), g in df.groupby(["param", "value", "detector"], sort=True):
        n = len(g)
        k_det, k_fa = int(g["detected"].sum()), int(g["false_alarm"].sum())
        det_lo, det_hi = wilson_interval(k_det, n)
        fa_lo, fa_hi = wilson_interval(k_fa, n)
        delays = g["delay"].dropna().astype(float)
        rows.append({
            "param": param, "value": value, "detector": detector, "n": n,
            "detection_rate": k_det / n, "detection_ci_low": det_lo, "detection_ci_high": det_hi,
            "false_alarm_rate": k_fa / n, "false_alarm_ci_low": fa_lo, "false_alarm_ci_high": fa_hi,
            "mean_delay_s": float(delays.mean()) if len(delays) else float("nan"),
            "median_delay_s": float(delays.median()) if len(delays) else float("nan"),
            "mean_id_precision": float(g["id_precision"].mean()),
            "mean_id_recall": float(g["id_recall"].mean()),
            "mean_recovery_error_m": float(g["recovery_error_m"].mean()),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class SweepSpec:
    """A ready-made sweep: which parameter, which values, which base scenario."""

    param: str
    values: tuple[float, ...]
    scenario: str
    description: str


def _n_targets_values() -> tuple[float, ...]:
    """1 .. floor(N/2) + 1 for the default 10-drone swarm (where voting collapses)."""
    n = SimConfig().n_drones
    return tuple(float(k) for k in range(1, math.floor(n / 2) + 2))


BUILTIN_SWEEPS: dict[str, SweepSpec] = {
    "magnitude_single": SweepSpec("magnitude", (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20), "single_hijack",
                                  "Constant offset on one drone, 1-20 m"),
    "magnitude_swarm": SweepSpec("magnitude", (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20), "rigid_swarm_shift",
                                 "Swarm-wide rigid shift, 1-20 m"),
    "drift_single": SweepSpec("drift_rate", (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0), "slow_drift",
                              "Single-drone drift, 0.05-1.0 m/s"),
    "drift_swarm": SweepSpec("drift_rate", (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0),
                             "rigid_swarm_drift", "Swarm-wide drift, 0.05-1.0 m/s"),
    "n_targets": SweepSpec("n_targets", _n_targets_values(), "coordinated_pair",
                           "Number of spoofed drones, 1 to floor(N/2)+1"),
    "anchor_ratio": SweepSpec("anchor_ratio", (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), "rigid_swarm_shift",
                              "Share of anchored drones under a swarm shift, 0-0.6"),
    "n_drones": SweepSpec("n_drones", (5, 8, 10, 15, 20, 25, 30), "single_hijack",
                          "Swarm size 5-30 with one hijacked drone"),
    "gps_sigma": SweepSpec("gps_sigma", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0), "single_hijack",
                           "GPS noise 0.5-3.0 m with one hijacked drone"),
}


def builtin_values(param: str, scenario: str) -> tuple[float, ...]:
    """Default values for (param, scenario): an exact built-in match, else any with that param."""
    matches = [s for s in BUILTIN_SWEEPS.values() if s.param == param]
    exact = [s for s in matches if s.scenario == scenario]
    if exact or matches:
        return (exact or matches)[0].values
    raise ValueError(f"no built-in values for {param!r}; pass values explicitly")


# --------------------------------------------------------------------------
# Scenario evaluation (the README results tables)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalCase:
    """One row of the README results tables.

    scenario       preset scenario key
    remove_attack  True: fly the scenario's swarm with its attack switched
                   off, to measure false alarms of THAT swarm configuration
    """

    scenario: str
    remove_attack: bool = False

    @property
    def label(self) -> str:
        """Row label, e.g. "single_hijack" or "faulty_anchor_shift swarm, attack removed"."""
        return f"{self.scenario} swarm, attack removed" if self.remove_attack else self.scenario


def default_eval_cases() -> list[EvalCase]:
    """Every preset scenario in display order, plus faulty_anchor_shift with its attack removed.

    WHY the extra case: faulty_anchor_shift has faulty anchors built into its
    swarm, so Detector C alarms there even without an attack. Measuring that
    swarm clean shows how much of C's "detection" is really the anchors.
    """
    return [EvalCase(s.key) for s in list_scenarios()] + [EvalCase("faulty_anchor_shift", True)]


def case_kind(case: EvalCase) -> str:
    """"false_alarm" for runs without an attack, else "detection"."""
    if case.remove_attack or get_scenario(case.scenario).attack.type == "none":
        return "false_alarm"
    return "detection"


def _case_tasks(
    case: EvalCase, seeds: range, require_calibration: bool
) -> list[tuple[Any, SimConfig, AttackConfig, DetectorConfig, int, dict[str, Any]]]:
    """run_tasks entries for one case: same thresholds for every seed."""
    scenario = get_scenario(case.scenario)
    if require_calibration and load_calibration(scenario.sim) is None:
        raise ValueError(f"{case.scenario} has no calibration; run "
                         f"'cli.py calibrate --scenario {case.scenario}' first")
    det = load_or_default(scenario.sim)
    atk = replace(scenario.attack, type="none") if case.remove_attack else scenario.attack
    return [(case, scenario.sim, atk, det, s, {"include_recovery": True}) for s in seeds]


def evaluate_scenarios(
    cases: Sequence[EvalCase],
    n_seeds: int,
    workers: int = 1,
    progress: ProgressFn = None,
    seed_start: int = EVALUATION_SEED_START,
    require_calibration: bool = True,
) -> pd.DataFrame:
    """Run every case on seeds seed_start .. seed_start + n_seeds - 1.

    Returns a tidy DataFrame, one row per (case, seed, detector) with columns
    case, scenario, kind, seed, detector, hit, delay. "hit" means
      kind "detection":   an ALARM on an attacked drone at or after onset
      kind "false_alarm": any ALARM or swarm flag on a run with no attack
    WHY require_calibration defaults to True: the README numbers must use
    calibrated thresholds; silently falling back to defaults would change
    them without anyone noticing (fix F3).
    """
    if n_seeds < 1 or not cases:
        raise ValueError("evaluation needs n_seeds >= 1 and at least one case")
    if seed_start < EVALUATION_SEED_START:
        raise ValueError(f"evaluation uses seeds >= {EVALUATION_SEED_START}, got {seed_start}")
    seeds = range(seed_start, seed_start + n_seeds)
    tasks = [t for case in cases for t in _case_tasks(case, seeds, require_calibration)]
    rows = []
    for case, trial in run_tasks(tasks, workers, progress):
        kind = case_kind(case)
        for short in DETECTOR_ORDER:
            m: RunMetrics = trial["metrics"][short]
            rows.append({"case": case.label, "scenario": case.scenario, "kind": kind,
                         "seed": trial["seed"], "detector": short,
                         "hit": m.clean_false_alarm if kind == "false_alarm" else m.detected,
                         "delay": m.detection_delay})
    return pd.DataFrame(rows)


def evaluation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (case, detector): k hits of n runs, rate and 95 % Wilson interval. Keeps case order."""
    rows = []
    for (label, detector), g in df.groupby(["case", "detector"], sort=False):
        k, n = int(g["hit"].sum()), len(g)
        low, high = wilson_interval(k, n)
        rows.append({"case": label, "kind": g["kind"].iloc[0], "detector": detector,
                     "seed_first": int(g["seed"].min()), "seed_last": int(g["seed"].max()),
                     "k": k, "n": n, "rate": k / n, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def format_rate(k: int, n: int) -> str:
    """"14/50 = 28.0 % (17.5-41.7 %)": exact count, rate and 95 % Wilson interval."""
    low, high = wilson_interval(k, n)
    return f"{k}/{n} = {100 * k / n:.1f} % ({100 * low:.1f}-{100 * high:.1f} %)"


def markdown_rows(table: pd.DataFrame, kind: str) -> list[str]:
    """README table rows (Scenario | Seeds | A | B | C | Fused F) for one kind."""
    lines = []
    part = table[table["kind"] == kind]
    for label, g in part.groupby("case", sort=False):
        by_det = {r.detector: r for r in g.itertuples()}
        first = g.iloc[0]
        seeds = f"{first['n']} ({first['seed_first']}-{first['seed_last']})"
        cells = " | ".join(format_rate(by_det[d].k, by_det[d].n) for d in DETECTOR_ORDER)
        lines.append(f"| {label} | {seeds} | {cells} |")
    return lines
