"""Scoring a detector run against the hidden truth (EVALUATION ONLY).

This module is one of the few places allowed to read truth (rule R1): it
compares a DetectionRun with the SimResult's attack labels.

Two different questions are measured separately:
    DETECTION       "did the detector notice that something is wrong?"
    IDENTIFICATION  "did it name the right drones?"
WHY separate them: a detector that alarms on the wrong drone has still
detected an attack but has failed to identify it. Mixing the two into one
accuracy number would hide that.

All numbers produced here are simulation results.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from core.schema import SimResult
from core.utils import wilson_interval
from detection.base import DetectionRun

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """How one detector did on one simulation run.

    attacked                 drones labelled attacked at any time (truth)
    onset                    first time any drone is attacked, None if no attack
    detected                 ALARM on an attacked drone at t >= onset
    detection_delay          (first such ALARM time) - onset, seconds
    false_alarm_drones       never-attacked drones that were ever in ALARM
    early_alarms             True if any ALARM happened before onset
    clean_false_alarm        no-attack run with any ALARM or swarm flag
    id_precision             |alarmed after onset AND attacked| / |alarmed after onset|
    id_recall                |alarmed after onset AND attacked| / |attacked|
    time_to_full_identification  seconds from onset until every attacked
                             drone has been in ALARM at least once
    swarm_level_detected     swarm flag raised at t >= onset
    """

    detector: str
    seed: int
    attacked: set[int] = field(default_factory=set)
    onset: float | None = None
    detected: bool = False
    detection_delay: float | None = None
    false_alarm_drones: set[int] = field(default_factory=set)
    early_alarms: bool = False
    clean_false_alarm: bool = False
    id_precision: float = float("nan")
    id_recall: float = float("nan")
    time_to_full_identification: float | None = None
    swarm_level_detected: bool = False

    @property
    def is_attack_run(self) -> bool:
        """True if the run contained an attack."""
        return self.onset is not None

    def to_row(self) -> dict[str, object]:
        """Flat dictionary for a DataFrame (sets become sorted lists)."""
        row = asdict(self)
        row["attacked"] = sorted(self.attacked)
        row["false_alarm_drones"] = sorted(self.false_alarm_drones)
        return row


def _identification(
    run: DetectionRun,
    attacked: set[int],
    onset: float,
) -> tuple[float, float, float | None]:
    """Return (id_precision, id_recall, time_to_full_identification).

    precision = |A ∩ T| / |A|  (NaN if nothing alarmed: undefined, not 0)
    recall    = |A ∩ T| / |T|
    where A = drones alarmed at t >= onset and T = attacked drones.
    """
    alarmed = run.alarm_drones(after=onset)
    hits = alarmed & attacked
    precision = len(hits) / len(alarmed) if alarmed else float("nan")
    recall = len(hits) / len(attacked) if attacked else float("nan")
    firsts = [run.first_alarm_time(i, after=onset) for i in sorted(attacked)]
    found = [f for f in firsts if f is not None]
    full = max(found) - onset if found and len(found) == len(firsts) else None
    return precision, recall, full


def compute_metrics(result: SimResult, run: DetectionRun) -> RunMetrics:
    """Score one DetectionRun against the truth stored in a SimResult."""
    if run.n_drones != result.n_drones or len(run.steps) != result.n_steps:
        raise ValueError("DetectionRun and SimResult do not describe the same run")
    attacked = result.attacked_drones()
    onset = result.attack_onset()
    times = run.times()
    flags = run.flags()
    ever = run.alarm_drones()
    metrics = RunMetrics(
        detector=run.short,
        seed=result.config.seed,
        attacked=attacked,
        onset=onset,
        false_alarm_drones=ever - attacked,
    )
    if onset is None:
        metrics.clean_false_alarm = bool(flags.any() or run.swarm_flags().any())
        return metrics

    metrics.early_alarms = bool(flags[times < onset].any())
    firsts = [run.first_alarm_time(i, after=onset) for i in sorted(attacked)]
    hit_times = [f for f in firsts if f is not None]
    metrics.detected = bool(hit_times)
    metrics.detection_delay = min(hit_times) - onset if hit_times else None
    precision, recall, full = _identification(run, attacked, onset)
    metrics.id_precision, metrics.id_recall = precision, recall
    metrics.time_to_full_identification = full
    metrics.swarm_level_detected = run.first_swarm_alarm_time(after=onset) is not None
    return metrics


def _rate_columns(prefix: str, k: int, n: int) -> dict[str, float]:
    """Rate k/n with a 95 % Wilson interval, as three named columns."""
    low, high = wilson_interval(k, n)
    return {
        f"{prefix}_rate": k / n if n else float("nan"),
        f"{prefix}_ci_low": low,
        f"{prefix}_ci_high": high,
    }


def _summarise_group(items: Sequence[RunMetrics]) -> dict[str, object]:
    """One summary row for the runs of one detector."""
    attack_runs = [m for m in items if m.is_attack_run]
    clean_runs = [m for m in items if not m.is_attack_run]
    delays = [m.detection_delay for m in attack_runs if m.detection_delay is not None]
    row: dict[str, object] = {
        "n_runs": len(items),
        "n_attack_runs": len(attack_runs),
        "n_clean_runs": len(clean_runs),
    }
    row.update(_rate_columns("detection", sum(m.detected for m in attack_runs), len(attack_runs)))
    row.update(_rate_columns(
        "swarm_detection", sum(m.swarm_level_detected for m in attack_runs), len(attack_runs)))
    row.update(_rate_columns(
        "clean_false_alarm", sum(m.clean_false_alarm for m in clean_runs), len(clean_runs)))
    row.update(_rate_columns(
        "false_alarm_drone", sum(bool(m.false_alarm_drones) for m in attack_runs), len(attack_runs)))
    row["mean_delay_s"] = float(np.mean(delays)) if delays else float("nan")
    row["median_delay_s"] = float(np.median(delays)) if delays else float("nan")
    precisions = [m.id_precision for m in attack_runs if not math.isnan(m.id_precision)]
    recalls = [m.id_recall for m in attack_runs if not math.isnan(m.id_recall)]
    row["mean_id_precision"] = float(np.mean(precisions)) if precisions else float("nan")
    row["mean_id_recall"] = float(np.mean(recalls)) if recalls else float("nan")
    return row


def aggregate(metrics: Sequence[RunMetrics]) -> pd.DataFrame:
    """Summarise many RunMetrics: one row per detector.

    Rates come with 95 % Wilson confidence intervals (see core/utils.py):
    WHY: "50 of 50 detected" is not "100 % certain"; Wilson still reports a
    sensible range (about 93-100 %). Detection rates use only attack runs;
    clean false-alarm rates use only no-attack runs. Precision is averaged
    only over runs where something alarmed (otherwise it is undefined).
    """
    if not metrics:
        raise ValueError("aggregate() needs at least one RunMetrics")
    detectors = sorted({m.detector for m in metrics})
    rows = []
    for name in detectors:
        row = {"detector": name}
        row.update(_summarise_group([m for m in metrics if m.detector == name]))
        rows.append(row)
    return pd.DataFrame(rows)
