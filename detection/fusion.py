"""Fusing detector evidence with Dempster-Shafer theory, keeping uncertainty visible.

MASSES
Each detector gives, per drone and step, three numbers that sum to 1:
    m_attack     evidence that this drone's GPS is attacked
    m_normal     evidence that it is not
    m_uncertain  evidence that could be either ("I don't know")
Unlike a single probability, "I don't know" is kept separate from "50/50".

From a raw belief b in [0, 1] and a reliability alpha:
    m_attack = alpha * b,  m_normal = alpha * (1 - b),  m_uncertain = 1 - alpha
    A: b = clip(score_a, 0, 1)                         alpha = 0.70
    B: b = max(1 - exp(-S / cusum_h), min(NIS / (3 nis_threshold), 1))
                                                       alpha = 0.80
    C: b = clip((r - tau_c / 2) / tau_c, 0, 1)         alpha = 0.85
       (alpha = 0.60 while RANSAC reports outlier anchors)
    NO_EVIDENCE -> (0, 0, 1): total ignorance, which changes nothing.

WHY discounting (alpha < 1): Dempster's rule assumes independent sources,
but A and B both read the same GPS. Holding back some mass as "unknown"
limits double counting. This is a limitation we state openly, not a proof
of independence.

DEMPSTER'S RULE (two sources)
    K = m1a m2n + m1n m2a                      (conflict)
    a = (m1a m2a + m1a m2u + m1u m2a) / (1 - K)
    n = (m1n m2n + m1n m2u + m1u m2n) / (1 - K)
    u = (m1u m2u) / (1 - K)
If K > 0.95 the sources flatly contradict each other; instead of trusting
the tiny remainder we return (0, 0, 1) and flag "high_conflict".
The rule is commutative and associative, so the order A, B, C does not matter.

VERDICT  (belief = a, plausibility = a + u)
    plausibility < 0.3     NORMAL
    belief >= 0.7          COMPROMISED
    u > 0.8                INSUFFICIENT_EVIDENCE
    otherwise              SUSPICIOUS

FUSED ALARM = COMPROMISED in at least persist_m of the last persist_n steps
(the same m-of-n rule the detectors use). WHY: A, B and C all read the
same GPS value, so one large GPS noise spike can make all three suspicious
at the same step, and fusion would call that single step COMPROMISED. This
was observed on an evaluation run (one step, one honest drone). The
per-step verdicts are kept unchanged for display; only the alarm needs
persistence, which removes such one-step coincidences.

AGREEMENT PATTERN
Which detectors alarmed is often more telling than any averaged score: A
silent while B and C alarm is the fingerprint of a coordinated swarm shift.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DetectorConfig
from core.schema import DroneStatus, Verdict
from detection.base import DetectionRun, StepResult, persistence_over_time

logger = logging.getLogger(__name__)

RELIABILITY: dict[str, float] = {"A": 0.70, "B": 0.80, "C": 0.85}
C_RELIABILITY_WITH_OUTLIERS = 0.60
HIGH_CONFLICT = 0.95
PLAUSIBILITY_NORMAL = 0.3
BELIEF_COMPROMISED = 0.7
UNCERTAIN_INSUFFICIENT = 0.8

PATTERN_DESCRIPTIONS: dict[str, str] = {
    "A_B_C_silent": "No anomaly detected by any detector.",
    "A_silent_B_C_alarm": "Coordinated swarm shift suspected: the formation stays geometrically "
                          "consistent while motion and anchor checks disagree with GPS.",
    "A_silent_B_alarm_C_no_evidence": "Swarm-wide shift suspected; no anchor readings are "
                                      "available to confirm it.",
    "A_C_silent_B_alarm": "Motion inconsistency only: possibly a slow drift on drones without "
                          "anchor coverage, or a false alarm.",
    "C_silent_A_B_alarm": "Individual drone(s) disagree with neighbours and with their own "
                          "motion; the affected drones carry no anchor.",
    "A_B_C_alarm": "All three detectors disagree with GPS on at least one drone.",
}


# --------------------------------------------------------------------------
# Masses
# --------------------------------------------------------------------------
def _discount(belief: np.ndarray, alpha: np.ndarray | float, evidence: np.ndarray) -> np.ndarray:
    """(…, 3) masses from belief and reliability; (0, 0, 1) where no evidence."""
    b = np.clip(np.nan_to_num(belief, nan=0.0), 0.0, 1.0)
    alpha = np.broadcast_to(alpha, b.shape)
    m = np.stack([alpha * b, alpha * (1.0 - b), 1.0 - alpha], axis=-1)
    return np.where(evidence[..., None], m, np.array([0.0, 0.0, 1.0]))


def raw_belief(short: str, scores: np.ndarray, nis: np.ndarray | None, cfg: DetectorConfig) -> np.ndarray:
    """Raw belief b in [0, 1] for detector A, B or C (formulas in the module docstring)."""
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
    if short == "A":
        return np.clip(s, 0.0, 1.0)
    if short == "B":
        nis_part = np.zeros_like(s) if nis is None else np.nan_to_num(nis, nan=0.0)
        return np.maximum(1.0 - np.exp(-s / cfg.cusum_h),
                          np.minimum(nis_part / (3.0 * cfg.nis_threshold), 1.0))
    if short == "C":
        return np.clip((s - cfg.tau_c / 2.0) / cfg.tau_c, 0.0, 1.0)
    raise ValueError(f"fusion knows detectors A, B, C; got {short!r}")


def masses_for_step(
    short: str, step: StepResult, cfg: DetectorConfig, c_outliers: bool = False,
) -> np.ndarray:
    """(N, 3) masses (m_attack, m_normal, m_uncertain) of one detector at one step."""
    evidence = np.array([s != DroneStatus.NO_EVIDENCE for s in step.status])
    nis = step.diagnostics.get("nis") if short == "B" else None
    belief = raw_belief(short, step.scores, nis, cfg)
    alpha = RELIABILITY[short]
    if short == "C" and c_outliers:
        alpha = C_RELIABILITY_WITH_OUTLIERS
    return _discount(belief, alpha, evidence)


# --------------------------------------------------------------------------
# Combination and verdicts
# --------------------------------------------------------------------------
def dempster(m1: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine two mass arrays of shape (…, 3). Returns (masses, high_conflict mask)."""
    a1, n1, u1 = m1[..., 0], m1[..., 1], m1[..., 2]
    a2, n2, u2 = m2[..., 0], m2[..., 1], m2[..., 2]
    k = a1 * n2 + n1 * a2
    high = k > HIGH_CONFLICT
    denom = np.where(high, 1.0, 1.0 - k)          # WHY: avoid dividing by ~0
    out = np.stack([
        (a1 * a2 + a1 * u2 + u1 * a2) / denom,
        (n1 * n2 + n1 * u2 + u1 * n2) / denom,
        (u1 * u2) / denom,
    ], axis=-1)
    out = np.where(high[..., None], np.array([0.0, 0.0, 1.0]), out)
    return out, high


def verdict_from_masses(m: np.ndarray) -> Verdict:
    """Verdict for one (a, n, u) triple (rules in the module docstring)."""
    a, _, u = float(m[0]), float(m[1]), float(m[2])
    if a + u < PLAUSIBILITY_NORMAL:
        return Verdict.NORMAL
    if a >= BELIEF_COMPROMISED:
        return Verdict.COMPROMISED
    if u > UNCERTAIN_INSUFFICIENT:
        return Verdict.INSUFFICIENT_EVIDENCE
    return Verdict.SUSPICIOUS


_VERDICT_TO_STATUS = {
    Verdict.NORMAL: DroneStatus.NORMAL,
    Verdict.SUSPICIOUS: DroneStatus.SUSPICIOUS,
    Verdict.COMPROMISED: DroneStatus.ALARM,
    Verdict.INSUFFICIENT_EVIDENCE: DroneStatus.NO_EVIDENCE,
}


@dataclass
class FusionResult:
    """Fused evidence for a whole run.

    masses        (T, N, 3) combined (m_attack, m_normal, m_uncertain)
    verdicts      (T, N) Verdict values
    high_conflict (T, N) True where some combination hit K > 0.95
    per_detector  {short: (T, N, 3)} masses before combination
    persist_m/n   m-of-n rule turning COMPROMISED verdicts into a fused alarm
    """

    steps: np.ndarray
    times: np.ndarray
    masses: np.ndarray
    verdicts: np.ndarray
    high_conflict: np.ndarray
    per_detector: dict[str, np.ndarray]
    persist_m: int = 3
    persist_n: int = 5

    def compromised(self) -> np.ndarray:
        """(T, N) True where the single-step verdict is COMPROMISED."""
        return np.array([[v == Verdict.COMPROMISED for v in row] for row in self.verdicts])

    def alarms(self) -> np.ndarray:
        """(T, N) fused alarm: COMPROMISED in >= persist_m of the last persist_n steps."""
        return persistence_over_time(self.compromised(), self.persist_m, self.persist_n)

    @property
    def belief(self) -> np.ndarray:
        """(T, N) belief = m_attack."""
        return self.masses[..., 0]

    @property
    def plausibility(self) -> np.ndarray:
        """(T, N) plausibility = m_attack + m_uncertain."""
        return self.masses[..., 0] + self.masses[..., 2]

    def to_dataframe(self) -> pd.DataFrame:
        """Long table, one row per (step, drone)."""
        t_count, n = self.verdicts.shape
        return pd.DataFrame({
            "step": np.repeat(self.steps, n),
            "t": np.repeat(self.times, n),
            "drone": np.tile(np.arange(n), t_count),
            "m_attack": self.masses[..., 0].reshape(-1),
            "m_normal": self.masses[..., 1].reshape(-1),
            "m_uncertain": self.masses[..., 2].reshape(-1),
            "belief": self.belief.reshape(-1),
            "plausibility": self.plausibility.reshape(-1),
            "verdict": [v.value for v in self.verdicts.reshape(-1)],
            "high_conflict": self.high_conflict.reshape(-1),
            "fused_alarm": self.alarms().reshape(-1),
        })

    def as_detection_run(self) -> DetectionRun:
        """View the fused verdicts as a DetectionRun "F" so Phase 2 metrics apply.

        flag = fused alarm (persistent COMPROMISED), raw flag = COMPROMISED
        this step, score = belief. Status follows the verdict, except that a
        COMPROMISED step without persistence is shown as SUSPICIOUS.
        """
        raw_all = self.compromised()
        flags_all = self.alarms()
        steps = []
        for k in range(len(self.steps)):
            status = [_VERDICT_TO_STATUS[v] for v in self.verdicts[k]]
            status = [DroneStatus.SUSPICIOUS if st == DroneStatus.ALARM and not flags_all[k, i] else st
                      for i, st in enumerate(status)]
            steps.append(StepResult(
                step=int(self.steps[k]), t=float(self.times[k]), flags=flags_all[k],
                raw_flags=raw_all[k], scores=self.belief[k].copy(), status=status,
                swarm_flag=bool(flags_all[k].any()), swarm_score=float(self.belief[k].max()),
            ))
        return DetectionRun(name="Fusion", short="F", steps=steps)


def _runs_by_short(runs: Sequence[DetectionRun]) -> dict[str, DetectionRun]:
    """Index runs by detector letter, checking they cover the same steps."""
    by_short = {r.short: r for r in runs}
    unknown = sorted(set(by_short) - set(RELIABILITY))
    if unknown:
        raise ValueError(f"fusion knows detectors A, B, C; got {unknown}")
    if not by_short:
        raise ValueError("fuse() needs at least one DetectionRun")
    lengths = {len(r.steps) for r in by_short.values()}
    if len(lengths) != 1:
        raise ValueError(f"detection runs have different lengths: {sorted(lengths)}")
    return by_short


def fuse(
    runs: Sequence[DetectionRun],
    det_cfg: DetectorConfig,
    c_outliers: np.ndarray | None = None,
) -> FusionResult:
    """Fuse detectors A, B, C step by step.

    c_outliers: optional (T,) bool, True while recovery's RANSAC reports
    outlier anchors (lowers C's reliability). Without it C uses 0.85.
    """
    by_short = _runs_by_short(runs)
    first = next(iter(by_short.values()))
    t_count, n = len(first.steps), first.n_drones
    outliers = np.zeros(t_count, dtype=bool) if c_outliers is None else np.asarray(c_outliers, bool)
    if outliers.shape != (t_count,):
        raise ValueError(f"c_outliers must have shape ({t_count},), got {outliers.shape}")
    per_detector = {
        short: np.stack([masses_for_step(short, st, det_cfg, bool(outliers[k]))
                         for k, st in enumerate(run.steps)])
        for short, run in sorted(by_short.items())
    }
    combined = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (t_count, n, 3)).copy()
    high = np.zeros((t_count, n), dtype=bool)
    for masses in per_detector.values():
        combined, conflict = dempster(combined, masses)
        high |= conflict
    verdicts = np.array([[verdict_from_masses(m) for m in row] for row in combined], dtype=object)
    return FusionResult(
        steps=np.array([s.step for s in first.steps]), times=first.times(),
        masses=combined, verdicts=verdicts, high_conflict=high, per_detector=per_detector,
        persist_m=det_cfg.persist_m, persist_n=det_cfg.persist_n,
    )


# --------------------------------------------------------------------------
# Agreement matrix and pattern
# --------------------------------------------------------------------------
def _slice(run: DetectionRun, step_range: tuple[int, int] | None) -> list[StepResult]:
    """The StepResults inside [start, stop) (all of them if step_range is None)."""
    if step_range is None:
        return run.steps
    start, stop = step_range
    return [s for s in run.steps if start <= s.step < stop]


def detector_state(run: DetectionRun, step_range: tuple[int, int] | None = None) -> str:
    """Swarm-level state of one detector: "alarm", "silent" or "no_evidence".

    alarm:       any per-drone ALARM or swarm flag in the range
    no_evidence: every status in the range was NO_EVIDENCE
    """
    steps = _slice(run, step_range)
    if any(s.flags.any() or s.swarm_flag for s in steps):
        return "alarm"
    if steps and all(st == DroneStatus.NO_EVIDENCE for s in steps for st in s.status):
        return "no_evidence"
    return "silent"


def agreement_matrix(
    runs: Sequence[DetectionRun], step_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Drones x detectors table of "alarm" / "silent" / "no_evidence"."""
    by_short = _runs_by_short(runs)
    n = next(iter(by_short.values())).n_drones
    table: dict[str, list[str]] = {"drone": list(range(n))}
    for short, run in sorted(by_short.items()):
        steps = _slice(run, step_range)
        column = []
        for i in range(n):
            if any(s.flags[i] for s in steps):
                column.append("alarm")
            elif steps and all(s.status[i] == DroneStatus.NO_EVIDENCE for s in steps):
                column.append("no_evidence")
            else:
                column.append("silent")
        table[short] = column
    return pd.DataFrame(table)


def pattern_key(runs: Sequence[DetectionRun], step_range: tuple[int, int] | None = None) -> str:
    """Name the swarm-level agreement pattern, e.g. "A_silent_B_C_alarm".

    Groups in the order silent, alarm, no_evidence; letters sorted inside a
    group; empty groups left out. Example: all quiet -> "A_B_C_silent".
    """
    by_short = _runs_by_short(runs)
    states = {short: detector_state(run, step_range) for short, run in sorted(by_short.items())}
    parts = []
    for state in ("silent", "alarm", "no_evidence"):
        letters = [s for s, st in states.items() if st == state]
        if letters:
            parts.append("_".join(letters) + "_" + state)
    return "_".join(parts)


def describe_pattern(key: str) -> str:
    """Human sentence for a pattern key (generic text for unlisted patterns)."""
    if key in PATTERN_DESCRIPTIONS:
        return PATTERN_DESCRIPTIONS[key]
    return f"Detector agreement pattern {key}; no standard interpretation."
