"""Incidents: the forensic record of one run in which a detector raised an alarm.

An Incident gathers, in one JSON-serialisable object:
    identity      incident_id "SG-YYYYMMDD-NNN", creation time (UTC)
    provenance    scenario, configs, seed, calibration hash, software
                  version and git commit, so the run can be repeated exactly
    findings      what each detector saw (first alarm, drones, peak score),
                  the estimated onset, fusion verdicts, agreement pattern,
                  recovery summary and a plain-language explanation
    ground truth  what the SIMULATION really did, clearly labelled as
                  "not available to detectors" (rule R1)

ONSET ESTIMATE
    If Detector B's CUSUM alarmed on a drone that some detector flagged,
        onset = earliest CUSUM change point among those drones
    (the change point is the last step where CUSUM was 0 before it alarmed,
    i.e. when the evidence started building up). Otherwise
        onset = earliest first-alarm time of any detector.
WHY prefer the CUSUM change point: an alarm comes AFTER evidence has piled
up, so it is late; the change point estimates when the anomaly began.

WHY record git commit, version and calibration hash: an investigator must
be able to say exactly which code and which thresholds produced a finding.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from analysis.calibration import calibration_path
from analysis.experiments import PipelineResult
from analysis.metrics import compute_metrics
from config import AttackConfig, DetectorConfig, SimConfig
from database.storage import Storage
from core.schema import Verdict
from detection.base import DetectionRun
from detection.fusion import FusionResult, describe_pattern, detector_state, pattern_key
from forensics.hashing import canonical_json, sha256_bytes, sha256_file

logger = logging.getLogger(__name__)

VERSION = "0.4.0"
GROUND_TRUTH_LABEL = "SIMULATION GROUND TRUTH: not available to detectors"
NO_CALIBRATION = "none (analytical defaults)"
GIT_TIMEOUT_S = 5.0

DETECTOR_NAMES: dict[str, str] = {
    "A": "relative geometry",
    "B": "motion consistency",
    "C": "absolute anchors",
}
SCORE_NAMES: dict[str, str] = {
    "A": "share of disagreeing pairs",
    "B": "CUSUM statistic",
    "C": "GPS-to-anchor distance",
}


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------
def git_commit(repo_dir: str | Path = ".") -> str:
    """Current git commit hash, or "unknown".

    Runs `git rev-parse HEAD` as an argument list (no shell, so nothing in
    the path can be interpreted as a command) with a timeout, so a missing
    or hanging git never blocks an investigation.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True,
                             text=True, timeout=GIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("git commit unavailable: %s", exc)
        return "unknown"
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else "unknown"


def calibration_hash_for(sim: SimConfig) -> str:
    """SHA-256 of the calibration file used for this swarm config, or NO_CALIBRATION."""
    path = calibration_path(sim)
    return sha256_file(path) if path.exists() else NO_CALIBRATION


def run_key(sim: SimConfig, atk: AttackConfig, det: DetectorConfig) -> str:
    """SHA-256 identifying one exact run: swarm config (incl. seed), attack and thresholds.

        run_key = sha256(canonical_json({"sim": ..., "attack": ..., "detector": ...}))

    Same key = same inputs = identical output (rule R2).
    """
    return sha256_bytes(canonical_json({"sim": sim.to_dict(), "attack": atk.to_dict(),
                                        "detector": det.to_dict()}))


def make_incident_id(day: date, number: int) -> str:
    """("2026-11-05", 7) -> "SG-20261105-007"."""
    if not (1 <= number <= 999):
        raise ValueError(f"incident number must be 1..999, got {number}")
    return f"SG-{day.strftime('%Y%m%d')}-{number:03d}"


def _f(value: float | np.floating | None, digits: int = 4) -> float | None:
    """Plain rounded float for JSON, or None for None / NaN."""
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


# --------------------------------------------------------------------------
# The incident record
# --------------------------------------------------------------------------
@dataclass
class Incident:
    """One forensic incident (see module docstring). Drone indices are 0-based."""

    incident_id: str
    created_utc: str
    scenario_key: str
    config: dict[str, Any]
    attack: dict[str, Any]
    seed: int
    calibration_hash: str
    software_version: str
    git_commit: str
    detected: dict[str, dict[str, Any]]
    affected_drones_detected: list[int]
    onset_estimate: float | None
    onset_source: str
    fusion: dict[str, Any]
    pattern: str
    pattern_description: str
    recovery: dict[str, Any]
    simulation_ground_truth: dict[str, Any]
    explanation: list[str] = field(default_factory=list)
    run_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Incident:
        """Rebuild from to_dict() output (unknown keys are rejected)."""
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Incident: unknown field(s) {unknown}")
        return cls(**data)

    def alarmed_detectors(self) -> list[str]:
        """Letters of detectors that raised an alarm, e.g. ["B", "C"]."""
        return [s for s, d in sorted(self.detected.items()) if d.get("state") == "alarm"]


def should_open_incident(runs: Sequence[DetectionRun]) -> bool:
    """True if any detector raised any ALARM (per drone or swarm level)."""
    return any(r.flags().any() or r.swarm_flags().any() for r in runs)


# --------------------------------------------------------------------------
# Building an incident from a pipeline run
# --------------------------------------------------------------------------
def _threshold(short: str, det: DetectorConfig) -> float:
    """The threshold each detector's score is compared with."""
    return {"A": det.vote_fraction, "B": det.cusum_h, "C": det.tau_c}[short]


def _peak_nis(run: DetectionRun) -> float | None:
    """Largest Detector B NIS over the run (None if never computed)."""
    values = [s.diagnostics.get("nis") for s in run.steps]
    stacked = np.array([v for v in values if v is not None], dtype=float)
    return _f(np.nanmax(stacked)) if stacked.size and np.isfinite(stacked).any() else None


def detector_summary(run: DetectionRun, det: DetectorConfig) -> dict[str, Any]:
    """What one detector saw over the whole run.

    first_alarm_t  earliest per-drone ALARM (or swarm-level flag), seconds
    drones         drones that were ever in ALARM
    peak_score     largest score over all steps and drones (None if no evidence)
    state          "alarm" | "silent" | "no_evidence" (same rule as fusion)
    """
    times = run.times()
    any_alarm = run.flags().any(axis=1) | run.swarm_flags()
    hits = np.flatnonzero(any_alarm)
    scores = run.scores()
    has_score = np.isfinite(scores).any()
    peak_drone = int(np.nanargmax(np.where(np.isfinite(scores), scores, -np.inf)) % run.n_drones) \
        if has_score else None
    out: dict[str, Any] = {
        "name": DETECTOR_NAMES.get(run.short, run.name),
        "state": detector_state(run),
        "first_alarm_t": _f(times[hits[0]], 2) if hits.size else None,
        "drones": sorted(run.alarm_drones()),
        "swarm_alarm": bool(run.swarm_flags().any()),
        "peak_score": _f(np.nanmax(scores)) if has_score else None,
        "peak_drone": peak_drone,
        "score_name": SCORE_NAMES.get(run.short, "score"),
        "threshold": _threshold(run.short, det),
    }
    if run.short == "A":
        out["tau_a_m"] = det.tau_a
    if run.short == "B":
        out["peak_nis"] = _peak_nis(run)
        out["nis_threshold"] = det.nis_threshold
    return out


def estimate_onset(runs: dict[str, DetectionRun], flagged: set[int]) -> tuple[float | None, str]:
    """(onset seconds, source) using the rule in the module docstring."""
    b = runs.get("B")
    if b is not None and flagged:
        change_points: dict[int, int] = b.steps[-1].diagnostics.get("change_points", {})
        candidate_steps = [step for drone, step in change_points.items() if drone in flagged]
        if candidate_steps:
            step_to_t = {s.step: s.t for s in b.steps}
            first = min(candidate_steps)
            return _f(step_to_t.get(first, first * (b.steps[1].t - b.steps[0].t)), 2), \
                "Detector B CUSUM change point"
    firsts = []
    for run in runs.values():
        hits = np.flatnonzero(run.flags().any(axis=1) | run.swarm_flags())
        if hits.size:
            firsts.append(float(run.times()[hits[0]]))
    if firsts:
        return _f(min(firsts), 2), "earliest detector alarm"
    return None, "no alarm"


def fusion_summary(fusion: FusionResult) -> dict[str, Any]:
    """Per drone: verdict, belief and plausibility at the step of peak belief.

    WHY the peak-belief step rather than the last step: an intermittent
    attack may have stopped by the end of the run; the peak shows the
    strongest evidence fusion ever had for that drone.
    """
    alarms = fusion.alarms()
    drones = []
    for i in range(fusion.verdicts.shape[1]):
        k = int(np.argmax(fusion.belief[:, i]))
        drones.append({
            "drone": i,
            "verdict": fusion.verdicts[k, i].value,
            "belief": _f(fusion.belief[k, i]),
            "plausibility": _f(fusion.plausibility[k, i]),
            "t": _f(fusion.times[k], 2),
            "fused_alarm": bool(alarms[:, i].any()),
            "final_verdict": fusion.verdicts[-1, i].value,
        })
    compromised = [d["drone"] for d in drones if d["fused_alarm"]]
    return {"drones": drones, "fused_alarm_drones": compromised,
            "high_conflict_steps": int(fusion.high_conflict.any(axis=1).sum()),
            "verdict_counts_final": {v.value: int(sum(x == v for x in fusion.verdicts[-1]))
                                     for v in Verdict}}


def recovery_summary(pipe: PipelineResult) -> dict[str, Any]:
    """Recovery method/status and the estimated swarm-wide attack vector (JSON-ready)."""
    if pipe.recovery is None:
        return {"method": "-", "status": "not_run", "attack_vector": None,
                "attack_magnitude": None, "outlier_anchors": [], "n_steps": 0}
    rec = dict(pipe.recovery.summary())
    vec = rec.get("attack_vector")
    rec["attack_vector"] = [_f(vec[0], 3), _f(vec[1], 3)] if vec is not None else None
    rec["attack_magnitude"] = _f(rec.get("attack_magnitude"), 3)
    rec["outlier_anchors"] = [int(i) for i in rec.get("outlier_anchors", [])]
    return rec


def ground_truth_summary(pipe: PipelineResult) -> dict[str, Any]:
    """EVALUATION ONLY: what the simulation really did, labelled as such (rule R1)."""
    result = pipe.sim_result
    return {"label": GROUND_TRUTH_LABEL, "type": result.attack.type,
            "targets": sorted(result.attacked_drones()), "onset": _f(result.attack_onset(), 2)}


def from_run(
    pipe: PipelineResult,
    scenario_key: str,
    incident_id: str,
    calibration_hash: str,
    created_utc: str | None = None,
) -> Incident:
    """Build an Incident from a finished pipeline run (explanation included)."""
    from forensics.explainer import explain_incident  # WHY local: explainer imports Incident

    sim_cfg, atk_cfg = pipe.sim_result.config, pipe.sim_result.attack
    runs = {r.short: r for r in pipe.runs}
    detected = {s: detector_summary(r, pipe.det_cfg) for s, r in sorted(runs.items())}
    flagged = set().union(*(set(d["drones"]) for d in detected.values()))
    onset, source = estimate_onset(runs, flagged)
    key = pattern_key(pipe.runs)
    incident = Incident(
        incident_id=incident_id,
        created_utc=created_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        scenario_key=scenario_key,
        config={"sim": sim_cfg.to_dict(), "detector": pipe.det_cfg.to_dict(),
                "sim_config_hash": sim_cfg.config_hash(),
                "detector_config_hash": pipe.det_cfg.config_hash()},
        attack=atk_cfg.to_dict(), seed=int(sim_cfg.seed), calibration_hash=calibration_hash,
        software_version=VERSION, git_commit=git_commit(),
        detected=detected, affected_drones_detected=sorted(int(i) for i in flagged),
        onset_estimate=onset, onset_source=source,
        fusion=fusion_summary(pipe.fusion), pattern=key, pattern_description=describe_pattern(key),
        recovery=recovery_summary(pipe), simulation_ground_truth=ground_truth_summary(pipe),
        run_key=run_key(sim_cfg, atk_cfg, pipe.det_cfg),
    )
    incident.explanation = explain_incident(incident)
    logger.info("opened incident %s (%s)", incident_id, key)
    return incident


# --------------------------------------------------------------------------
# Opening an incident (shared by the CLI and the app)
# --------------------------------------------------------------------------
def save_run(storage: Storage, scenario_key: str, pipe: PipelineResult) -> int:
    """Store the run and its per-detector metrics (evaluation view); returns the run id."""
    result = pipe.sim_result
    sim, atk = result.config, result.attack
    config = {"sim": sim.to_dict(), "attack": atk.to_dict(), "detector": pipe.det_cfg.to_dict(),
              "run_key": run_key(sim, atk, pipe.det_cfg)}
    summary = {"n_steps": result.n_steps, "n_drones": result.n_drones,
               "pattern": pattern_key(pipe.runs),
               "attack_onset_SIMULATION": result.attack_onset()}
    run_id = storage.save_run(scenario_key, sim.seed, sim.config_hash(), config, summary)
    rows = []
    for run in [*pipe.runs, pipe.fusion.as_detection_run()]:
        m = compute_metrics(result, run)
        rows.append({"detector": run.short, "detected": m.detected, "delay": m.detection_delay,
                     "false_alarms": len(m.false_alarm_drones), "id_precision": m.id_precision,
                     "id_recall": m.id_recall})
    storage.save_detection_summary(run_id, rows)
    return run_id


def open_incident(
    pipe: PipelineResult,
    scenario_key: str,
    storage: Storage,
    day: date | None = None,
) -> tuple[Incident, bool]:
    """Open (or reuse) the incident for this run. Returns (incident, newly_opened).

    If an incident with the same run_key exists it is returned unchanged
    (same inputs give identical output, R2). Otherwise the run and its
    metrics are saved, the next id "SG-YYYYMMDD-NNN" is taken and the
    incident is stored. The caller decides whether an incident is warranted
    (see should_open_incident).
    """
    sim = pipe.sim_result.config
    key = run_key(sim, pipe.sim_result.attack, pipe.det_cfg)
    existing = storage.find_incident_by_run_key(key)
    if existing is not None:
        return Incident.from_dict(existing), False
    run_id = save_run(storage, scenario_key, pipe)
    today = day or datetime.now(timezone.utc).date()
    incident_id = make_incident_id(today, storage.next_incident_number(today))
    incident = from_run(pipe, scenario_key, incident_id, calibration_hash_for(sim))
    storage.save_incident(incident.to_dict(), run_id)
    return incident, True
