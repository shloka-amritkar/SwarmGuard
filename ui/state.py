"""Session state, cached pipeline runs, and data shaping for the views.

SESSION KEYS (docs/phases/phase-5.md section 5.11)
    scenario_key, sim_cfg, atk_cfg, det_cfg, seed, pipeline_key,
    sim_result, detection_runs, fusion, recovery, incident_id
Every view reads the CURRENT run from these keys, so the Dashboard,
Detection, Recovery and Evidence pages all talk about the same flight.

CACHING
run_configs() calls _cached_pipeline(), wrapped in st.cache_data and keyed
on the run key = sha256 of the canonical sim + attack + detector configs
(the sim config includes the seed). WHY that key: same inputs give the same
output (rule R2), so a repeat request can safely reuse the stored result.
Opening an incident is a side effect (it writes to the database), so it is
done OUTSIDE the cached function; open_incident() itself reuses the existing
incident for a run key, so revisiting a run never duplicates incidents.

PATHS can be redirected with environment variables, which the tests use to
start every view on a fresh, empty database:
    SWARMGUARD_DB, SWARMGUARD_KEYS, SWARMGUARD_PACKAGES

The functions below the "data shaping" line are pure (no Streamlit), so
they are unit-tested directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from analysis.calibration import load_or_default
from analysis.experiments import PipelineResult, run_pipeline
from attacks.scenarios import Scenario, get_scenario
from config import AttackConfig, DetectorConfig, SimConfig
from core.schema import Verdict
from core.utils import drone_label
from database.storage import DEFAULT_DB_PATH, Storage
from detection.base import StepResult
from detection.fusion import describe_pattern, pattern_key
from forensics.explainer import explain_drone
from forensics.incident import open_incident, run_key, should_open_incident
from ui.theme import DETECTOR_NAMES

logger = logging.getLogger(__name__)

RECOVERY_EVERY = 1   # the CLI uses 1 too, so app and CLI incidents agree

KEYS = ("scenario_key", "sim_cfg", "atk_cfg", "det_cfg", "seed", "pipeline_key",
        "sim_result", "detection_runs", "fusion", "recovery", "incident_id", "pipeline")


# --------------------------------------------------------------------------
# Paths and storage
# --------------------------------------------------------------------------
def db_path() -> Path:
    """SQLite database path (SWARMGUARD_DB overrides data/swarmguard.db)."""
    return Path(os.environ.get("SWARMGUARD_DB", str(DEFAULT_DB_PATH)))


def keys_dir() -> Path:
    """Folder of the Ed25519 key pair (SWARMGUARD_KEYS overrides keys/)."""
    return Path(os.environ.get("SWARMGUARD_KEYS", "keys"))


def packages_dir() -> Path:
    """Folder for evidence packages (SWARMGUARD_PACKAGES overrides data/packages)."""
    return Path(os.environ.get("SWARMGUARD_PACKAGES", str(Path("data") / "packages")))


@st.cache_resource(show_spinner=False)
def _storage_for(path: str) -> Storage:
    """One Storage object per database path, shared across reruns."""
    return Storage(path)


def get_storage() -> Storage:
    """The app's Storage (creates the schema on first use)."""
    return _storage_for(str(db_path()))


# --------------------------------------------------------------------------
# Running the pipeline
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=12)
def _cached_pipeline(key: str, sim_json: str, atk_json: str, det_json: str) -> PipelineResult:
    """simulate -> detect -> recover -> fuse. `key` is the run key (it names the cache entry)."""
    logger.info("running pipeline %s", key[:12])
    return run_pipeline(SimConfig.from_json(sim_json), AttackConfig.from_json(atk_json),
                        DetectorConfig.from_json(det_json), recovery_every=RECOVERY_EVERY)


def run_configs(sim: SimConfig, atk: AttackConfig, det: DetectorConfig | None = None,
                scenario_key: str = "custom") -> PipelineResult:
    """Run (or reuse) the pipeline for these configs and make it the current run.

    det None -> the saved calibration for this swarm configuration, else the
    analytical defaults (load_or_default, Phase 2; F3: never silent).
    An incident is opened if any detector alarms.
    """
    det_cfg = det if det is not None else load_or_default(sim)
    key = run_key(sim, atk, det_cfg)
    pipe = _cached_pipeline(key, sim.to_json(), atk.to_json(), det_cfg.to_json())
    st.session_state.update({
        "scenario_key": scenario_key, "sim_cfg": sim, "atk_cfg": atk, "det_cfg": det_cfg,
        "seed": sim.seed, "pipeline_key": key, "pipeline": pipe, "sim_result": pipe.sim_result,
        "detection_runs": pipe.runs, "fusion": pipe.fusion, "recovery": pipe.recovery,
    })
    st.session_state["incident_id"] = maybe_open_incident(pipe, scenario_key)
    return pipe


def run_scenario(scenario_key: str, seed: int | None = None,
                 centre_gain: float | None = None) -> PipelineResult:
    """Run a preset scenario (optionally with another seed or centre_gain)."""
    scenario = get_scenario(scenario_key, seed=seed, centre_gain=centre_gain)
    return run_configs(scenario.sim, scenario.attack, scenario_key=scenario.key)


def get_or_run_pipeline() -> PipelineResult | None:
    """The current run, re-created from the stored configs if needed; None if nothing has run yet."""
    pipe = st.session_state.get("pipeline")
    if pipe is not None:
        return pipe
    sim, atk = st.session_state.get("sim_cfg"), st.session_state.get("atk_cfg")
    if sim is None or atk is None:
        return None
    return run_configs(sim, atk, st.session_state.get("det_cfg"),
                       st.session_state.get("scenario_key", "custom"))


def pipeline_for_incident(incident: dict[str, Any]) -> PipelineResult:
    """Re-run the exact pipeline of a stored incident from its saved configs.

    WHY this is valid evidence practice here: the simulation is deterministic
    (R2), so the same configs and seed reproduce the identical run.
    Does not change the current run.
    """
    sim = SimConfig.from_dict(incident["config"]["sim"])
    atk = AttackConfig.from_dict(incident["attack"])
    det = DetectorConfig.from_dict(incident["config"]["detector"])
    return _cached_pipeline(run_key(sim, atk, det), sim.to_json(), atk.to_json(), det.to_json())


def maybe_open_incident(pipe: PipelineResult, scenario_key: str) -> str | None:
    """Open (or reuse) an incident when any detector alarmed; returns its id or None."""
    if not should_open_incident(pipe.runs):
        return None
    incident, is_new = open_incident(pipe, scenario_key, get_storage())
    if is_new:
        logger.info("app opened incident %s", incident.incident_id)
    return incident.incident_id


def current_scenario() -> Scenario | None:
    """The preset Scenario of the current run (None for custom runs or nothing run)."""
    key = st.session_state.get("scenario_key")
    if not key or key == "custom":
        return None
    try:
        return get_scenario(key)
    except ValueError:
        return None


# ==========================================================================
# Data shaping (pure functions, no Streamlit)
# ==========================================================================
@dataclass(frozen=True)
class Event:
    """One line of the event feed / ribbon.

    t         time in seconds (simulation time)
    drone     0-based index, or None for swarm-level events
    detector  "A", "B", "C", "F" (fusion) or "pattern"
    status    status word used for the dot colour and the label
    text      the sentence shown
    """

    t: float
    drone: int | None
    detector: str
    status: str
    text: str


def build_events(pipe: PipelineResult, upto_t: float | None = None) -> list[Event]:
    """Detection events, newest first: first ALARM per (detector, drone), fused alarms, pattern.

    Only first alarms are listed; a drone that stays in alarm for 40 s is one
    event, not 400 rows. upto_t limits the feed to what had happened by then.
    """
    events: list[Event] = []
    for run in pipe.runs:
        name = DETECTOR_NAMES.get(run.short, run.name).lower()
        for drone in sorted(run.alarm_drones()):
            t = run.first_alarm_time(drone)
            if t is not None:
                events.append(Event(t, drone, run.short, "alarm",
                                    f"{drone_label(drone)} · {name} check · ALARM"))
        if not run.alarm_drones():
            t_swarm = run.first_swarm_alarm_time()
            if t_swarm is not None:
                events.append(Event(t_swarm, None, run.short, "alarm", f"Swarm · {name} check · ALARM"))
    alarms = pipe.fusion.alarms()
    for drone in range(alarms.shape[1]):
        hits = np.flatnonzero(alarms[:, drone])
        if hits.size:
            events.append(Event(float(pipe.fusion.times[hits[0]]), drone, "F", "alarm",
                                f"{drone_label(drone)} · fusion · COMPROMISED"))
    if events:
        key = pattern_key(pipe.runs)
        first = min(e.t for e in events)
        events.append(Event(first, None, "pattern", "alarm",
                            f"Swarm pattern · {key.replace('_', ' ')} · {describe_pattern(key)}"))
    if upto_t is not None:
        events = [e for e in events if e.t <= upto_t + 1e-9]
    return sorted(events, key=lambda e: (-e.t, e.detector, -1 if e.drone is None else e.drone))


def step_index(pipe: PipelineResult, t: float) -> int:
    """Index k of the last step with time <= t (0 if t is before the start)."""
    times = pipe.sim_result.times()
    return int(max(0, np.searchsorted(times, t, side="right") - 1))


def verdict_counts(pipe: PipelineResult, k: int) -> dict[str, int]:
    """Fused verdict counts at step k: total, normal, suspicious, compromised, insufficient_evidence."""
    row = pipe.fusion.verdicts[k]
    counts = {v.value: int(sum(x == v for x in row)) for v in Verdict}
    return {"total": len(row), **counts}


def step_results_at(pipe: PipelineResult, k: int) -> dict[str, StepResult]:
    """{detector letter: its StepResult at step k}."""
    return {run.short: run.steps[k] for run in pipe.runs}


def drone_rows(pipe: PipelineResult, k: int) -> pd.DataFrame:
    """One row per drone at step k: detector statuses and scores, fused verdict and belief.

    Columns: drone, A, A_score, B, B_cusum, B_nis, C, C_residual_m, verdict, belief.
    Status columns hold text labels (status is never colour alone).
    """
    steps = step_results_at(pipe, k)
    n = pipe.sim_result.n_drones
    nis = steps["B"].diagnostics.get("nis") if "B" in steps else None
    rows = []
    for i in range(n):
        row: dict[str, Any] = {"drone": drone_label(i)}
        for short in ("A", "B", "C"):
            if short in steps:
                row[short] = steps[short].status[i].value.replace("_", " ").upper()
        row["A_score"] = float(steps["A"].scores[i]) if "A" in steps else np.nan
        row["B_cusum"] = float(steps["B"].scores[i]) if "B" in steps else np.nan
        row["B_nis"] = float(nis[i]) if nis is not None else np.nan
        row["C_residual_m"] = float(steps["C"].scores[i]) if "C" in steps else np.nan
        row["verdict"] = pipe.fusion.verdicts[k, i].value.replace("_", " ").upper()
        row["belief"] = float(pipe.fusion.belief[k, i])
        rows.append(row)
    return pd.DataFrame(rows)


RECOMMENDED_ACTION: dict[str, str] = {
    Verdict.COMPROMISED.value: "Do not trust this drone's GPS. Use its recovered position (radio "
                               "ranges and anchors) and review the incident.",
    Verdict.SUSPICIOUS.value: "Watch this drone; check which detector flagged it before acting.",
    Verdict.INSUFFICIENT_EVIDENCE.value: "Not enough evidence either way; an anchor reading or more "
                                         "radio neighbours would help.",
    Verdict.NORMAL.value: "No anomaly detected. This does not prove its GPS is genuine.",
}


def inspector_data(pipe: PipelineResult, drone: int, k: int) -> dict[str, Any]:
    """Everything the Drone Inspector shows for one drone at step k."""
    if not 0 <= drone < pipe.sim_result.n_drones:
        raise ValueError(f"drone must be in [0, {pipe.sim_result.n_drones}), got {drone}")
    steps = step_results_at(pipe, k)
    verdict = pipe.fusion.verdicts[k, drone].value
    nis = steps["B"].diagnostics.get("nis") if "B" in steps else None
    detectors = []
    for short, step in sorted(steps.items()):
        score = float(step.scores[drone])
        detectors.append({"short": short, "name": DETECTOR_NAMES.get(short, short),
                          "status": step.status[drone].value,
                          "score": None if np.isnan(score) else score})
    return {
        "drone": drone, "label": drone_label(drone), "t": float(pipe.fusion.times[k]),
        "detectors": detectors,
        "nis": None if nis is None or np.isnan(nis[drone]) else float(nis[drone]),
        "verdict": verdict,
        "belief": float(pipe.fusion.belief[k, drone]),
        "plausibility": float(pipe.fusion.plausibility[k, drone]),
        "reasons": explain_drone(drone, steps, pipe.det_cfg),
        "action": RECOMMENDED_ACTION[verdict],
    }
