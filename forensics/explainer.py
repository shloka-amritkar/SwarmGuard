"""Plain-language explanations built ONLY from recorded facts.

explain_incident(incident) -> 10 to 12 sentences for reports and the app.
explain_drone(drone, step_results) -> short reasons for the UI inspector.

HOW IT WORKS: template sentences with blanks. Each blank is filled from one
field of the Incident (or one StepResult); if the field is missing, that
sentence is skipped. No sentence is written from anything else.
WHY templates and not free text (idea from the UAV Intelligence Fusion
System's template reports): an examiner can trace every claim back to a
number in incident.json, and the explainer can never "invent" a finding.

HONEST LANGUAGE (rule R4): a silent detector "detected no anomaly"; it never
says GPS was genuine or safe. Every explanation ends with the same
limitations, including that the system does not identify who caused it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from config import DetectorConfig
from core.schema import DroneStatus
from core.utils import drone_label, format_drones
from detection.base import StepResult

if TYPE_CHECKING:  # WHY: avoids a circular import (incident.py calls this module)
    from forensics.incident import Incident

logger = logging.getLogger(__name__)

FAR_ABOVE_RATIO = 3.0
LIMITATION_NO_PROOF = "No anomaly detected does not prove GPS was genuine."
LIMITATION_NO_ATTRIBUTION = "The system does not identify who caused this."
SIMULATION_NOTE = "All numbers in this explanation are simulation results."

PATTERN_DETAIL: dict[str, str] = {
    "A_silent_B_C_alarm": (
        "The formation stayed geometrically consistent, so relative ranging (Detector A) saw "
        "nothing, while the motion and anchor checks disagreed with GPS; this matches a "
        "coordinated whole-swarm shift, which a common offset hides from pairwise distances."
    ),
    "A_silent_B_alarm_C_no_evidence": (
        "The formation stayed geometrically consistent and only the motion check disagreed with "
        "GPS; with no anchor readings, a whole-swarm shift could not be confirmed absolutely."
    ),
    "C_silent_A_B_alarm": (
        "Individual drones disagreed both with their neighbours' radio ranges and with their own "
        "velocity sensors, the fingerprint of a targeted spoof on specific drones."
    ),
}


def _above(value: float, threshold: float) -> str:
    """"far above" when value >= 3x threshold, else "above"."""
    return "far above" if threshold > 0 and value >= FAR_ABOVE_RATIO * threshold else "above"


def _and(items: list[str]) -> str:
    """["B"] -> "B"; ["B", "C"] -> "B and C"; ["A", "B", "C"] -> "A, B and C"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _who(drones: list[int]) -> str:
    """"Drone 04" for one drone, "3 drones (Drone 01, Drone 02, Drone 05)" for several."""
    if len(drones) == 1:
        return drone_label(drones[0])
    return f"{len(drones)} drones ({format_drones(drones)})"


def _alarm_sentence(short: str, d: dict[str, Any]) -> str:
    """One sentence for a detector that raised an alarm."""
    head = f"Detector {short} ({d['name']}) first alarmed at {d['first_alarm_t']:.1f} s"
    head += f" on {_who(d['drones'])}" if d["drones"] else " at swarm level"
    peak, thr = d.get("peak_score"), d.get("threshold")
    if short == "A" and peak is not None:
        return (f"{head}; up to {peak:.0%} of a drone's pair distances disagreed with radio ranges "
                f"by more than {d.get('tau_a_m', 0):.1f} m, {_above(peak, thr)} the {thr:.0%} vote threshold.")
    if short == "B" and d.get("peak_nis") is not None:
        nis, nis_thr = d["peak_nis"], d["nis_threshold"]
        return (f"{head}; its NIS peaked at {nis:.1f}, {_above(nis, nis_thr)} the {nis_thr:.2f} "
                f"threshold, and its CUSUM reached {peak:.1f} (alarm level {thr:.1f}).")
    if short == "C" and peak is not None:
        return (f"{head}; the GPS-to-anchor distance peaked at {peak:.1f} m, "
                f"{_above(peak, thr)} the {thr:.1f} m threshold.")
    return head + "."


def _quiet_sentence(short: str, d: dict[str, Any]) -> str:
    """One sentence for a silent or no-evidence detector (never "GPS genuine")."""
    if d["state"] == "no_evidence":
        return (f"Detector {short} ({d['name']}) had no usable data (NO_EVIDENCE), "
                "so it could not check this run.")
    peak, thr = d.get("peak_score"), d.get("threshold")
    if peak is None:
        return f"Detector {short} ({d['name']}) detected no anomaly."
    unit = " m" if short == "C" else ""
    shown = f"{peak:.0%}" if short == "A" else f"{peak:.1f}{unit}"
    limit = f"{thr:.0%}" if short == "A" else f"{thr:.1f}{unit}"
    head = f"Detector {short} ({d['name']}) detected no anomaly: its {d['score_name']} "
    if peak < thr:
        return head + f"peaked at {shown}, below the {limit} threshold."
    # WHY this case exists: a single noisy step can cross the threshold, but
    # the persistence rule (m of the last n steps) never confirmed it, so no
    # alarm was raised. Saying "below the threshold" here would be false.
    return head + (f"briefly reached {shown}, above the {limit} threshold, but never on enough "
                   "consecutive steps to pass the persistence rule.")


def _fusion_sentence(fusion: dict[str, Any]) -> str:
    """How many drones fusion marked COMPROMISED persistently."""
    drones = fusion.get("fused_alarm_drones", [])
    if not drones:
        return "Evidence fusion did not mark any drone as compromised persistently."
    beliefs = [d["belief"] for d in fusion.get("drones", []) if d["drone"] in drones and d["belief"] is not None]
    peak = f", with belief up to {max(beliefs):.2f}" if beliefs else ""
    return f"Evidence fusion marked {_who(drones)} as compromised{peak}."


def _recovery_sentences(rec: dict[str, Any]) -> list[str]:
    """Recovery result and the estimated attack vector (if any)."""
    if rec.get("status") != "ok" or rec.get("attack_vector") is None:
        return [f"Position recovery without GPS was not possible (status: {rec.get('status', 'unknown')})."]
    vx, vy = rec["attack_vector"]
    out = [f"Recovery from radio ranges and anchors ({rec['method']}) estimated a common GPS offset "
           f"of ({vx:.2f}, {vy:.2f}) m east/north, magnitude {rec['attack_magnitude']:.2f} m."]
    if rec.get("outlier_anchors"):
        out.append(f"RANSAC rejected the anchor readings of {format_drones(rec['outlier_anchors'])} "
                   "as inconsistent with the rest.")
    return out


def explain_incident(incident: Incident) -> list[str]:
    """Return 10 to 12 factual sentences describing the incident.

    Order: overview; one sentence per detector (A, B, C); onset; agreement
    pattern; fusion; recovery; limitations. All values come from incident fields.
    """
    alarmed = incident.alarmed_detectors()
    overview = (f"Incident {incident.incident_id} concerns a simulated run of scenario "
                f"'{incident.scenario_key}' (seed {incident.seed})")
    overview += (f", in which Detector{'s' if len(alarmed) > 1 else ''} {_and(alarmed)} raised an alarm."
                 if alarmed else ", in which no detector raised an alarm.")
    sentences = [overview]
    for short, d in sorted(incident.detected.items()):
        sentences.append(_alarm_sentence(short, d) if d["state"] == "alarm" else _quiet_sentence(short, d))
    if incident.onset_estimate is not None:
        sentences.append(f"The anomaly is estimated to have started at {incident.onset_estimate:.1f} s "
                         f"(from the {incident.onset_source}).")
    detail = PATTERN_DETAIL.get(incident.pattern)
    sentences.append(f"Agreement pattern {incident.pattern}: "
                     + (detail if detail else incident.pattern_description))
    sentences.append(_fusion_sentence(incident.fusion))
    sentences.extend(_recovery_sentences(incident.recovery))
    sentences += [LIMITATION_NO_PROOF, LIMITATION_NO_ATTRIBUTION, SIMULATION_NOTE]
    return sentences


# --------------------------------------------------------------------------
# One drone, one step (for the UI's decision inspector)
# --------------------------------------------------------------------------
def _reason(short: str, i: int, step: StepResult, det: DetectorConfig) -> str:
    """One short reason from one detector's StepResult for drone i."""
    status = step.status[i]
    if status == DroneStatus.NO_EVIDENCE:
        return f"{short}: no evidence (no usable reading for this drone)"
    score = float(step.scores[i])
    label = status.value.upper()
    if short == "A":
        return (f"{short}: {label} - {score:.0%} of pair distances disagree with radio ranges "
                f"(vote threshold {det.vote_fraction:.0%})")
    if short == "B":
        nis = step.diagnostics.get("nis")
        nis_txt = (f"NIS {float(nis[i]):.1f} vs {det.nis_threshold:.2f}"
                   if nis is not None and np.isfinite(nis[i]) else "NIS not yet available")
        return f"{short}: {label} - {nis_txt}, CUSUM {score:.1f} vs {det.cusum_h:.1f}"
    if short == "C":
        return f"{short}: {label} - GPS is {score:.1f} m from the anchor fix (threshold {det.tau_c:.1f} m)"
    return f"{short}: {label} - score {score:.2f}"


def explain_drone(
    drone: int,
    step_results: dict[str, StepResult],
    det_cfg: DetectorConfig | None = None,
) -> list[str]:
    """Short reasons, one per detector, for one drone at one step.

    step_results maps detector letter -> that detector's StepResult at the
    step being inspected. det_cfg supplies the thresholds quoted (defaults
    to the analytical DetectorConfig()).
    """
    det = det_cfg if det_cfg is not None else DetectorConfig()
    reasons = []
    for short, step in sorted(step_results.items()):
        if not (0 <= drone < len(step.status)):
            raise ValueError(f"drone must be in [0, {len(step.status)}), got {drone}")
        reasons.append(_reason(short, drone, step, det))
    return reasons
