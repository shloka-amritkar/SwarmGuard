"""Attack Lab: build your own flight and attack, run it, and see which detectors catch it.

Left column: a preset to start from, then outlined expanders for the swarm,
sensors, anchors, manoeuvres, the attack and the detector thresholds.
Right column: one primary RUN button and the results.

WHY widget keys include the preset name: choosing another preset then
starts every control from that preset's values instead of keeping stale ones.
WHY configs are built inside try/except ValueError: SimConfig and
AttackConfig validate themselves (Phase 1), so an impossible combination
becomes a friendly message instead of a traceback.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import streamlit as st

from analysis.calibration import load_or_default
from attacks.scenarios import Scenario, get_scenario, scenario_keys
from config import (MULTI_TARGET_ATTACKS, SINGLE_TARGET_ATTACKS, VALID_ATTACK_TYPES, VALID_FORMATIONS,
                    VALID_PATHS, VALID_RAMP_PROFILES, AttackConfig, DetectorConfig, ManoeuvreEvent,
                    SimConfig)
from core.utils import drone_label
from ui import components as c
from ui.state import detection_results_table, expected_vs_actual, run_configs

DRIFTING = ("gradual_drift", "swarm_wide")
FORMATIONS = [f for f in VALID_FORMATIONS if f != "custom"]   # custom needs typed points


def _k(preset: str, name: str) -> str:
    """Widget key that changes with the preset."""
    return f"lab-{preset}-{name}"


def swarm_controls(p: str, sim: SimConfig) -> dict[str, Any]:
    """Swarm size, formation, path and timing."""
    with st.expander("Swarm", expanded=True):
        a, b = st.columns(2)
        out = {
            "n_drones": a.slider("Drones", 4, 30, sim.n_drones, key=_k(p, "n")),
            "formation": b.selectbox("Formation", FORMATIONS, key=_k(p, "form"),
                                     index=FORMATIONS.index(sim.formation) if sim.formation in FORMATIONS else 0),
            "radius": a.slider("Formation radius (m)", 3.0, 30.0, float(sim.radius), 0.5, key=_k(p, "r")),
            "path": b.selectbox("Path", VALID_PATHS, index=list(VALID_PATHS).index(sim.path), key=_k(p, "path")),
            "duration": a.slider("Duration (s)", 20.0, 120.0, float(sim.duration), 5.0, key=_k(p, "dur")),
            "center_speed": b.slider("Speed (m/s)", 0.0, 5.0, float(sim.center_speed), 0.5, key=_k(p, "v")),
            "seed": int(a.number_input("Seed", 0, 1_000_000, int(sim.seed), key=_k(p, "seed"))),
        }
    return out


def sensor_controls(p: str, sim: SimConfig) -> dict[str, Any]:
    """Noise levels and radio range, then the anchor settings."""
    with st.expander("Sensors"):
        a, b = st.columns(2)
        out = {
            "gps_sigma": a.slider("GPS noise sigma (m)", 0.2, 3.0, float(sim.gps_sigma), 0.1, key=_k(p, "gs")),
            "range_sigma": b.slider("Radio range sigma (m)", 0.05, 1.0, float(sim.range_sigma), 0.05,
                                    key=_k(p, "rs")),
            "vel_sigma": a.slider("Velocity sigma (m/s)", 0.02, 0.5, float(sim.vel_sigma), 0.02, key=_k(p, "vs")),
            "comm_range": b.slider("Radio range limit (m)", 10.0, 100.0, float(sim.comm_range), 5.0,
                                   key=_k(p, "cr")),
        }
    with st.expander("Anchors"):
        a, b = st.columns(2)
        out.update({
            "anchor_ratio": a.slider("Share of drones with an anchor", 0.0, 1.0, float(sim.anchor_ratio), 0.05,
                                     key=_k(p, "ar")),
            "anchor_sigma": b.slider("Anchor noise sigma (m)", 0.1, 2.0, float(sim.anchor_sigma), 0.1,
                                     key=_k(p, "as")),
            "anchor_fault_ratio": a.slider("Share of faulty anchors", 0.0, 0.5, float(sim.anchor_fault_ratio),
                                           0.05, key=_k(p, "af")),
            "anchor_dropout": b.slider("Anchor dropout probability", 0.0, 0.9, float(sim.anchor_dropout), 0.05,
                                       key=_k(p, "ad")),
        })
    return out


def _preset_event(base: list[ManoeuvreEvent], kind: str) -> ManoeuvreEvent | None:
    """The preset's first manoeuvre of this kind, if any."""
    return next((m for m in base if m.kind == kind), None)


def manoeuvre_controls(p: str, base: list[ManoeuvreEvent], duration: float) -> list[ManoeuvreEvent]:
    """Optional legitimate rotation and scaling (they must NOT be flagged as attacks).

    Defaults come from the preset, so an untouched preset is reproduced exactly.
    """
    events: list[ManoeuvreEvent] = []
    rot, scl = _preset_event(base, "rotate"), _preset_event(base, "scale")
    with st.expander("Manoeuvres"):
        st.caption("Legitimate formation changes. A good detector stays quiet during these.")
        if st.checkbox("Rotate the formation", value=rot is not None, key=_k(p, "rot")):
            a, b = st.columns(2)
            window = (rot.start, rot.end) if rot else (10.0, min(20.0, duration))
            start, end = a.slider("Rotation window (s)", 0.0, duration, window, 0.5, key=_k(p, "rotw"))
            angle = b.slider("Total turn (degrees)", -180.0, 180.0, rot.value if rot else 90.0, 5.0,
                             key=_k(p, "rotv"))
            events.append(ManoeuvreEvent("rotate", start, max(end, start + 0.5), angle))
        if st.checkbox("Scale the formation", value=scl is not None, key=_k(p, "scl")):
            a, b = st.columns(2)
            window = (scl.start, scl.end) if scl else (25.0, min(35.0, duration))
            start, end = a.slider("Scaling window (s)", 0.0, duration, window, 0.5, key=_k(p, "sclw"))
            factor = b.slider("Final size factor", 0.5, 2.0, scl.value if scl else 1.5, 0.1, key=_k(p, "sclv"))
            events.append(ManoeuvreEvent("scale", start, max(end, start + 0.5), factor))
    return events


def attack_controls(p: str, atk: AttackConfig, n_drones: int, duration: float) -> dict[str, Any]:
    """Attack type, targets, size, direction and timing."""
    with st.expander("Attack", expanded=True):
        kind = st.selectbox("Attack type", VALID_ATTACK_TYPES, index=list(VALID_ATTACK_TYPES).index(atk.type),
                            key=_k(p, "type"), format_func=lambda t: t.replace("_", " "))
        out: dict[str, Any] = {"type": kind, "targets": []}
        if kind == "none":
            st.caption("No attack: use this to measure false alarms.")
            return out
        options = list(range(n_drones))
        default = [t for t in atk.targets if t < n_drones] or [0]
        if kind in SINGLE_TARGET_ATTACKS:
            out["targets"] = [st.selectbox("Target drone", options, index=default[0], format_func=drone_label,
                                           key=_k(p, "t1"))]
        elif kind in MULTI_TARGET_ATTACKS:
            out["targets"] = st.multiselect("Target drones", options, default=default, format_func=drone_label,
                                            key=_k(p, "tn"))
        a, b = st.columns(2)
        out["magnitude_m"] = a.slider("Offset size (m)", 0.0, 40.0, float(atk.magnitude_m), 0.5, key=_k(p, "mag"))
        out["direction_deg"] = b.slider("Direction (degrees, 0 = east)", 0.0, 359.0, float(atk.direction_deg) % 360,
                                        1.0, key=_k(p, "dir"))
        out["start_time"] = a.slider("Start (s)", 0.0, duration - 1.0, min(float(atk.start_time), duration - 1.0),
                                     0.5, key=_k(p, "start"))
        out["ramp_profile"] = b.selectbox("Ramp profile", VALID_RAMP_PROFILES,
                                          index=list(VALID_RAMP_PROFILES).index(atk.ramp_profile), key=_k(p, "ramp"))
        if kind in DRIFTING:
            out["drift_rate"] = a.slider("Drift rate (m/s)", 0.0, 2.0, float(atk.drift_rate), 0.05, key=_k(p, "dr"))
            out["drift_cap"] = b.checkbox("Stop drifting at the offset size", value=atk.drift_cap, key=_k(p, "cap"))
        if kind == "rigid_rotation":
            out["rotation_rate_deg_s"] = a.slider("Rotation rate (deg/s)", 0.1, 5.0, float(atk.rotation_rate_deg_s),
                                                  0.1, key=_k(p, "rr"))
        if kind == "intermittent":
            out["pulse_period"] = a.slider("Pulse period (s)", 0.5, 10.0, float(atk.pulse_period), 0.5, key=_k(p, "pp"))
            out["pulse_duty"] = b.slider("Share of time on", 0.1, 1.0, float(atk.pulse_duty), 0.05, key=_k(p, "pd"))
    return out


def detector_controls(p: str, calibrated: DetectorConfig) -> DetectorConfig | None:
    """Calibrated thresholds by default; manual values only when the box is unticked (F3)."""
    with st.expander("Detector thresholds"):
        if st.checkbox("Use calibrated thresholds", value=True, key=_k(p, "cal")):
            st.caption(f"tau_a {calibrated.tau_a:.1f} m · CUSUM h {calibrated.cusum_h:.1f} · "
                       f"NIS {calibrated.nis_threshold:.2f} · tau_c {calibrated.tau_c:.1f} m")
            return None
        a, b = st.columns(2)
        return replace(
            calibrated,
            tau_a=a.slider("tau_a (m)", 1.0, 15.0, float(calibrated.tau_a), 0.1, key=_k(p, "ta")),
            cusum_h=b.slider("CUSUM h", 2.0, 100.0, float(calibrated.cusum_h), 0.5, key=_k(p, "h")),
            nis_threshold=a.slider("NIS threshold", 2.0, 30.0, float(calibrated.nis_threshold), 0.1, key=_k(p, "nis")),
            tau_c=b.slider("tau_c (m)", 1.0, 15.0, float(calibrated.tau_c), 0.1, key=_k(p, "tc")),
        )


def build_configs(p: str, base: Scenario) -> tuple[SimConfig, AttackConfig, DetectorConfig | None]:
    """Read every control and build validated configs (raises ValueError on bad combinations)."""
    swarm = swarm_controls(p, base.sim)
    sensors = sensor_controls(p, base.sim)
    manoeuvres = manoeuvre_controls(p, base.sim.manoeuvres, swarm["duration"])
    attack = attack_controls(p, base.attack, swarm["n_drones"], swarm["duration"])
    # WHY keep the preset's shape sequence only while the formation is unchanged:
    # a light-show sequence belongs to that preset's formation.
    shapes = base.sim.shape_sequence if swarm["formation"] == base.sim.formation else []
    sim = replace(base.sim, **swarm, **sensors, manoeuvres=manoeuvres, shape_sequence=shapes, custom_points=[])
    atk = replace(base.attack, **attack, seed=swarm["seed"])
    return sim, atk, detector_controls(p, load_or_default(sim))


def results(expected: dict[str, str]) -> None:
    """Results of the last Attack Lab run."""
    pipe = st.session_state.get("pipeline")
    if pipe is None or st.session_state.get("lab-ran") != st.session_state.get("pipeline_key"):
        c.empty_state("Set up a flight and an attack on the left, then press Run.", sticker_name="crosshair")
        return
    c.section("Results", "Evaluation: compares alarms with the simulation's attack labels.", eyebrow="Simulation result")
    st.dataframe(detection_results_table(pipe), hide_index=True, width="stretch", column_config={
        "detector": "Detector", "detected": "Detected",
        "delay_s": st.column_config.NumberColumn("Delay (s)", format="%.1f"),
        "identified": "Drones flagged", "false_alarms": "False alarms"})
    check = expected_vs_actual(expected, pipe)
    if not check.empty:
        c.section("Expected vs actual", "What the preset documents each detector should do.")
        st.dataframe(check, hide_index=True, width="stretch")
    incident = st.session_state.get("incident_id")
    if incident:
        st.success(f"Incident {incident} opened for this run.")
        if st.button("Open the incident", key="lab-open-incident"):
            st.session_state["incidents-select"] = incident
            st.switch_page("views/incidents.py")
    else:
        st.info("No detector alarmed: no anomaly detected, so no incident was opened.")


def main() -> None:
    """Draw the page."""
    c.page_header("Attack", "lab", eyebrow="Operations",
                  caption="Design a flight and a simulated GPS spoof, then see which detectors catch it. "
                          "All attacks are software offsets on simulated numbers.")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        preset = st.selectbox("Start from preset", scenario_keys(), key="lab-preset",
                              format_func=lambda k: get_scenario(k).title)
        base = get_scenario(preset)
        st.caption(base.description)
        try:
            sim, atk, det = build_configs(preset, base)
            error = None
        except ValueError as exc:
            error = str(exc)
    with right:
        if error is not None:
            st.error(f"These settings cannot be simulated: {error}")
        elif st.button("Run attack", type="primary", key="lab-run", width="stretch"):
            with st.spinner("Simulating, detecting, fusing and recovering..."):
                run_configs(sim, atk, det, scenario_key=preset if _unchanged(sim, atk, base) else "custom")
            st.session_state["lab-ran"] = st.session_state.get("pipeline_key")
        results(base.expected if error is None and _unchanged(sim, atk, base) else {})


def _unchanged(sim: SimConfig, atk: AttackConfig, base: Scenario) -> bool:
    """True if the controls still equal the preset (then its expectations apply)."""
    return sim.to_dict() == base.sim.to_dict() and atk.to_dict() == base.attack.to_dict()


main()
