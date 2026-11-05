"""Detection Analysis: what each detector saw, drone by drone, and why.

Left: per-drone detector time series, Detector A's residual heatmap at a
chosen step, the fusion chart, the comparison and agreement tables.
Right: the Drone Inspector for the chosen drone and step (the "decision
inspector" pattern credited to ZYRNAV Resilience Lab, R6).
Behind the "Experimental ML detector" toggle: Detector D (Phase 6), shown
separately and never mixed into fusion.
"""

from __future__ import annotations

import streamlit as st

from analysis.experiments import PipelineResult
from core.utils import drone_label
from ui import components as c
from ui import plots
from ui.state import agreement_matrix, comparison_table, inspector_data, ml_run, step_index


def pick_drone(pipe: PipelineResult) -> int:
    """Drone selector; defaults to the first drone any detector alarmed on."""
    alarmed = sorted({i for r in pipe.runs for i in r.alarm_drones()})
    n = pipe.sim_result.n_drones
    return st.selectbox("Drone", list(range(n)), index=alarmed[0] if alarmed else 0,
                        format_func=drone_label, key="det-drone")


def timeseries_section(pipe: PipelineResult, drone: int, onset: float | None) -> None:
    """A vote share, B NIS and CUSUM, C residual for one drone."""
    c.section(f"{drone_label(drone)} over time", "Each panel is one detector's score with its threshold "
              "(dashed). Simulation result.", eyebrow="Detectors A, B, C")
    c.chart(plots.detector_timeseries(pipe, drone, onset), key="det-ts")
    c.explain_box(
        "**A vote share**: the share of this drone's pairs whose GPS distance disagrees with the radio "
        "range by more than tau_a. Above the 0.5 line, a majority of neighbours disagree.\n\n"
        "**B NIS** (log scale): how surprising each GPS fix is to a Kalman filter driven by the "
        "velocity sensor. **B CUSUM** adds up small surprises, so slow drifts build up until they "
        "cross h.\n\n**C residual**: distance between GPS and the drone's absolute anchor fix. Empty "
        "means the drone carries no anchor (no evidence, never 'normal'). The dotted line marks the "
        "simulation's attack onset, which the detectors never see.")


def heatmap_section(pipe: PipelineResult) -> None:
    """Detector A's pair residual matrix at a chosen step."""
    c.section("Pair residuals at one moment", "|GPS distance - radio range| for every pair of drones.",
              eyebrow="Detector A")
    times = pipe.sim_result.times()
    t = st.slider("Time (s)", float(times[0]), float(times[-1]), float(times[-1]), 0.5, key="det-heat-t")
    obs = pipe.sim_result.frames[step_index(pipe, t)].observed
    c.chart(plots.residual_heatmap(obs, pipe.det_cfg.tau_a), key="det-heat")
    c.explain_box("Each cell compares two drones. White means GPS and radio agree; blush means the "
                  "residual is approaching tau_a; red means it is well above. A single spoofed drone "
                  "shows as one red row and column. A swarm-wide shift leaves the whole matrix white: "
                  "Detector A's blind spot.")


def fusion_section(pipe: PipelineResult, drone: int, onset: float | None) -> None:
    """Belief and plausibility for one drone, then the comparison and agreement tables."""
    c.section("Fused evidence", "Dempster-Shafer belief and plausibility that the drone is compromised.",
              eyebrow="Fusion")
    c.chart(plots.fusion_chart(pipe, drone, onset), key="det-fusion")
    c.explain_box("**Belief** counts only evidence that directly supports 'compromised'. **Plausibility** "
                  "is belief plus everything still uncertain. A wide gap between them means the "
                  "detectors lack evidence, not that the drone is fine.")
    c.section("Detector comparison", "Over the whole run; uses only what a real drone could measure.")
    st.dataframe(comparison_table(pipe), hide_index=True, width="stretch", column_config={
        "detector": "Detector", "state": "State",
        "first_alarm_s": st.column_config.NumberColumn("First alarm (s)", format="%.1f"),
        "drones": "Drones in alarm", "peak": st.column_config.NumberColumn("Peak score", format="%.2f"),
        "score": "Score", "threshold": "Threshold"})
    c.section("Agreement matrix", "Did each detector ever alarm on each drone? F = fused verdict.")
    st.dataframe(agreement_matrix(pipe), hide_index=True, width="stretch")
    c.explain_box("Reading across a row shows which detectors agree about one drone. A column of "
                  "'silent' under A next to alarms under B and C is the signature of a rigid swarm "
                  "shift. 'No evidence' means the detector had no usable data for that drone.")


def ml_section(pipe: PipelineResult, drone: int, onset: float | None) -> None:
    """Experimental Detector D: probability for the chosen drone and its alarms."""
    c.section("Experimental ML detector", "A Random Forest trained only on this simulator. "
              "Experimental; kept out of fusion. Simulation result.", eyebrow="Detector D")
    trained = c.require_ml_model("detection")
    if trained is None:
        return
    run = ml_run(pipe, trained)
    c.html_block(c.detector_pill_html("D", "Experimental ML"))
    c.chart(plots.ml_probability_chart(run, drone, onset, trained.info.get("threshold", 0.5)), key="det-ml-p")
    c.chart(plots.alarm_timeline([*pipe.runs, run], onset, "simulation onset"), key="det-ml-tl")
    st.caption(f"Model sha256 {trained.sha256[:16]}... trained on "
               f"{trained.info.get('dataset', {}).get('n_runs', '?')} simulated runs. "
               "Simulation result; not evidence about real drones.")
    c.explain_box("**P(attacked)** is the share of the forest's trees that vote 'attacked' for this "
                  "drone at this step. A raw flag needs P > 0.5; an alarm also needs the same m-of-n "
                  "persistence as A, B and C. The model only knows the attacks our simulator can "
                  "produce, so treat it as an experiment, not a fourth independent witness: its inputs "
                  "ARE detectors A, B and C plus GPS motion features.")


def main() -> None:
    """Draw the page."""
    c.page_header("How each detector", "saw it", eyebrow="04 · Detection analysis",
                  caption="Why each detector did or did not alarm. Simulation result.")
    pipe = c.require_run("detection")
    if pipe is None:
        return
    onset = pipe.sim_result.attack_onset()
    left, right = st.columns([2, 1], gap="large")
    with right:
        show_ml = st.toggle("Experimental ML detector", key="det-ml",
                            help="Adds Detector D, a Random Forest trained on simulated runs.")
        drone = pick_drone(pipe)
        times = pipe.sim_result.times()
        t = st.slider("Inspect at time (s)", float(times[0]), float(times[-1]), float(times[-1]), 0.5,
                      key="det-inspect-t")
        c.drone_inspector(inspector_data(pipe, drone, step_index(pipe, t)))
    with left:
        timeseries_section(pipe, drone, onset)
        heatmap_section(pipe)
        fusion_section(pipe, drone, onset)
        if show_ml:
            ml_section(pipe, drone, onset)


main()
