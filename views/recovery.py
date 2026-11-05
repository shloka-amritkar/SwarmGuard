"""Recovery: where are the drones really, if GPS cannot be trusted?

Two methods from Phase 3, both using only observations (rule R1):
  MDS + RANSAC   rebuilds the whole formation from radio ranges, then pins
                 it to the map with the anchors RANSAC accepts. The mean of
                 (GPS - recovered) is the estimated swarm-wide attack vector.
  Multilateration  repairs one flagged drone from its trusted neighbours'
                 ranges; flagged neighbours are never used (fix F1).
The before/after error table compares with the TRUE positions, so it is
labelled as evaluation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.experiments import PipelineResult
from core.utils import drone_label, format_drones
from recovery.mds_alignment import RecoveryResult, RecoveryStatus
from recovery.multilateration import RepairResult, recovery_error
from ui import components as c
from ui import plots
from ui.state import step_index


def nearest_result(pipe: PipelineResult, k: int) -> RecoveryResult | None:
    """The recovery result at step k, or the last computed one before it."""
    if pipe.recovery is None:
        return None
    for j in range(k, -1, -1):
        if pipe.recovery.results[j] is not None:
            return pipe.recovery.results[j]
    return None


def summary_kpis(pipe: PipelineResult) -> None:
    """Run-level attack vector estimate (median over the last 5 s)."""
    s = pipe.recovery.summary() if pipe.recovery is not None else {"status": "not_run"}
    vec = s.get("attack_vector")
    c.kpi_row([
        c.KPI("Recovery status", str(s.get("status", "-")).replace("_", " ")),
        c.KPI("Attack vector size (m)", f"{s['attack_magnitude']:.2f}" if vec else "-"),
        c.KPI("East / north (m)", f"{vec[0]:+.1f} / {vec[1]:+.1f}" if vec else "-"),
        c.KPI("Anchors rejected", len(s.get("outlier_anchors") or []),
              "suspicious" if s.get("outlier_anchors") else None,
              delta=format_drones(s["outlier_anchors"]) if s.get("outlier_anchors") else "none"),
    ])


def repairs_table(repairs: dict[int, RepairResult]) -> pd.DataFrame:
    """One row per flagged drone: status, neighbours used, residual RMS (no truth)."""
    rows = [{"drone": drone_label(i), "status": r.status.value.replace("_", " ").upper(),
             "neighbours used": len(r.neighbours), "residual RMS (m)": r.residual_rms}
            for i, r in sorted(repairs.items())]
    return pd.DataFrame(rows, columns=["drone", "status", "neighbours used", "residual RMS (m)"])


def evaluation_table(pipe: PipelineResult, k: int, repairs: dict[int, RepairResult]) -> None:
    """EVALUATION: GPS error vs repaired error against the true positions."""
    c.section("Before and after", "Evaluation: compares with simulator ground truth, which recovery "
              "never sees.", eyebrow="Evaluation view")
    df = recovery_error(pipe.sim_result.frames[k], repairs)
    df["drone"] = df["drone"].map(drone_label)
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "gps_error_m": st.column_config.NumberColumn("GPS error (m)", format="%.2f"),
        "repaired_error_m": st.column_config.NumberColumn("Repaired error (m)", format="%.2f")})


def main() -> None:
    """Draw the page."""
    c.page_header("Position", "recovery", eyebrow="Analysis",
                  caption="Estimating true positions from radio ranges and anchors. Simulation result.")
    pipe = c.require_run("recovery")
    if pipe is None:
        return
    summary_kpis(pipe)
    times = pipe.sim_result.times()
    t = st.slider("Time (s)", float(times[0]), float(times[-1]), float(times[-1]), 0.5, key="rec-t")
    k = step_index(pipe, t)
    res = nearest_result(pipe, k)
    obs = pipe.sim_result.frames[k].observed
    repairs = pipe.repairs_at(k)
    repaired = {i: r.position for i, r in repairs.items() if r.position is not None}
    show_truth = st.toggle("Evaluation view: show true positions", key="rec-truth")
    truth = pipe.sim_result.frames[k].true_pos if show_truth else None
    ok = res is not None and res.status == RecoveryStatus.OK
    if not ok:
        st.warning("MDS recovery was not possible at this step: "
                   f"{res.status.value.replace('_', ' ') if res else 'not run'}.")
    with c.paper_card("recovery-map"):
        c.chart(plots.recovery_map(obs, res.positions if ok else None, res.inlier_mask if ok else None,
                                   res.outlier_anchors if ok else [], repaired, truth), key="rec-map")
    c.explain_box("Hollow circles are the reported GPS positions; filled forest dots are where MDS + RANSAC "
                  "puts the drones; grey lines join the two, so a swarm-wide shift shows as many parallel "
                  "lines. Ochre diamonds are anchors RANSAC trusted; a red cross is an anchor it rejected. "
                  "Lavender squares are multilateration repairs of flagged drones.")
    c.section("Estimated attack vector", "Mean of (GPS - recovered position) over all drones.")
    c.chart(plots.attack_vector_chart(times, pipe.recovery.attack_vector_series(),
                                      pipe.sim_result.attack_onset()), key="rec-vec")
    c.section("Multilateration repairs", "Flagged drones at this time, repaired from trusted neighbours "
              "only (flagged neighbours are never used, fix F1).")
    if not repairs:
        c.html_block('<p class="sg-card__body">No drone is flagged at this time, so nothing needs repair.</p>')
        return
    st.dataframe(repairs_table(repairs), hide_index=True, width="stretch")
    if any(r.status == RecoveryStatus.INSUFFICIENT_NEIGHBOURS for r in repairs.values()):
        st.info("INSUFFICIENT NEIGHBOURS: fewer than three trusted neighbours with a radio range, "
                "so that drone's position cannot be fixed by multilateration.")
    evaluation_table(pipe, k, repairs)


main()
