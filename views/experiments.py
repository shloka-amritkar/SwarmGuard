"""Experiments: calibration, parameter sweeps, the stealth envelope and swarm propagation.

These are the statistical results of the project: each one repeats the
simulation over many seeds (or many drones) instead of trusting one flight.

WHY a runtime estimate before every job: one trial takes about a quarter of
a second, so a sweep of 11 values x 20 seeds already takes about a minute.
The page says so before starting. WHY one process: a process pool inside
Streamlit on Windows is fragile; the CLI (--workers) is the tool for large
runs. Finished results are kept in session state, so switching tabs or
pages does not lose them.

The "Experimental ML detector" toggle adds a fifth tab with Detector D's
held-out test metrics and feature importance (Phase 6).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from analysis.calibration import calibrate, load_calibration, save_calibration
from analysis.evasion_search import stealth_envelope
from analysis.experiments import BUILTIN_SWEEPS, summarise, sweep
from analysis.propagation import PropagationResult, propagation_study
from attacks.scenarios import get_scenario, list_scenarios, scenario_keys
from core.utils import drone_label
from ui import components as c
from ui import plots

SECONDS_PER_TRIAL = 0.25   # measured on the default 10-drone scenario


def _progress_bar(label: str) -> Callable[[float], None]:
    """A styled progress bar and the callback the analysis functions call with 0..1."""
    bar = st.progress(0.0, text=label)
    return lambda fraction: bar.progress(min(max(float(fraction), 0.0), 1.0), text=label)


def _estimate(trials: int) -> str:
    """'about 40 s' / 'about 3 min' for a number of trials."""
    seconds = trials * SECONDS_PER_TRIAL
    return f"about {seconds:.0f} s" if seconds < 90 else f"about {seconds / 60:.0f} min"


def _results() -> dict:
    """Finished experiment results for this browser session."""
    return st.session_state.setdefault("exp-results", {})


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibration_tab() -> None:
    """Saved calibration for a swarm configuration, or a button to run one."""
    c.section("Threshold calibration", "Thresholds chosen on clean flights so that at most the target "
              "share of runs raises a false alarm (fix F3: never changed silently).")
    key = st.selectbox("Swarm configuration of scenario", scenario_keys(), key="exp-cal-scn",
                       format_func=lambda k: get_scenario(k).title)
    sim = get_scenario(key).sim
    result = load_calibration(sim)
    if result is None:
        c.empty_state("No calibration saved for this swarm configuration yet; the analytical "
                      "defaults are in use.", sticker_name="radar")
    else:
        table = result.table()
        for col in ("fa_rate", "ci_low", "ci_high"):
            table[col] = table[col].map(lambda v: f"{v:.1%}")
        st.caption(f"Simulation result · {result.n_seeds} clean runs (seeds {result.seed_start}-"
                   f"{result.seed_end}) · target {result.target_fa:.1%} · created {result.created_utc}")
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "detector": "Detector", "thresholds": "Calibrated thresholds",
            "false_alarm_runs": "Runs with a false alarm", "runs": "Clean runs",
            "fa_rate": "False-alarm rate", "ci_low": "95 % CI low", "ci_high": "95 % CI high"})
        chosen = {"A": result.thresholds.tau_a, "C": result.thresholds.tau_c}
        c.chart(plots.calibration_curve(result.curves, result.target_fa, chosen), key="exp-cal-curve")
        for note in result.notes:
            st.caption(f"Note: {note}")
    with st.expander("Run a new calibration"):
        n = st.slider("Clean runs", 20, 300, 100, 10, key="exp-cal-n")
        st.warning(f"This flies {n} clean runs: {_estimate(n * 2)}. It replaces the saved calibration.")
        if st.button("Calibrate", type="primary", key="exp-cal-run"):
            new = calibrate(sim, n_seeds=n, progress=_progress_bar("Calibrating on clean flights..."))
            save_calibration(new)
            st.rerun()


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------
def sweep_tab() -> None:
    """Detection rate as one attack or swarm parameter changes."""
    c.section("Parameter sweeps", "Detection rate with 95 % Wilson intervals as one parameter changes.")
    name = st.selectbox("Sweep", list(BUILTIN_SWEEPS), key="exp-sweep-name",
                        format_func=lambda k: BUILTIN_SWEEPS[k].description)
    spec = BUILTIN_SWEEPS[name]
    n = st.slider("Seeds per value", 5, 50, 10, 5, key="exp-sweep-n")
    trials = n * len(spec.values)
    st.caption(f"{len(spec.values)} values x {n} seeds = {trials} trials, {_estimate(trials)}. "
               f"Base scenario: {spec.scenario}.")
    if st.button("Run sweep", type="primary", key="exp-sweep-run"):
        df = sweep(spec.param, spec.values, spec.scenario, n, include_recovery=False,
                   progress=_progress_bar(f"Sweeping {spec.param}..."))
        _results()[("sweep", name, n)] = df
    done = [k for k in _results() if k[0] == "sweep" and k[1] == name]
    if not done:
        c.empty_state("No sweep results yet in this session. Choose a sweep and press Run sweep.",
                      sticker_name="signal")
        return
    df = _results()[done[-1]]
    summary = summarise(df)
    st.caption(f"Simulation result · {done[-1][2]} seeds per value")
    c.chart(plots.detection_curve(summary, spec.param), key="exp-sweep-curve")
    c.explain_box("Each line is one detector (F = fusion); the shaded band is the 95 % Wilson interval, "
                  "which is wide when few seeds were run. On a swarm-wide shift Detector A stays near "
                  "zero at every size: the common offset cancels in every pair distance.")
    c.chart(plots.delay_box(df), key="exp-sweep-delay")
    pivot = summary.pivot(index="value", columns="detector", values="false_alarm_rate")
    st.caption("False-alarm rate (honest drone flagged, or alarm before onset)")
    st.dataframe(pivot.map(lambda v: f"{v:.0%}"), width="stretch")


# --------------------------------------------------------------------------
# Evasion
# --------------------------------------------------------------------------
def evasion_tab() -> None:
    """Smallest attack each detector catches in 50 % and 90 % of runs."""
    c.section("Stealth envelope", "How small can an attack be and still be caught? Found by bisection "
              "over the attack size, each point measured on many seeds.")
    attacks = [s.key for s in list_scenarios() if s.attack.type != "none"]
    key = st.selectbox("Scenario", attacks, key="exp-ev-scn", format_func=lambda k: get_scenario(k).title)
    a, b = st.columns(2)
    n = a.slider("Seeds per evaluation", 10, 40, 20, 5, key="exp-ev-n")
    delay = b.slider("Must be detected within (s)", 0.0, 30.0, 10.0, 1.0, key="exp-ev-delay",
                     help="0 = any time after onset")
    st.warning(f"The bisection evaluates about 12 attack sizes: {_estimate(12 * n)}.")
    if st.button("Find the envelope", type="primary", key="exp-ev-run"):
        df = stealth_envelope(key, n_seeds=n, max_delay_s=delay if delay > 0 else None,
                              progress=_progress_bar("Bisecting attack sizes..."))
        _results()[("evasion", key)] = df
    df = _results().get(("evasion", key))
    if df is None:
        c.empty_state("No envelope computed yet in this session.", sticker_name="crosshair")
        return
    st.caption("Simulation result")
    c.chart(plots.stealth_envelope_chart(df), key="exp-ev-chart")
    shown = df.assign(threshold=df["threshold"].map(
        lambda v: "not reliably detected in range" if pd.isna(v) else f"{v:.3f}"))
    st.dataframe(shown, hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------
def _propagation_result(res: PropagationResult, key: str) -> None:
    """One centre_gain column: heatmap, influence ranking, separation."""
    off = res.matrix[res.matrix > 0]
    c.kpi_row([c.KPI("Mean spill-over (m)", f"{off.mean():.2f}" if off.size else "0.00"),
               c.KPI("Worst spill-over (m)", f"{res.matrix.max():.2f}"),
               c.KPI("Near collisions", sum(len(v) for v in res.near_collisions.values()),
                     "suspicious" if res.near_collisions else "normal")])
    c.chart(plots.propagation_heatmap(res.matrix), key=f"{key}-heat")
    c.chart(plots.pagerank_bar(res.influence), key=f"{key}-rank")
    top = res.influence.argsort()[::-1][:3]
    st.caption("Most influential: " + ", ".join(f"{drone_label(int(i))} ({res.influence[i]:.3f})" for i in top)
               + f" · min separation clean {res.min_separation_baseline:.2f} m, worst attacked "
               f"{res.min_separation_attacked.min():.2f} m")


def propagation_tab() -> None:
    """How one spoofed drone disturbs the others, at centre_gain 0 and 0.5 (D1)."""
    c.section("Swarm propagation", "Spoof each drone alone and measure how far the others are pushed. "
              "Evaluation: uses true positions. Run at both control settings (design decision D1).")
    key = st.selectbox("Swarm of scenario", scenario_keys(), key="exp-pr-scn",
                       format_func=lambda k: get_scenario(k).title)
    a, b = st.columns(2)
    magnitude = a.slider("Spoof size (m)", 1.0, 20.0, 6.0, 0.5, key="exp-pr-mag")
    seed = int(b.number_input("Seed", 0, 1_000_000, 5000, key="exp-pr-seed"))
    n = get_scenario(key).sim.n_drones
    st.caption(f"{2 * (n + 1)} short simulations (no detectors): a few seconds.")
    if st.button("Run propagation study", type="primary", key="exp-pr-run"):
        with st.spinner("Spoofing each drone in turn at centre_gain 0 and 0.5..."):
            _results()[("prop", key, magnitude, seed)] = propagation_study(get_scenario(key).sim, magnitude=magnitude,
                                                                           seed=seed)
    found = _results().get(("prop", key, magnitude, seed))
    if found is None:
        c.empty_state("No propagation study yet in this session.", sticker_name="constellation")
        return
    cols = st.columns(len(found))
    for col, res in zip(cols, found):
        with col:
            c.section(f"centre_gain = {res.centre_gain:g}", "Simulation result", eyebrow="Control law")
            _propagation_result(res, f"exp-pr-{res.centre_gain:g}")
    c.explain_box("Row = the drone that was spoofed, column = a drone that was pushed. With centre_gain 0 "
                  "(pure consensus) the spoof spreads about D/N to every drone. With centre_gain 0.5 the "
                  "path term keeps the others near their path while the spoofed drone itself is dragged. "
                  "PageRank ranks which drones spread disturbance the most.")


# --------------------------------------------------------------------------
# Experimental ML (Detector D)
# --------------------------------------------------------------------------
def _fmt(value: float) -> str:
    """0.934 -> "0.93"; NaN (undefined) -> "-"."""
    return "-" if value is None or pd.isna(value) else f"{value:.2f}"


def ml_tab() -> None:
    """Held-out metrics, confusion matrix and feature importance of the trained forest."""
    c.section("Experimental ML detector", "A Random Forest trained on simulated runs, tested on runs it "
              "never saw (split by run, fix F6). Experimental: it can only be as realistic as the "
              "simulator.", eyebrow="Detector D")
    trained = c.require_ml_model("experiments")
    if trained is None:
        return
    info = trained.info
    m = info["metrics"]
    st.caption("Simulation result, held-out test runs")
    c.kpi_row([c.KPI("Accuracy", _fmt(m["accuracy"])), c.KPI("Precision", _fmt(m["precision"])),
               c.KPI("Recall", _fmt(m["recall"])), c.KPI("F1", _fmt(m["f1"])),
               c.KPI("ROC-AUC", _fmt(m["roc_auc"]))])
    left, right = st.columns([3, 2], gap="large")
    with left:
        c.section("Feature importance", "Which measurements the forest relies on most.")
        c.chart(plots.feature_importance_bar(info["feature_importances"]), key="exp-ml-imp")
    with right:
        c.section("Confusion matrix", "Rows are the truth, columns the forest's answer (per drone-step).")
        c.chart(plots.confusion_chart(info["confusion"]), key="exp-ml-cm")
        recall = pd.DataFrame({"attack type": list(info["per_type_recall"]),
                               "recall": list(info["per_type_recall"].values())})
        st.dataframe(recall, hide_index=True, width="stretch", column_config={
            "recall": st.column_config.ProgressColumn("Recall", format="percent", min_value=0, max_value=1)})
        st.caption(f"Clean-row false-positive rate: {_fmt(info.get('clean_false_positive_rate'))}")
    data = info.get("dataset", {})
    st.caption(f"Trained {info.get('created_utc', '?')} on {data.get('n_runs', '?')} runs "
               f"(seeds {data.get('seed_start', '?')}-{data.get('seed_end', '?')}): "
               f"{len(info['train_groups'])} train runs, {len(info['test_groups'])} test runs, "
               f"no run in both. sha256 {trained.sha256}")
    c.explain_box("**Feature importance** is how much, on average, splits on a feature purify the "
                  "trees' groups. A dominant 'cusum' says the forest mostly re-uses Detector B's evidence. "
                  "**Recall per attack type** shows which attacks the forest learned to recognise. "
                  "**Why split by run**: steps 0.1 s apart are near-copies, so a row-level split would "
                  "test the forest on rows it has effectively already seen and overstate its accuracy.")


def main() -> None:
    """Draw the page."""
    c.page_header("Experiments", eyebrow="Analysis",
                  caption="Statistics over many seeded runs. Every rate is a simulation result.")
    show_ml = st.toggle("Experimental ML detector", key="exp-ml",
                        help="Adds a tab for Detector D, a Random Forest trained on simulated runs.")
    names = ["Calibration", "Sweeps", "Evasion", "Propagation"] + (["Experimental ML"] if show_ml else [])
    tabs = st.tabs(names)
    with tabs[0]:
        calibration_tab()
    with tabs[1]:
        sweep_tab()
    with tabs[2]:
        evasion_tab()
    with tabs[3]:
        propagation_tab()
    if show_ml:
        with tabs[4]:
            ml_tab()


main()
