"""Swarm Simulator: watch a preset flight as an animation over Pune.

The map animates every 5th step (WHY: 600 frames would make the browser lag)
with play/pause and a time slider. Two toggles change how it is drawn:
  Evaluation view  also shows TRUE positions (ground truth, rule R1). Only
                   the UI may use truth, and only under this label.
  Offline map      the same data on a plain metric grid (no map tiles),
                   for presenting without internet.
"""

from __future__ import annotations

import streamlit as st

from analysis.propagation import NEAR_COLLISION_M, min_separation
from attacks.scenarios import get_scenario, scenario_keys
from ui import components as c
from ui import plots
from ui.state import QUICK_START, run_scenario, verdict_strings

LEGEND = (
    ("normal", "var(--sg-status-normal)"),
    ("suspicious", "var(--sg-status-suspicious)"),
    ("compromised", "var(--sg-status-alarm)"),
    ("insufficient evidence", "var(--sg-status-no-evidence)"),
    ("trail and formation links", "var(--sg-mint)"),
)


def controls() -> tuple[bool, bool, bool]:
    """Scenario, seed and display toggles. Runs the scenario when Run is pressed.

    Returns (evaluation, offline, links).
    """
    keys = scenario_keys()
    current = st.session_state.get("scenario_key")
    default = keys.index(current) if current in keys else keys.index(QUICK_START)
    left, mid, right = st.columns([2, 1, 1])
    with left:
        key = st.selectbox("Scenario", keys, index=default, key="sim-scenario",
                           format_func=lambda k: get_scenario(k).title)
    with mid:
        seed = int(st.number_input("Seed", min_value=0, max_value=1_000_000,
                                   value=int(st.session_state.get("seed") or 0), step=1, key="sim-seed"))
    with right:
        st.html('<div style="height:28px"></div>')
        if st.button("Run simulation", type="primary", key="sim-run", width="stretch"):
            with st.spinner("Simulating..."):
                run_scenario(key, seed=seed)
    t1, t2, t3 = st.columns(3)
    evaluation = t1.toggle("Evaluation view (ground truth)", key="sim-eval")
    offline = t2.toggle("Offline map", key="sim-offline")
    links = t3.toggle("Formation links", value=True, key="sim-links")
    st.caption(get_scenario(key).description)
    return evaluation, offline, links


def main() -> None:
    """Draw the page."""
    c.page_header("Swarm", "simulator", eyebrow="Operations",
                  caption="Pick a preset flight, run it, and play it back. Simulation result.")
    evaluation, offline, links = controls()
    pipe = st.session_state.get("pipeline")
    if pipe is None:
        c.empty_state("Choose a scenario above and press Run simulation.", sticker_name="drone_side")
        return
    result = pipe.sim_result
    live = "Evaluation view: ground truth" if evaluation else f"{result.n_drones} drones"
    light = st.session_state.get("scenario_key") == "light_show_pune"
    c.console_panel(f"Flight · {st.session_state.get('scenario_key')} · seed {result.config.seed}",
                    lambda: c.chart(plots.swarm_map(result, verdict_strings(pipe), links=links,
                                                    evaluation=evaluation, offline=offline,
                                                    light_show=light, height=540), key="sim-map"),
                    key="sim-map", live=live)
    c.html_block(c.legend_html(LEGEND))
    c.explain_box("Each dot is one drone, coloured by its **fused verdict** and placed at its reported "
                  "GPS position. Press play to run the flight. In the *Evaluation view* hollow markers "
                  "show where the drones truly are, joined to their GPS by dashed lines; that view uses "
                  "simulator ground truth, which no detector ever sees.")

    c.section("Closest approach", "Smallest TRUE distance between any two drones. Evaluation: "
              "uses ground truth.", eyebrow="Safety")
    series, events = min_separation(result)
    c.chart(plots.min_separation_chart(result.times(), series, NEAR_COLLISION_M), key="sim-sep")
    if events:
        st.warning(f"{len(events)} near-collision event(s) below {NEAR_COLLISION_M:g} m in this run "
                   "(simulation result).")


main()
