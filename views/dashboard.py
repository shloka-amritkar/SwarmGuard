"""Dashboard: the current run at a chosen moment in time.

One time slider drives everything on the page: the KPIs, the map, the drone
table and the event feed all show the swarm as it was at time t. WHY one
slider: it lets a presenter scrub through the attack and watch the verdicts
change, and every panel always agrees with every other.
"""

from __future__ import annotations

import streamlit as st

from ui import components as c
from ui import plots
from ui.state import (build_events, drone_rows, get_storage, incident_table, pattern_banner_text, step_index,
                      verdict_counts, verdict_strings)


def kpis(counts: dict[str, int]) -> None:
    """Total, normal, suspicious, compromised, insufficient evidence."""
    c.kpi_row([
        c.KPI("Drones", counts["total"]),
        c.KPI("Normal", counts["normal"], "normal"),
        c.KPI("Suspicious", counts["suspicious"], "suspicious"),
        c.KPI("Compromised", counts["compromised"], "compromised"),
        c.KPI("No evidence", counts["insufficient_evidence"], "insufficient_evidence"),
    ])


def latest_incidents() -> None:
    """The five most recent incidents in the database."""
    c.section("Latest incidents", "Opened automatically when any detector alarms.")
    rows = get_storage().list_incidents()[:5]
    if not rows:
        c.html_block('<p class="sg-card__body">No incidents yet.</p>')
        return
    st.dataframe(incident_table(rows), hide_index=True, width="stretch")
    if st.button("Open the Incidents page", key="dash-incidents"):
        st.switch_page("views/incidents.py")


def main() -> None:
    """Draw the page."""
    c.page_header("Swarm", "dashboard", eyebrow="01 · Operations dashboard",
                  caption="Fused verdicts, live map and detection events for the current run. "
                          "Simulation result.")
    pipe = c.require_run("dashboard")
    if pipe is None:
        return
    times = pipe.sim_result.times()
    t = st.slider("Time (s)", float(times[0]), float(times[-1]), float(times[-1]), step=0.5,
                  key="dash-t", help="Every panel below shows the swarm at this moment.")
    k = step_index(pipe, t)
    kpis(verdict_counts(pipe, k))
    banner = pattern_banner_text(pipe)
    if banner is not None:
        c.pattern_banner(*banner)

    events = build_events(pipe, upto_t=t)
    light = st.session_state.get("scenario_key") == "light_show_pune"
    left, right = st.columns([1.7, 1])
    def live_map() -> None:
        # WHY the camera control lives inside the panel: it belongs to the map.
        # The time comes from the slider above, so switching camera keeps it.
        camera = c.camera_control("dash")
        c.chart(plots.swarm_map(pipe.sim_result, verdict_strings(pipe), step=k, light_show=light,
                                height=470, camera=camera), key=f"dash-map-{camera}")

    with left:
        c.console_panel("Live map · Pune", live_map, key="dash-map", live=f"t = {times[k]:.1f} s")
    with right:
        c.event_feed(events, key="dash-feed", limit=9)
    c.ribbon(events)
    c.hash_ribbon(slim=True)

    c.section("Drones", "Per-drone detector statuses and fused verdict at the chosen time.")
    c.drone_table(drone_rows(pipe, k), key="dash-table")
    c.explain_box("Each row is one drone. **A**, **B** and **C** are the three detectors' statuses; "
                  "the bars show how strongly each one disagrees with GPS. **Fused verdict** combines "
                  "them with Dempster-Shafer fusion; **belief** is the evidence that the drone is "
                  "compromised. *No anomaly detected* never proves a drone's GPS is genuine.")
    latest_incidents()


main()
