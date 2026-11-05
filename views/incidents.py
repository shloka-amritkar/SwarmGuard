"""Incidents: every run in which a detector alarmed, with its explanation.

Incidents are opened automatically (Phase 4, open_incident) whenever any
detector raises an ALARM. The alarm timeline is rebuilt by re-running the
stored configs: the simulation is deterministic (R2), so the same configs
and seed reproduce the identical run.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core.utils import format_drones
from ui import components as c
from ui import plots
from ui.state import PATTERN_TITLES, get_storage, incident_table, pipeline_for_incident

SELECT_KEY = "incidents-select"


def filtered_rows() -> list[dict[str, Any]]:
    """Incident rows after the pattern / scenario filters."""
    rows = get_storage().list_incidents()
    a, b = st.columns(2)
    patterns = sorted({r["pattern"] for r in rows})
    scenarios = sorted({r["scenario"] for r in rows if r["scenario"]})
    pattern = a.selectbox("Agreement pattern", ["all", *patterns], key="inc-pattern")
    scenario = b.selectbox("Scenario", ["all", *scenarios], key="inc-scenario")
    return [r for r in rows if (pattern == "all" or r["pattern"] == pattern)
            and (scenario == "all" or r["scenario"] == scenario)]


def _num(value: Any, digits: int) -> str:
    """Number -> fixed decimals; missing -> "-" (never "None" on screen)."""
    return "-" if value is None else f"{float(value):.{digits}f}"


def detector_details(incident: dict[str, Any]) -> pd.DataFrame:
    """One row per detector from the stored incident."""
    rows = []
    for short, d in sorted(incident["detected"].items()):
        rows.append({"detector": f"{short} · {d.get('name', short)}", "state": d["state"].replace("_", " "),
                     "first alarm (s)": _num(d.get("first_alarm_t"), 1),
                     "drones": format_drones(d["drones"]) if d.get("drones") else "-",
                     "peak score": _num(d.get("peak_score"), 2), "threshold": _num(d.get("threshold"), 2)})
    return pd.DataFrame(rows)


def case_card(incident: dict[str, Any], stamp: str | None = "filed") -> str:
    """The incident as a case-file card (M6): tab with the case number, FILED time, pills, stamp."""
    pills = [c.status_pill_html("alarm", "Incident")]
    pills += [c.detector_pill_html(s, f"{s} · alarm") for s, d in sorted(incident["detected"].items())
              if d["state"] == "alarm"]
    title = PATTERN_TITLES.get(incident["pattern"], "Detector alarm")
    detail = f"Scenario {incident['scenario_key']} · seed {incident['seed']} · simulation result"
    return c.case_card_html(incident["incident_id"], title, c.utc_label(incident["created_utc"]), pills,
                            detail=detail, stamp=stamp)


def show_incident(incident: dict[str, Any]) -> None:
    """Case-file card, alarm timeline, explanation and detector table for one incident."""
    c.html_block(case_card(incident))
    c.section("Alarm timeline", f"Onset estimate from the {incident['onset_source']}. Simulation result.")
    pipe = pipeline_for_incident(incident)
    c.chart(plots.alarm_timeline(pipe.all_runs(), incident.get("onset_estimate")), key="inc-timeline")
    c.section("What happened", incident["pattern_description"], eyebrow="Explanation")
    with c.paper_card("inc-explanation"):
        st.markdown("\n".join(f"- {s}" for s in incident["explanation"]))
    c.section("Detector details")
    st.dataframe(detector_details(incident), hide_index=True, width="stretch")
    if st.button("Open in the Evidence Centre", type="primary", key="inc-to-evidence"):
        st.session_state["evidence-select"] = incident["incident_id"]
        st.switch_page("views/evidence.py")


def comparison_rows(a: dict[str, Any], b: dict[str, Any]) -> pd.DataFrame:
    """Key fields of two incidents side by side."""
    def fields(i: dict[str, Any]) -> dict[str, str]:
        """The compared values of one incident, as text."""
        rec = i.get("recovery") or {}
        mag = rec.get("attack_magnitude")
        return {
            "scenario": i["scenario_key"], "seed": str(i["seed"]), "pattern": i["pattern"],
            "detectors alarmed": ", ".join(s for s, d in sorted(i["detected"].items()) if d["state"] == "alarm")
                                 or "none",
            "drones flagged": format_drones(i["affected_drones_detected"]) or "-",
            "onset estimate (s)": "-" if i["onset_estimate"] is None else f"{i['onset_estimate']:.1f}",
            "attack vector (m)": "-" if mag is None else f"{mag:.2f}",
        }
    fa, fb = fields(a), fields(b)
    return pd.DataFrame({"field": list(fa), a["incident_id"]: list(fa.values()), b["incident_id"]: list(fb.values())})


def compare_section(ids: list[str]) -> None:
    """Pick two incidents and compare them."""
    c.section("Compare two incidents", eyebrow="Side by side")
    if len(ids) < 2:
        c.html_block('<p class="sg-card__body">At least two incidents are needed to compare.</p>')
        return
    a, b = st.columns(2, gap="large")
    first = a.selectbox("First", ids, index=0, key="inc-cmp-a")
    second = b.selectbox("Second", ids, index=1, key="inc-cmp-b")
    storage = get_storage()
    one, two = storage.get_incident(first), storage.get_incident(second)
    # WHY no stamps here: two stamps side by side would crowd the comparison.
    a.html(case_card(one, stamp=None))
    b.html(case_card(two, stamp=None))
    st.dataframe(comparison_rows(one, two), hide_index=True, width="stretch")


def main() -> None:
    """Draw the page."""
    c.page_header("Incident", "files", eyebrow="07 · Incidents",
                  caption="Opened automatically when any detector alarms. Simulation results.")
    if not get_storage().list_incidents():
        if c.empty_state("No incidents yet. Run a scenario with an attack and one opens automatically.",
                         "Open the Attack Lab", sticker_name="document", key="incidents"):
            st.switch_page("views/attack_lab.py")
        return
    rows = filtered_rows()
    st.dataframe(incident_table(rows), hide_index=True, width="stretch")
    ids = [r["id"] for r in rows]
    if not ids:
        st.info("No incident matches these filters.")
        return
    # WHY pop + write the widget's own key: another page (Attack Lab) hands over
    # an incident id once; afterwards the selectbox keeps the user's choice.
    handoff = st.session_state.pop(SELECT_KEY, None)
    if handoff in ids:
        st.session_state["inc-pick"] = handoff
    chosen = st.selectbox("Incident", ids, key="inc-pick")
    show_incident(get_storage().get_incident(chosen))
    compare_section(ids)


main()
