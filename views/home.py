"""Home: radar hero, what the platform does, why one detector is not enough, quick starts.

Layout (docs/phases/phase-7.md B8, B9):
  hero (iframe) -> real CTA buttons + project stats -> hash ribbon
  -> Detect / Recover / Prove cards -> the cancellation equation
  -> architecture diagram -> scenario cards with crosshair corners.
Survey-mark dividers (§ 01, § 02, ...) separate the sections.

WHY real buttons under the hero: the hero lives in an iframe, which cannot
press Streamlit buttons.
"""

from __future__ import annotations

import html

import streamlit as st

from attacks.scenarios import get_scenario
from ui import components as c
from ui.hero import render_hero, stats_line
from ui.plots import architecture_dot
from ui.state import QUICK_START, run_scenario

DASHBOARD = "views/dashboard.py"

FEATURES: tuple[tuple[str, str, str], ...] = (
    ("radar_dish", "Detect",
     "Three independent checks (geometry, motion, anchors) look at the same flight. "
     "Where one is blind, another can still see."),
    ("survey_tripod", "Recover",
     "Radio ranges and trusted anchors rebuild where the drones really are, "
     "without trusting GPS at all."),
    ("sealed_document", "Prove",
     "Every incident becomes a hashed, Ed25519-signed evidence package that "
     "anyone can verify later."),
)

# (scenario key, sticker) for the four quick-start cards
QUICK_CARDS: tuple[tuple[str, str], ...] = (
    ("single_hijack", "crosshair"),
    ("slow_drift", "constellation"),
    ("rigid_swarm_shift", "satellite"),
    ("light_show_pune", "quadcopter_top"),
)


def run_and_open(scenario_key: str) -> None:
    """Run a scenario (cached), then open the Dashboard on it."""
    with st.spinner("Simulating, detecting, fusing and recovering..."):
        run_scenario(scenario_key)
    st.switch_page(DASHBOARD)


def quick_start_buttons() -> None:
    """The real call-to-action buttons directly under the hero, then the project stats.

    "Run a scenario" runs the Pune light show (the quick start) and opens the
    Dashboard; "How it works" opens the Methodology page.
    """
    with st.container(key="sg-quickstart"):
        left, right, _ = st.columns([1, 1, 3])
        with left:
            if st.button("Run a scenario", type="primary", key="cta-quick", width="stretch"):
                run_and_open(QUICK_START)
        with right:
            if st.button("How it works", key="cta-how", width="stretch"):
                st.switch_page("views/methodology.py")
        c.html_block(f'<p class="sg-stats">{html.escape(stats_line())} · simulation results</p>')


def feature_cards() -> None:
    """Detect / Recover / Prove."""
    c.html_block(c.survey_divider_html(1))
    cols = st.columns(3, gap="large")
    for col, (name, title, body) in zip(cols, FEATURES):
        with col:
            c.html_block(c.feature_card_html(name, title, body))


def why_section() -> None:
    """The central scientific idea: a common offset cancels in every pairwise distance."""
    c.html_block(c.survey_divider_html(2))
    c.section("Why one detector isn't enough", eyebrow="The blind spot", anchor="sg-how")
    with c.paper_card("equation"):
        st.latex(r"(g_i + D) - (g_j + D) \;=\; g_i - g_j")
        c.html_block('<p class="sg-equation-note">If every drone\'s GPS is shifted by the same '
                     "offset D, every distance between drones stays the same. Detector A, which "
                     "compares those distances with radio ranges, cannot see the attack; motion "
                     "(B) and anchor (C) checks still can.</p>")


def architecture() -> None:
    """The processing pipeline as a Graphviz diagram on a paper card."""
    c.html_block(c.survey_divider_html(3))
    c.section("How a flight is analysed", "Detectors, fusion and recovery only ever see "
              "observations, never the simulator's ground truth.")
    with c.paper_card("architecture"):
        st.graphviz_chart(architecture_dot(), width="stretch")


def scenario_cards() -> None:
    """Four preset scenarios; each card runs its scenario and opens the Dashboard."""
    c.html_block(c.survey_divider_html(4))
    c.section("Try a scenario", "Each card runs the full pipeline on a preset attack. "
              "The pills show what each detector is expected to do.", eyebrow="Quick start")
    cols = st.columns(len(QUICK_CARDS), gap="medium")
    for col, (key, sticker_name) in zip(cols, QUICK_CARDS):
        scenario = get_scenario(key)
        with col:
            if c.scenario_card(key, scenario.title, scenario.description, scenario.expected, sticker_name):
                run_and_open(key)


render_hero()
quick_start_buttons()
c.hash_ribbon()
feature_cards()
why_section()
architecture()
scenario_cards()
