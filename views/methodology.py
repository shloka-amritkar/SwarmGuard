"""Methodology: every equation the platform uses, its assumptions, fixes, decisions and credits.

The equations are written to match the code exactly (each section names
the file it comes from), so this page doubles as a viva reference.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from attacks.scenarios import list_scenarios
from ui import components as c

# (title, source file, LaTeX, one-sentence explanation)
EQUATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("Pairwise residual", "detection/detector_a_geometry.py",
     r"R_{ij} = \bigl|\, \lVert g_i - g_j \rVert - r_{ij} \,\bigr|",
     "GPS distance between two drones minus their radio range; large when one of them is spoofed."),
    ("Majority vote", "detection/detector_a_geometry.py",
     r"\text{score}_i = \frac{\#\{j : R_{ij} > \tau_a\}}{\#\{j : r_{ij}\ \text{valid}\}}, \qquad "
     r"\text{flag}_i = \text{score}_i > 0.5",
     "A spoofed drone disagrees with nearly every partner; an honest drone only with the liar."),
    ("Kalman predict", "detection/detector_b_motion.py",
     r"\hat{x}^- = \hat{x} + u\,\Delta t, \qquad P^- = P + Q",
     "Where the drone should be, from its own velocity sensor u (independent of GPS)."),
    ("Kalman update", "detection/detector_b_motion.py",
     r"y = z - \hat{x}^-,\quad S = P^- + R,\quad K = P^- S^{-1},\quad \hat{x} = \hat{x}^- + K y",
     "Blend the prediction with the GPS fix z, weighted by their uncertainties."),
    ("Normalised innovation squared", "detection/detector_b_motion.py",
     r"\text{NIS} = y^\top S^{-1} y \;\sim\; \chi^2_2, \qquad P(\text{NIS} > 9.21) = 0.01",
     "How surprising this GPS fix is; for clean data it follows a chi-square law with 2 degrees of freedom."),
    ("CUSUM", "detection/detector_b_motion.py",
     r"S_t = \max\bigl(0,\; S_{t-1} + \text{NIS}_t - (2 + k)\bigr), \qquad \text{alarm if } S_t > h",
     "Adds up small surprises, so a slow drift that never trips the NIS test still builds up."),
    ("Anchor residual", "detection/detector_c_anchor.py",
     r"r_i = \lVert g_i - a_i \rVert, \qquad \text{median}(9.8,\,10.1,\,10.2,\,100) = 10.15",
     "Direct absolute error; the swarm-level median ignores one faulty anchor (the mean would be 32.5)."),
    ("Dempster's rule", "detection/fusion.py",
     r"K = m_{1a}m_{2n} + m_{1n}m_{2a}, \qquad a = \frac{m_{1a}m_{2a} + m_{1a}m_{2u} + m_{1u}m_{2a}}{1 - K}",
     "Combines two detectors' evidence while keeping 'I don't know' separate from '50/50'."),
    ("MDS double centering", "recovery/mds_alignment.py",
     r"B = -\tfrac{1}{2} J D^{(2)} J,\quad J = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top,\quad "
     r"X = V_2 \Lambda_2^{1/2}",
     "Rebuilds the formation's shape from radio ranges alone (up to a shift, rotation and mirror)."),
    ("Kabsch alignment", "recovery/mds_alignment.py",
     r"H = \tilde{A}^\top \tilde{B} = U \Sigma V^\top, \qquad R = V\,\mathrm{diag}(1, d)\,U^\top,\ "
     r"d = \det(V U^\top)",
     "The best rotation that lays the rebuilt shape onto the anchors; RANSAC drops anchors that do not fit."),
    ("Multilateration", "recovery/multilateration.py",
     r"\hat{p}_i = \arg\min_p \sum_{j \in \mathcal{T}_i} \bigl(\lVert p - p_j \rVert - r_{ij}\bigr)^2",
     "Repairs one drone from trusted (never flagged) neighbours; needs at least three of them."),
    ("Experimental ML detector (D)", "detection/detector_d_ml.py",
     r"P_i = rac{1}{T}\sum_{t=1}^{T} p_t(	ext{attacked} \mid x_i), \qquad 	ext{raw flag}_i = P_i > 0.5",
     "Average vote of T = 200 trees on 11 features built from A, B, C and GPS motion; trained on "
     "simulated runs only, split by run (F6). Experimental, never part of fusion."),
    ("Wilson interval", "core/utils.py",
     r"\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}",
     "A 95 % interval for a rate that stays sensible near 0 % and 100 % and for few runs."),
)

ASSUMPTIONS = (
    "Flat 2D world in local East-North metres around Pune (18.5204 N, 73.8567 E).",
    "Radio ranging and velocity sensors cannot be affected by the GPS spoof.",
    "Anchors are absolute fixes; a configurable share of them can be faulty.",
    "Sensor noise is Gaussian with known standard deviations.",
    "Detectors, fusion and recovery see only observations, never the simulator's truth (R1).",
)

FIXES = (
    ("F1", "Multilateration never uses already-flagged neighbours."),
    ("F2", "Range noise is actually stored in the range matrix."),
    ("F3", "Thresholds are never silently overwritten."),
    ("F4", "Robust RANSAC alignment is the default in scoring."),
    ("F5", "ML labels stay 1 for the whole attack window."),
    ("F6", "ML train/test split by run, never by row."),
    ("F7", "Kalman filter works in metres and actually uses the velocity input."),
    ("F8", "Detector C returns NO_EVIDENCE with zero anchors, not a 'clean' 0."),
)

DECISIONS = (
    ("D1", "centre_gain (default 0.5): a path term in the controller so a swarm-wide spoof drags the "
           "true swarm; propagation is reported at both 0 and 0.5."),
    ("D2", "drift_cap: a drifting attack can stop growing once it reaches its magnitude."),
    ("D3", "custom_points: custom formations use the given offsets exactly, one per drone."),
    ("D4", "Velocity timing: vel_meas[k] is the velocity flown during [t_k, t_k + dt]."),
)

CREDITS = (
    ("RigidShift", "Minseok Park, Joon Soo Yoo", "rigid-covert GNSS spoofing, anchor consensus, MDS + rigid alignment"),
    ("SwarmFuzz", "Elaine Yao, Pritam Dash, Karthik Pattabiraman (IEEE/IFIP DSN 2023)",
     "swarm propagation vulnerabilities, graph + PageRank seeding"),
    ("SwarmRaft", "Kapel Dev, Yash Madhwal, Sofia Shevelo, Pavel Osinenko, Yury Yanovich",
     "neighbour voting, multilateration recovery"),
    ("GNSS Guardian", "GitHub user szyraf", "feature engineering, Random Forest, Streamlit dashboard"),
    ("UAV Intelligence Fusion System", "Arzoo Tiwary", "Dempster-Shafer fusion, template reports, modular pipeline"),
    ("ZYRNAV Resilience Lab", "Chris Matheus Rozario", "scenario presets, decision inspector UI pattern"),
    ("Wispr Flow", "visual design reference", "ivory canvas, serif headlines, pill navbar, flowing motion"),
    ("Urban Macaw", "visual design reference (style only)", "tactile outlined buttons, hard shadows, bold numerals"),
)

LIMITATIONS = (
    "Everything is simulated: no real radio signals, receivers or drones are involved.",
    "Detectors A and B both read the same GPS, so Dempster's independence assumption only holds approximately.",
    "If most anchors are faulty, the median and RANSAC can both be fooled.",
    "A valid signature proves the evidence is unchanged since signing, not that the telemetry was genuine.",
    "The signing key is stored unencrypted on disk; production use would require a protected key "
    "(passphrase or hardware module).",
    "The platform never identifies an attacker.",
    "Detector D (experimental ML) learns only from this simulator's attacks and noise, so its scores "
    "say nothing about real drones; its inputs are A, B and C, so it is not an independent witness.",
    "Streamlit reruns the page on every interaction, so there are no animated page transitions.",
)


def equations() -> None:
    """Every equation in a paper card, with its source file and one sentence."""
    c.section("Equations", "Each one matches the code in the file named under it.", eyebrow="Mathematics")
    for i in range(0, len(EQUATIONS), 2):
        cols = st.columns(2)
        for col, (title, source, latex, note) in zip(cols, EQUATIONS[i:i + 2]):
            with col, c.paper_card(f"eq-{i}-{title}"):
                c.html_block(f'<p class="sg-card__title">{title}</p><p class="sg-hash">{source}</p>')
                st.latex(latex)
                c.html_block(f'<p class="sg-equation-note">{note}</p>')


def blind_spot_table() -> pd.DataFrame:
    """Scenario x detector expectations, taken from the scenario presets themselves."""
    rows = []
    for s in list_scenarios():
        if not s.expected:
            continue
        rows.append({"scenario": s.title, "attack": s.attack.type.replace("_", " "),
                     **{f"{d}": s.expected.get(d, "-") for d in ("A", "B", "C")}})
    return pd.DataFrame(rows)


def lists() -> None:
    """Assumptions, blind spots, fixes, decisions, credits, limitations."""
    c.section("Assumptions")
    st.markdown("\n".join(f"- {a}" for a in ASSUMPTIONS))
    c.section("Blind-spot table", "What each detector is expected to do per preset attack: detect, "
              "blind (mathematically cannot see it) or partial (depends on anchors / persistence).")
    st.dataframe(blind_spot_table(), hide_index=True, width="stretch")
    # WHY lists, not tables: these are sentences, and a table cell cuts them off.
    left, right = st.columns(2)
    with left:
        c.section("Fixes over prior work")
        st.markdown("\n".join(f"- **{k}** {v}" for k, v in FIXES))
    with right:
        c.section("Design decisions")
        st.markdown("\n".join(f"- **{k}** {v}" for k, v in DECISIONS))
    c.section("Credits", "Concepts only; all code was written independently.")
    st.dataframe(pd.DataFrame(CREDITS, columns=["work", "authors", "concepts used"]), hide_index=True,
                 width="stretch")
    c.section("Limitations")
    st.markdown("\n".join(f"- {x}" for x in LIMITATIONS))


def main() -> None:
    """Draw the page."""
    c.page_header("How it", "works", eyebrow="Reference",
                  caption="The mathematics, assumptions and honest limits behind every result.")
    equations()
    lists()


main()
