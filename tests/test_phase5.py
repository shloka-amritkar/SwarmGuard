"""Phase 5 tests: design tokens, stickers, hero, components, plots, state helpers.

Updated in Phase 7 for the "Ground Truth" design (new token values, sticker
stroke width, hero content and font names); the behaviour tested is the same.

These test the pure (non-Streamlit) parts. Whole pages are loaded with
Streamlit's AppTest in tests/test_phase5_views.py.
"""

from __future__ import annotations

import io
import re
import tomllib
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from analysis.calibration import load_or_default
from analysis.experiments import PipelineResult, run_pipeline
from analysis.propagation import min_separation
from assets.stickers import STICKER_NAMES, sticker
from attacks.scenarios import get_scenario
from core.palette import PALETTE
from database.storage import Storage
from forensics.evidence_package import Check
from forensics.incident import open_incident
from ui import plots
from ui.components import (
    KPI,
    event_feed_html,
    kpi_card_html,
    kpi_row_html,
    page_header_html,
    pattern_banner_html,
    ribbon_html,
    status_pill_html,
    verification_table_html,
    wordmark_html,
)
from ui.hero import hero_html
from ui.state import Event, build_events, drone_rows, inspector_data, step_index, verdict_counts
from ui.theme import css_variables, rgba, status_label, status_tone, stylesheet

ROOT = Path(__file__).resolve().parent.parent
HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")


@pytest.fixture(scope="module")
def shift_pipe() -> PipelineResult:
    """Rigid swarm shift (alarms on B and C)."""
    sc = get_scenario("rigid_swarm_shift", seed=5000)
    return run_pipeline(sc.sim, sc.attack, load_or_default(sc.sim), recovery_every=10)


@pytest.fixture(scope="module")
def clean_pipe() -> PipelineResult:
    """Clean baseline."""
    sc = get_scenario("clean_baseline", seed=5000)
    return run_pipeline(sc.sim, sc.attack, load_or_default(sc.sim), recovery_every=10)


# --------------------------------------------------------------------------
# One source of truth for colours
# --------------------------------------------------------------------------
def test_config_toml_matches_palette() -> None:
    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    assert theme["backgroundColor"] == PALETTE.sand
    assert theme["secondaryBackgroundColor"] == PALETTE.card
    assert theme["textColor"] == PALETTE.espresso
    assert theme["primaryColor"] == PALETTE.olive
    assert theme["redColor"] == PALETTE.status_alarm
    assert theme["dataframeBorderColor"] == PALETTE.rule


@pytest.mark.parametrize("path", ["assets/style.css", "assets/hero.html"])
def test_no_hard_coded_colours(path: str) -> None:
    """style.css and hero.html must take every colour from the generated variables."""
    text = (ROOT / path).read_text(encoding="utf-8")
    assert HEX.findall(text) == []


def test_css_variables_cover_the_palette() -> None:
    css = css_variables()
    for f in fields(PALETTE):
        assert f"--sg-{f.name.replace('_', '-')}: {getattr(PALETTE, f.name)};" in css
    assert "--sg-radius-card: 14px;" in css and "--sg-dur-press: 90ms;" in css
    sheet = stylesheet()
    assert sheet.startswith("@import url(")        # @import must come first
    assert "prefers-reduced-motion" in sheet


def test_every_css_variable_used_is_defined() -> None:
    sheet = stylesheet()
    defined = set(re.findall(r"(--sg-[a-z0-9-]+):", sheet))
    used = set(re.findall(r"var\((--sg-[a-z0-9-]+)", sheet))
    # --sg-pill-colour, --sg-target, --sg-ribbon-duration and --sg-delay are set per element.
    assert used - defined <= {"--sg-pill-colour", "--sg-target", "--sg-ribbon-duration", "--sg-delay"}


def test_status_helpers() -> None:
    assert status_label("no_evidence") == "NO EVIDENCE"
    assert status_tone("compromised") == "alarm"
    assert status_tone("insufficient_evidence") == "none"
    assert rgba("#0B4D40", 0.2) == "rgba(11,77,64,0.2)"
    with pytest.raises(ValueError):
        rgba("#0B4D40", 1.5)


# --------------------------------------------------------------------------
# Stickers and hero
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", STICKER_NAMES)
def test_stickers_are_valid_svg(name: str) -> None:
    root = ET.fromstring(sticker(name, size=48))
    assert root.tag.endswith("svg")
    assert root.get("aria-label")
    assert root.get("stroke") == PALETTE.espresso and root.get("stroke-width") == "1.5"


def test_sticker_validation() -> None:
    assert len(STICKER_NAMES) == 12
    with pytest.raises(ValueError):
        sticker("face")
    with pytest.raises(ValueError):
        sticker("radar", rotate=10)
    with pytest.raises(ValueError):
        sticker("radar", size=0)


def test_hero_is_complete() -> None:
    page = hero_html()
    assert re.findall(r"__[A-Z0-9_]+__", page) == []          # every placeholder filled
    for needed in ("prefers-reduced-motion", "pauseAnimations", "Where the drones", "really are.",
                   "ALARM · motion check", "18.530°N", "73.865°E", "spoof-dot", "true-dot", "48.6"):
        assert needed in page
    assert "--sg-olive:" in page


# --------------------------------------------------------------------------
# Component HTML builders
# --------------------------------------------------------------------------
def test_status_pill_always_has_text() -> None:
    pill = status_pill_html("compromised")
    assert "COMPROMISED" in pill and "sg-pill--alarm" in pill
    assert ">NO EVIDENCE<" in status_pill_html("no_evidence")


def test_builders_escape_text() -> None:
    nasty = "<script>alert(1)</script>"
    for markup in (page_header_html(nasty, caption=nasty), pattern_banner_html(nasty, nasty),
                   status_pill_html("normal", nasty)):
        assert "<script>" not in markup and "&lt;script&gt;" in markup


def test_kpi_count_up_and_static_values() -> None:
    counted = kpi_card_html("Compromised", 3, "compromised")
    assert "--sg-target:3" in counted and '<span class="sg-sr">3</span>' in counted
    assert "sg-kpi--alarm" in counted
    static = kpi_card_html("Offset", "10.1 m")
    assert "10.1 m" in static and "sg-count" not in static
    assert kpi_row_html([KPI("A", 1), KPI("B", 2)]).count("sg-kpi ") == 2


def test_feed_ribbon_and_verification_html() -> None:
    events = [Event(25.0, 3, "B", "alarm", "Drone 04 · motion consistency check · ALARM"),
              Event(20.0, None, "pattern", "alarm", "Swarm pattern")]
    feed = event_feed_html(events)
    assert feed.count("sg-feed__row") == 2 and "t = 25.0 s" in feed
    assert "no anomaly detected" in event_feed_html([])
    ribbon = ribbon_html(events)
    assert ribbon.count("Swarm pattern") == 2        # doubled for a seamless loop
    table = verification_table_html([Check("file a.csv", "PASS", "hash matches"),
                                     Check("signature", "FAIL", "signature does not match")])
    assert "sg-check" in table and 'class="sg-verify--fail"' in table and ">FAIL<" in table
    assert "SWARM" in wordmark_html() and "GUARD" in wordmark_html()


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def test_templates_registered() -> None:
    assert plots.LIGHT in pio.templates and plots.CONSOLE in pio.templates
    assert pio.templates[plots.CONSOLE].layout.paper_bgcolor == PALETTE.charcoal


def test_swarm_map_animation_and_modes(shift_pipe: PipelineResult) -> None:
    res = shift_pipe.sim_result
    verdicts = np.vectorize(lambda v: v.value)(shift_pipe.fusion.verdicts)
    fig = plots.swarm_map(res, verdicts)
    expected_frames = len(range(0, res.n_steps, plots.FRAME_EVERY)) + (0 if (res.n_steps - 1) % plots.FRAME_EVERY == 0 else 1)
    assert len(fig.frames) == expected_frames
    assert isinstance(fig.data[0], go.Scattermap) and fig.layout.map.style == plots.MAP_STYLE
    still = plots.swarm_map(res, verdicts, step=300, offline=True, evaluation=True)
    assert len(still.frames) == 0
    names = [t.name for t in still.data]
    assert "truth" in names and isinstance(still.data[0], go.Scatter)
    drones = [t for t in still.data if t.name == "drones"][0]
    assert "diamond" in drones.marker.symbol                       # anchored drones
    with pytest.raises(ValueError):
        plots.swarm_map(res, verdicts[:-1])


def test_analysis_figures_build(shift_pipe: PipelineResult) -> None:
    pipe = shift_pipe
    onset = pipe.sim_result.attack_onset()
    ts = plots.detector_timeseries(pipe, drone=2, onset=onset)
    assert len(ts.data) == 4 and ts.layout.yaxis2.type == "log"
    assert plots.residual_heatmap(pipe.sim_result.frames[400].observed, pipe.det_cfg.tau_a).data[0].type == "heatmap"
    assert len(plots.fusion_chart(pipe, 2).data) == 2
    assert len(plots.alarm_timeline(pipe.all_runs(), onset).data) > 0
    rec = next(r for r in reversed(pipe.recovery.results) if r is not None)
    obs = pipe.sim_result.frames[rec.step].observed
    fig = plots.recovery_map(obs, rec.positions, rec.inlier_mask, rec.outlier_anchors,
                             truth=pipe.sim_result.frames[rec.step].true_pos)
    assert {"reported GPS", "recovered", "true position (evaluation)"} <= {t.name for t in fig.data}
    sep, _ = min_separation(pipe.sim_result)
    assert len(plots.min_separation_chart(pipe.sim_result.times(), sep).data) >= 1
    for fig in (ts,):
        assert fig.layout.template.layout.font.family.startswith("'IBM Plex Sans'")


def test_experiment_figures_build() -> None:
    summary = pd.DataFrame({"detector": ["A", "B", "C", "F"], "value": [5.0] * 4,
                            "detection_rate": [0.0, 1.0, 0.9, 0.95],
                            "detection_ci_low": [0.0, 0.8, 0.7, 0.75], "detection_ci_high": [0.2, 1.0, 1.0, 1.0]})
    assert plots.comparison_bar(summary).data[0].error_y.array is not None
    assert len(plots.detection_curve(summary, "magnitude").data) == 8       # band + line per detector
    raw = pd.DataFrame({"detector": ["B", "B", "C"], "delay": [1.0, 2.0, np.nan]})
    assert len(plots.delay_box(raw).data) == 1
    env = pd.DataFrame({"detector": ["A", "B"] * 2, "param": ["magnitude"] * 4,
                        "target_rate": [0.5, 0.5, 0.9, 0.9], "threshold": [np.nan, 1.2, np.nan, 2.0]})
    bars = plots.stealth_envelope_chart(env)
    assert "not reliably detected" in bars.data[0].text
    m = np.array([[0.0, 0.3], [0.1, 0.0]])
    assert plots.propagation_heatmap(m).data[0].type == "heatmap"
    assert plots.pagerank_bar(np.array([0.4, 0.6])).data[0].x[0] == "Drone 02"
    curves = {"A": {"threshold": [4.3, 4.4], "fa_rate": [0.02, 0.01]}, "C": {"threshold": [3.4], "fa_rate": [0.0]}}
    assert len(plots.calibration_curve(curves, 0.01, {"A": 4.4, "C": 3.4}).data) == 2
    assert plots.architecture_dot().startswith("digraph")


# --------------------------------------------------------------------------
# State helpers (pure)
# --------------------------------------------------------------------------
def test_events_newest_first_and_honest(shift_pipe: PipelineResult, clean_pipe: PipelineResult) -> None:
    events = build_events(shift_pipe)
    assert events and all(a.t >= b.t for a, b in zip(events, events[1:]))
    assert any(e.detector == "pattern" for e in events)
    assert not any(e.detector == "A" for e in events)          # A is blind to the rigid shift
    assert build_events(shift_pipe, upto_t=1.0) == []
    if not any(r.flags().any() for r in clean_pipe.runs):
        assert build_events(clean_pipe) == []


def test_counts_rows_and_inspector(shift_pipe: PipelineResult) -> None:
    k = step_index(shift_pipe, 40.0)
    assert abs(shift_pipe.sim_result.times()[k] - 40.0) < 0.11
    counts = verdict_counts(shift_pipe, k)
    assert counts["total"] == shift_pipe.sim_result.n_drones
    assert sum(v for key, v in counts.items() if key != "total") == counts["total"]
    rows = drone_rows(shift_pipe, k)
    assert list(rows.columns) == ["drone", "A", "B", "C", "A_score", "B_cusum", "B_nis", "C_residual_m",
                                  "verdict", "belief"]
    data = inspector_data(shift_pipe, 2, k)
    assert len(data["detectors"]) == 3 and len(data["reasons"]) == 3 and data["action"]
    with pytest.raises(ValueError):
        inspector_data(shift_pipe, 99, k)


def test_open_incident_reuses_same_run(tmp_path: Path, shift_pipe: PipelineResult) -> None:
    store = Storage(tmp_path / "sg.db")
    first, new1 = open_incident(shift_pipe, "rigid_swarm_shift", store)
    second, new2 = open_incident(shift_pipe, "rigid_swarm_shift", store)
    assert new1 and not new2 and first.incident_id == second.incident_id
    assert len(store.list_incidents()) == 1 and first.run_key


def test_cli_investigate_does_not_duplicate(tmp_path: Path) -> None:
    from cli import main
    args = ["investigate", "--scenario", "single_hijack", "--seed", "5000", "--db", str(tmp_path / "sg.db"),
            "--keys", str(tmp_path / "keys"), "--out", str(tmp_path / "packages")]
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(args) == 0
        assert main(args) == 0
    text = out.getvalue()
    assert "opened" in text and "already investigated" in text
    assert len(list((tmp_path / "packages").glob("*.zip"))) == 1


# --------------------------------------------------------------------------
# Step 2: view helpers and every view through the real app (AppTest)
# --------------------------------------------------------------------------
def test_view_helpers(shift_pipe: PipelineResult, clean_pipe: PipelineResult) -> None:
    from ui.state import (affected_text, agreement_matrix, comparison_table, detection_results_table,
                          expected_vs_actual, pattern_banner_text, verdict_strings)

    assert verdict_strings(shift_pipe).shape == (shift_pipe.sim_result.n_steps, shift_pipe.sim_result.n_drones)
    assert pattern_banner_text(shift_pipe)[0] == "Coordinated swarm shift suspected"
    assert pattern_banner_text(clean_pipe) is None
    exp = expected_vs_actual(get_scenario("rigid_swarm_shift").expected, shift_pipe)
    assert set(exp["check"]) == {"as expected"}          # A blind, B and C alarm
    assert list(detection_results_table(shift_pipe)["detector"].str[0]) == ["A", "B", "C", "F"]
    matrix = agreement_matrix(shift_pipe)
    assert set(matrix.columns) == {"drone", "A", "B", "C", "F"}
    assert set(matrix["A"]) == {"silent"}                 # the blind spot, as text
    assert len(comparison_table(shift_pipe)) == 3
    assert affected_text("3,6") == "Drone 04, Drone 07" and affected_text("") == "-"


def test_attack_vector_chart_validates() -> None:
    times = np.arange(5.0)
    fig = plots.attack_vector_chart(times, np.zeros((5, 2)))
    assert fig.layout.template.layout.font.family is not None
    with pytest.raises(ValueError):
        plots.attack_vector_chart(times, np.zeros((4, 2)))


VIEWS = ("home", "dashboard", "simulator", "attack_lab", "detection", "experiments", "recovery",
         "incidents", "evidence", "methodology")
NEEDS_DATA = {"dashboard", "detection", "recovery", "incidents", "evidence"}


@pytest.fixture()
def fresh_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real app.py on an empty temporary database, key folder and package folder."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SWARMGUARD_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("SWARMGUARD_KEYS", str(tmp_path / "keys"))
    monkeypatch.setenv("SWARMGUARD_PACKAGES", str(tmp_path / "packages"))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    return at


def test_every_view_loads_on_a_fresh_database(fresh_app) -> None:
    for view in VIEWS:
        fresh_app.switch_page(f"views/{view}.py").run()
        assert not fresh_app.exception, f"{view}: {[e.value for e in fresh_app.exception]}"
        has_empty = any(b.key == f"sg-empty-{view}" for b in fresh_app.button)
        assert has_empty == (view in NEEDS_DATA), f"{view}: empty state shown = {has_empty}"


def test_quick_start_then_every_view_and_evidence(fresh_app) -> None:
    at = fresh_app
    at.button(key="cta-quick").click().run()
    assert not at.exception and at.session_state["incident_id"]
    for view in VIEWS:
        at.switch_page(f"views/{view}.py").run()
        assert not at.exception, f"{view}: {[e.value for e in at.exception]}"
    at.switch_page("views/evidence.py").run()
    at.button(key="ev-build").click().run()
    at.button(key="ev-verify-btn").click().run()
    assert at.session_state["ev-verify"][at.session_state["ev-pick"]].overall == "PASS"
    at.button(key="ev-tamper-btn").click().run()
    result = at.session_state["ev-tamper"][at.session_state["ev-pick"]]
    assert (result.before.overall, result.after.overall) == ("PASS", "FAIL")


def test_attack_lab_invalid_settings_are_friendly(fresh_app) -> None:
    at = fresh_app
    at.switch_page("views/attack_lab.py").run()
    at.selectbox(key="lab-preset").set_value("single_hijack").run()
    at.selectbox(key="lab-single_hijack-type").set_value("multi_drone").run()
    at.multiselect(key="lab-single_hijack-tn").set_value([]).run()
    assert not at.exception
    assert any("cannot be simulated" in e.value for e in at.error)


# --------------------------------------------------------------------------
# Step 3: regressions found while checking the app in a real browser
# --------------------------------------------------------------------------
def test_graphics_reach_st_html_as_images_not_inline_svg() -> None:
    """st.html strips <svg>; every graphic must arrive as an <img> with an SVG data URI."""
    from ui.components import empty_state_html, feature_card_html, navbar_html

    checks = [Check("manifest", "PASS", "ok"), Check("file a.csv", "FAIL", "hash differs"),
              Check("chain", "UNKNOWN", "-")]
    for markup in (feature_card_html("radar", "Detect", "body"), empty_state_html("Nothing yet."),
                   verification_table_html(checks), wordmark_html(), navbar_html("Home", None, None, None)):
        assert "<svg" not in markup
        assert 'src="data:image/svg+xml;base64,' in markup
    table = verification_table_html(checks)
    assert "sg-check" in table and "sg-cross" in table and "sg-dash" in table


def test_svg_img_validates_and_escapes() -> None:
    from assets.stickers import svg_img, sticker_img

    img = sticker_img("radar", size=40, label='a "quoted" label')
    assert img.startswith("<img") and 'alt="a &quot;quoted&quot; label"' in img
    assert 'aria-hidden="true"' in svg_img("<svg/>", 10)      # decorative icons are hidden
    with pytest.raises(ValueError):
        svg_img("<svg/>", 0)


@pytest.mark.parametrize("extent", [30.0, 115.0, 600.0])
def test_map_zoom_fits_the_flight(extent: float) -> None:
    """MapLibre (512 px tiles): metres per pixel = 78271.52 cos(lat) / 2^zoom."""
    import math

    zoom = plots._zoom_for(extent, plots.LAT0, pixels=400)
    mpp = 78271.52 * math.cos(math.radians(plots.LAT0)) / 2 ** zoom
    assert extent / mpp <= 400                        # the whole flight is on screen
    assert extent * 1.3 / mpp == pytest.approx(400, rel=1e-6)   # with a 30 % margin, no more
    with pytest.raises(ValueError):
        plots._zoom_for(extent, plots.LAT0, pixels=0)
