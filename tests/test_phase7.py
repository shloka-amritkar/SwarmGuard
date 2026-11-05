"""Phase 7 tests: map interaction and camera, the Ground Truth motifs, and the banned-element sweep.

The map tests check the maths of the camera (follow = GPS centroid, whole
flight = everything inside the padded bounds). The motif tests check the
pure HTML/SVG builders. The sweep makes sure no retired Phase 5 design
element (fonts, colours, curved text, waveform) creeps back into the code.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from attacks.scenarios import get_scenario
from core.palette import PALETTE
from core.utils import enu_to_latlon
from database.storage import Storage
from simulation.swarm import simulate
from ui import plots
from ui.components import (
    MAP_PLOTLY_CONFIG,
    PLOTLY_CONFIG,
    case_card_html,
    clock_html,
    custody_timeline_html,
    hash_ribbon_html,
    legend_html,
    navbar_html,
    short_hash,
    stamp_html,
    stamp_svg,
    status_pill_html,
    survey_divider_html,
    utc_label,
)
from ui.hero import blip_positions, bearing_of, hero_html, polar, ruler_html, stats_line
from ui.state import RibbonEntry
from ui.theme import contour_background, corner_marks

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def hijack():
    """single_hijack simulation result plus all-normal statuses."""
    sc = get_scenario("single_hijack")
    res = simulate(sc.sim, sc.attack)
    return res, np.full((res.n_steps, res.n_drones), "normal", dtype=object)


# --------------------------------------------------------------------------
# Part A: map interaction and camera
# --------------------------------------------------------------------------
def test_maps_pan_when_dragged(hijack) -> None:
    res, statuses = hijack
    for kwargs in ({}, {"camera": "whole"}, {"offline": True}, {"step": 100}):
        assert plots.swarm_map(res, statuses, **kwargs).layout.dragmode == "pan"
    obs = res.frames[100].observed
    assert plots.recovery_map(obs, None, None, []).layout.dragmode == "pan"


def test_map_config_scroll_zooms_and_has_no_box_tools() -> None:
    assert MAP_PLOTLY_CONFIG["scrollZoom"] is True and MAP_PLOTLY_CONFIG["displaylogo"] is False
    for tool in ("select2d", "lasso2d", "zoom2d"):
        assert tool in MAP_PLOTLY_CONFIG["modeBarButtonsToRemove"]
    assert "scrollZoom" not in PLOTLY_CONFIG       # ordinary charts must not hijack page scrolling


def test_follow_camera_centres_every_frame_on_the_gps_centroid(hijack) -> None:
    res, statuses = hijack
    fig = plots.swarm_map(res, statuses, camera="follow")
    gps = res.stack("gps_pos")
    times = res.times()
    for frame in fig.frames[::10]:
        k = int(np.argmin(np.abs(times - float(frame.name))))
        lat, lon = enu_to_latlon(*gps[k].mean(axis=0))
        assert frame.layout.map.center.lat == pytest.approx(float(lat), abs=1e-9)
        assert frame.layout.map.center.lon == pytest.approx(float(lon), abs=1e-9)


def test_follow_zoom_fills_about_half_and_keeps_every_drone_in_view(hijack) -> None:
    res, _ = hijack
    gps = res.stack("gps_pos")
    centres, zoom = plots.camera_follow(gps, pixels=340, fill=0.5)
    view_m = 340 * plots.metres_per_pixel(zoom)
    radius = np.linalg.norm(gps - centres[:, None, :], axis=-1).max(axis=1)
    assert 2 * np.median(radius) / view_m <= 0.5 + 1e-9          # never closer than "half the view"
    assert radius.max() < view_m / 2                             # no drone ever leaves the view


def test_whole_flight_camera_contains_every_point(hijack) -> None:
    res, _ = hijack
    points = res.stack("gps_pos").reshape(-1, 2)
    centre, zoom = plots.camera_whole(points, pixels=400, padding=0.15)
    half = 400 * plots.metres_per_pixel(zoom) / 2
    assert np.all(np.abs(points - centre) <= half + 1e-6)
    extent = float(np.ptp(points, axis=0).max())
    assert extent * 1.15 == pytest.approx(2 * half, rel=1e-6) or extent < 30.0


def test_static_maps_default_to_whole_flight_and_bad_camera_fails(hijack) -> None:
    res, statuses = hijack
    still = plots.swarm_map(res, statuses, step=50)
    whole = plots.swarm_map(res, statuses, step=50, camera="whole")
    assert still.layout.map.center == whole.layout.map.center
    assert plots.swarm_map(res, statuses, camera="whole").frames[0].layout.map.center.lat is None
    with pytest.raises(ValueError):
        plots.swarm_map(res, statuses, camera="orbit")


def test_map_layers_rings_diamonds_and_links(hijack) -> None:
    res, statuses = hijack
    names = {t.name for t in plots.swarm_map(res, statuses, step=10).data}
    assert {"range rings", "anchor diamond", "links", "trails", "drones", "outline"} <= names
    xs, ys = plots._diamond_points(np.array([[0.0, 0.0]]), 2.0)
    assert xs[:5] == [0.0, 2.0, 0.0, -2.0, 0.0] and ys[:5] == [2.0, 0.0, -2.0, 0.0, 2.0]


# --------------------------------------------------------------------------
# Part B: motifs
# --------------------------------------------------------------------------
def test_stamps_have_rough_ink_and_carry_the_case_number() -> None:
    svg = stamp_svg("verified", "SG-20261105-001", "2026-11-05 17:40 UTC")
    assert "feTurbulence" in svg and "feDisplacementMap" in svg and "VERIFIED" in svg
    assert "SG-20261105-001 · 2026-11-05 17:40 UTC" in svg and PALETTE.olive in svg
    assert PALETTE.status_alarm in stamp_svg("tampered", "SG-1")      # brick only for tampering
    img = stamp_html("filed", "SG-<x>")
    assert img.startswith("<img") and "sg-stamp--filed" in img and "<svg" not in img
    with pytest.raises(ValueError):
        stamp_svg("approved", "SG-1")


def test_hash_ribbon_uses_given_hashes_and_loops() -> None:
    digest = "9f2c" + "0" * 56 + "e41a"
    assert short_hash(digest) == "9f2c…e41a"
    real = hash_ribbon_html([RibbonEntry("SG-20261105-001", digest, True)])
    assert real.count("SG-20261105-001") == 2 and "9f2c…e41a" in real and "config" not in real
    fallback = hash_ribbon_html([RibbonEntry("single_hijack", digest, False)], slim=True)
    assert "config" in fallback and "sg-hashribbon--slim" in fallback
    assert hash_ribbon_html([]) == ""


def test_storage_lists_newest_packages(tmp_path: Path) -> None:
    store = Storage(tmp_path / "sg.db")
    assert store.list_packages() == []
    with pytest.raises(ValueError):
        store.list_packages(0)


def test_clock_starts_at_server_time() -> None:
    clock = clock_html(datetime(2026, 11, 5, 10, 7, 9, tzinfo=timezone.utc))
    assert "--sg-delay:-36429s" in clock and "--sg-delay:-429s" in clock and "--sg-delay:-9s" in clock
    assert "10:07:09 UTC" in clock                      # screen-reader copy
    bar = navbar_html("Home", "alarm", "Fused alarm", "single_hijack · seed 0")
    assert "18.5204°N 73.8567°E" in bar and "SWARMGUARD" in bar and "FUSED ALARM" in bar.upper()


def test_case_card_divider_and_timeline_escape_text() -> None:
    card = case_card_html("SG-<1>", "Title <b>", "2026-11-05 17:40 UTC", [status_pill_html("alarm")])
    assert "SG-&lt;1&gt;" in card and "Title &lt;b&gt;" in card and "sg-case__tab" in card
    assert "sg-stamp--filed" in card and "sg-stamp" not in case_card_html("SG-1", "t", "x", stamp=None)
    assert "§ 03 · 18.5204°N" in survey_divider_html(3)
    timeline = custody_timeline_html([{"seq": 1, "action": "verify_failed", "created_utc": "t",
                                       "manifest_sha256": "a" * 64, "previous_sha256": "GENESIS"}])
    assert "VERIFY FAILED" in timeline and "sg-timeline__node" in timeline and "GENESIS" in timeline
    assert utc_label("2026-11-05T17:40:17+00:00") == "2026-11-05 17:40 UTC"
    assert utc_label("not a date") == "not a date"


def test_legend_shapes() -> None:
    html = legend_html([("anchor", "var(--sg-amber)", "diamond"), ("normal", "var(--sg-status-normal)")])
    assert "sg-legend__swatch--diamond" in html and "sg-legend__swatch--dot" in html
    with pytest.raises(ValueError):
        legend_html([("x", "red", "star")])


def test_contours_and_corners_take_colours_from_the_palette() -> None:
    contour = contour_background()
    corners = corner_marks(PALETTE.espresso)
    assert "#" not in contour and PALETTE.contour.replace("#", "%23") in contour
    assert corners.count("url(") == 4 and "top left" in corners and "bottom right" in corners


def test_hero_geometry_and_content() -> None:
    x, y = polar(90.0, 10.0)
    assert x == pytest.approx(10.0) and y == pytest.approx(0.0, abs=1e-9)
    assert bearing_of(*polar(210.0, 50.0)) == pytest.approx(210.0)
    assert len(blip_positions()) == 10
    top, left = ruler_html("top"), ruler_html("left")
    assert "73.845°E" in top and "73.865°E" in top and "18.510°N" in left and "18.530°N" in left
    page = hero_html()
    assert page.count('class="blip"') == 9 and "true-dot" in page and "spoof-dot" in page
    assert "NIS" in page and "48.6" in page and "ALARM · motion check" in page
    assert "Ed25519-signed evidence" in stats_line()
    with pytest.raises(ValueError):
        ruler_html("right")


def test_count_collected_reads_pytest_output() -> None:
    from cli import count_collected

    out = "tests/test_a.py::test_one\ntests/test_a.py::test_two[1]\n\n2 tests collected in 0.1s\n"
    assert count_collected(out) == 2
    with pytest.raises(ValueError):
        count_collected("no tests ran")


# --------------------------------------------------------------------------
# Banned-element sweep (docs/phases/phase-7.md acceptance checks)
# --------------------------------------------------------------------------
BANNED = [
    r"Instrument Serif", r"Bricolage", r"JetBrains", r"'Inter'", r"\bInter\b,", r"E6DDFE", r"0B4D40",
    r"FFFEF2", r"textPath", r"waveform", r"lavender", r"sg-forest", r"PALETTE\.forest", r"sg-console",
    r"neon", r"glassmorphism", r"linear-gradient", r"radial-gradient", r"blur\(",
]
SWEPT = [*(ROOT / "assets").rglob("*.py"), *(ROOT / "assets").glob("*.css"), *(ROOT / "assets").glob("*.html"),
         *(ROOT / "ui").glob("*.py"), *(ROOT / "views").glob("*.py"), *(ROOT / "reports").glob("*.py"),
         ROOT / "app.py", ROOT / ".streamlit" / "config.toml"]


@pytest.mark.parametrize("path", SWEPT, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_retired_or_banned_design_elements(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    found = [pattern for pattern in BANNED if re.search(pattern, text, flags=re.IGNORECASE)]
    assert not found, f"{path.name} still contains {found}"
