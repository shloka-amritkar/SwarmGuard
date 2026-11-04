"""Phase 1 tests: configs, data model, formation, sensors, attacks, simulator."""

from __future__ import annotations

import math
import time
from dataclasses import fields

import numpy as np
import pytest

from attacks.gps_attacks import AttackModel, ramp_shape
from attacks.scenarios import get_scenario, list_scenarios, scenario_keys
from config import (
    VALID_ATTACK_TYPES,
    AttackConfig,
    DetectorConfig,
    ExperimentConfig,
    ManoeuvreEvent,
    SimConfig,
)
from core.schema import TRUTH_FIELDS, AttackType, ObservedFrame
from core.utils import (
    chi2_2dof_quantile,
    clip_norm,
    enu_to_latlon,
    latlon_to_enu,
    pairwise_distances,
    rayleigh_quantile,
    rotation_matrix,
    wilson_interval,
)
from simulation.formation import (
    ShapeSchedule,
    apply_manoeuvres,
    assign_slots,
    get_shape,
    min_separation,
)
from simulation.sensors import RngBundle, SensorSuite
from simulation.swarm import simulate


# ---------------------------------------------------------------- config ----
def test_config_json_round_trip_and_hash() -> None:
    sim = SimConfig(manoeuvres=[ManoeuvreEvent("rotate", 1, 2, 45)], shape_sequence=[(0, "star")])
    again = SimConfig.from_json(sim.to_json())
    assert again == sim
    assert again.config_hash() == sim.config_hash()
    assert SimConfig(radius=8).config_hash() == SimConfig(radius=8.0).config_hash()
    assert SimConfig(seed=1).config_hash() != SimConfig(seed=2).config_hash()
    for cls in (AttackConfig, DetectorConfig, ExperimentConfig):
        obj = cls()
        assert cls.from_json(obj.to_json()) == obj


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_drones": 3},
        {"n_drones": 31},
        {"formation": "hexagon"},
        {"dt": 0},
        {"anchor_ratio": 1.5},
        {"gps_sigma": -1},
        {"path": "waypoints"},
        {"formation": "custom"},
        {"center_speed": 10.0},
        {"shape_sequence": [(5, "star"), (3, "heart")]},
    ],
)
def test_sim_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SimConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "jamming"},
        {"type": "constant_offset"},
        {"type": "constant_offset", "targets": [1, 2]},
        {"type": "multi_drone"},
        {"targets": [1, 1]},
        {"end_time": 5.0},
        {"pulse_duty": 0.0},
        {"ramp_profile": "cubic"},
    ],
)
def test_attack_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AttackConfig(**kwargs)


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        SimConfig.from_dict({"gps_sigmaa": 2.0})


def test_detector_defaults_match_derivations() -> None:
    cfg = DetectorConfig()
    assert cfg.tau_a == pytest.approx(3 * math.sqrt(2.09), abs=0.05)
    assert cfg.nis_threshold == pytest.approx(chi2_2dof_quantile(0.99), abs=0.01)
    assert cfg.tau_c == pytest.approx(rayleigh_quantile(math.sqrt(1.25), 0.999), abs=0.05)


def test_attack_type_enum_matches_config() -> None:
    assert {a.value for a in AttackType} == set(VALID_ATTACK_TYPES)


# ----------------------------------------------------------------- utils ----
def test_latlon_round_trip() -> None:
    x, y = np.array([0.0, 120.0, -50.0]), np.array([0.0, -30.0, 400.0])
    lat, lon = enu_to_latlon(x, y)
    assert lat[0] == pytest.approx(18.5204) and lon[0] == pytest.approx(73.8567)
    x2, y2 = latlon_to_enu(lat, lon)
    np.testing.assert_allclose(x2, x, atol=1e-6)
    np.testing.assert_allclose(y2, y, atol=1e-6)


def test_pairwise_and_clip() -> None:
    d = pairwise_distances(np.array([[0.0, 0.0], [3.0, 4.0]]))
    np.testing.assert_allclose(d, [[0, 5], [5, 0]])
    v = clip_norm(np.array([[6.0, 8.0], [0.3, 0.4], [0.0, 0.0]]), 5.0)
    np.testing.assert_allclose(v, [[3, 4], [0.3, 0.4], [0, 0]])


def test_quantiles_and_wilson() -> None:
    assert chi2_2dof_quantile(0.99) == pytest.approx(9.2103, abs=1e-3)
    assert rayleigh_quantile(1.118, 0.999) == pytest.approx(4.156, abs=1e-2)
    low, high = wilson_interval(50, 50)
    assert high == pytest.approx(1.0) and 0.92 < low < 0.94
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high


# ------------------------------------------------------------- formation ----
@pytest.mark.parametrize("name", ["circle", "line", "grid", "star", "heart"])
@pytest.mark.parametrize("n", [4, 10, 30])
def test_shapes_have_radius_r(name: str, n: int) -> None:
    offsets = get_shape(name, n, 8.0)
    assert offsets.shape == (n, 2)
    assert np.max(np.linalg.norm(offsets, axis=1)) == pytest.approx(8.0)
    assert min_separation(offsets) > 0.1


def test_assign_slots_minimises_travel() -> None:
    current = get_shape("circle", 10, 8.0)
    target = current[::-1].copy()
    np.testing.assert_allclose(assign_slots(current, target), current)


def test_schedule_blends_and_manoeuvres() -> None:
    sched = ShapeSchedule("circle", 10, 8.0, [(0, "circle"), (10, "line")])
    np.testing.assert_allclose(sched.offsets(5.0), get_shape("circle", 10, 8.0))
    half = sched.offsets(11.0)
    full = sched.offsets(12.0)
    np.testing.assert_allclose(half, 0.5 * (sched.offsets(10.0) + full))
    events = [ManoeuvreEvent("rotate", 0, 10, 90), ManoeuvreEvent("scale", 0, 10, 2.0)]
    out = apply_manoeuvres(np.array([[1.0, 0.0]]), 10.0, events)
    np.testing.assert_allclose(out, [[0.0, 2.0]], atol=1e-12)


# --------------------------------------------------------------- sensors ----
def test_ranges_symmetric_noisy_and_masked() -> None:
    cfg = SimConfig(comm_range=12.0)
    suite = SensorSuite(cfg, RngBundle.from_seed_sequence(np.random.SeedSequence(1)))
    pos = get_shape("line", 10, 8.0)  # spans 16 m, so some pairs are out of range
    ranges, mask = suite.ranges(pos)
    true = pairwise_distances(pos)
    assert np.all(np.diag(ranges) == 0) and not mask.diagonal().any()
    np.testing.assert_array_equal(mask, mask.T)
    np.testing.assert_array_equal(np.isnan(ranges), ~mask & ~np.eye(10, dtype=bool))
    np.testing.assert_allclose(ranges[mask], ranges.T[mask])
    assert np.all(ranges[mask] >= 0)
    # F2: stored values are the NOISY ones, not the true distances.
    assert not np.allclose(ranges[mask], true[mask])
    assert np.std(ranges[mask] - true[mask]) == pytest.approx(0.3, abs=0.15)


def test_anchors_fault_and_dropout() -> None:
    cfg = SimConfig(anchor_dropout=1.0)
    suite = SensorSuite(cfg, RngBundle.from_seed_sequence(np.random.SeedSequence(2)))
    pos = np.zeros((10, 2))
    anchored = np.arange(10) < 5
    readings, mask = suite.anchors(pos, anchored, np.zeros(10, dtype=bool))
    assert not mask.any() and np.isnan(readings).all()
    cfg = SimConfig(anchor_sigma=0.0)
    suite = SensorSuite(cfg, RngBundle.from_seed_sequence(np.random.SeedSequence(2)))
    faulty = np.arange(10) == 0
    readings, mask = suite.anchors(pos, anchored, faulty)
    np.testing.assert_array_equal(mask, anchored)
    assert np.linalg.norm(readings[0]) == pytest.approx(15.0)
    assert np.isnan(readings[~anchored]).all()


# --------------------------------------------------------------- attacks ----
def test_ramp_shapes_go_from_zero_to_one() -> None:
    for profile in ("linear", "exponential", "polynomial", "sigmoid"):
        assert ramp_shape(0.0, profile) == pytest.approx(0.0, abs=1e-12)
        assert ramp_shape(1.0, profile) == pytest.approx(1.0)
    assert ramp_shape(0.0, "step") == 1.0


def test_intermittent_labels_cover_whole_window() -> None:
    cfg = AttackConfig(type="intermittent", targets=[2], magnitude_m=6, start_time=20,
                       end_time=30, pulse_period=4, pulse_duty=0.5)
    model = AttackModel(cfg, 10)
    off_on, lab_on = model.offsets(20.5)
    off_off, lab_off = model.offsets(23.0)  # "off" half of the pulse
    assert np.linalg.norm(off_on[2]) == pytest.approx(6.0)
    assert np.allclose(off_off, 0.0)
    assert lab_on[2] and lab_off[2]  # F5: still labelled attacked
    _, lab_after = model.offsets(30.0)
    _, lab_before = model.offsets(19.9)
    assert not lab_after.any() and not lab_before.any()


def test_rotation_preserves_pairwise_distances() -> None:
    cfg = AttackConfig(type="rigid_rotation", rotation_rate_deg_s=3.0, magnitude_m=5.0,
                       start_time=0.0)
    model = AttackModel(cfg, 10)
    clean = get_shape("star", 10, 8.0) + np.array([40.0, -12.0])
    offset, labels = model.offsets(20.0, clean)
    spoofed = clean + offset
    assert labels.all() and np.linalg.norm(offset, axis=1).max() > 1.0
    np.testing.assert_allclose(pairwise_distances(spoofed), pairwise_distances(clean), atol=1e-9)
    rot = rotation_matrix(0.7)
    np.testing.assert_allclose(rot.T @ rot, np.eye(2), atol=1e-12)


def test_attack_rejects_missing_target() -> None:
    with pytest.raises(ValueError):
        AttackModel(AttackConfig(type="constant_offset", targets=[12]), 10)


def test_drift_cap() -> None:
    cfg = AttackConfig(type="gradual_drift", targets=[0], drift_rate=1.0, magnitude_m=5,
                       drift_cap=True, start_time=0)
    offset, _ = AttackModel(cfg, 4).offsets(20.0)
    assert np.linalg.norm(offset[0]) == pytest.approx(5.0)


# -------------------------------------------------------- schema / swarm ----
def test_observed_frame_has_no_truth_fields() -> None:
    observed_names = {f.name for f in fields(ObservedFrame)}
    assert observed_names.isdisjoint(TRUTH_FIELDS)
    assert "true_pos" in TRUTH_FIELDS


def test_frames_are_read_only() -> None:
    result = simulate(SimConfig(duration=1.0))
    with pytest.raises(ValueError):
        result.frames[0].observed.gps_pos[0, 0] = 99.0


def test_same_seed_same_output_different_seed_different() -> None:
    scen = get_scenario("single_hijack")
    a = simulate(scen.sim, scen.attack).stack("gps_pos")
    b = simulate(get_scenario("single_hijack").sim, scen.attack).stack("gps_pos")
    c = simulate(get_scenario("single_hijack", seed=7).sim, scen.attack).stack("gps_pos")
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_velocity_matches_true_motion() -> None:
    result = simulate(SimConfig(duration=5.0))
    pos, vel = result.stack("true_pos"), result.stack("true_vel")
    np.testing.assert_allclose(pos[1:] - pos[:-1], vel[:-1] * result.config.dt, atol=1e-12)
    assert np.linalg.norm(vel, axis=-1).max() <= result.config.max_speed + 1e-9


def test_rigid_shift_is_invisible_to_pairwise_distances() -> None:
    scen = get_scenario("rigid_swarm_shift")
    result = simulate(scen.sim, scen.attack)
    gps = result.stack("gps_pos")
    off = result.stack("attack_offset")
    k24, k30 = 240, 300
    assert np.allclose(off[k24], 0) and np.allclose(np.linalg.norm(off[k30], axis=1), 10.0)
    d_before = pairwise_distances(gps[k24])
    d_after = pairwise_distances(gps[k30])
    diff = (d_after - d_before)[np.triu_indices(10, 1)]
    # Differences come only from GPS noise (pair sigma about 1.4 m), not the 10 m shift.
    assert np.abs(diff).max() < 7.0 and abs(diff.mean()) < 1.0
    # The attack offsets alone change no distance at all.
    clean_after = gps[k30] - off[k30]
    np.testing.assert_allclose(pairwise_distances(clean_after), d_after, atol=1e-9)


def test_all_scenarios_run_and_dataframes() -> None:
    assert len(list_scenarios()) == len(scenario_keys()) == 11
    for key in scenario_keys():
        scen = get_scenario(key)
        result = simulate(scen.sim, scen.attack)
        assert result.n_steps == scen.sim.n_steps
        attacked = result.attacked_drones()
        assert bool(attacked) == (scen.attack.type != "none")
    obs = result.to_observed_dataframe()
    truth = result.to_truth_dataframe()
    assert len(obs) == len(truth) == result.n_steps * result.n_drones
    assert "true_x" not in obs.columns


def test_get_scenario_centre_gain_override() -> None:
    assert get_scenario("light_show_pune").sim.centre_gain == 0.5
    assert get_scenario("light_show_pune", centre_gain=0.0).sim.centre_gain == 0.0
    assert get_scenario("single_hijack", seed=3, centre_gain=0.2).sim.seed == 3
    with pytest.raises(ValueError):
        get_scenario("single_hijack", centre_gain=-1.0)


def test_swarm_wide_spoof_cancels_without_path_term() -> None:
    # With centre_gain = 0, a swarm-wide offset cancels in every command, so
    # the true flight is identical to the clean flight (same seed).
    scen = get_scenario("rigid_swarm_shift", centre_gain=0.0)
    attacked = simulate(scen.sim, scen.attack).stack("true_pos")
    clean = simulate(scen.sim, AttackConfig()).stack("true_pos")
    np.testing.assert_allclose(attacked, clean, atol=1e-9)
    # With the default 0.5, the path term drags the true swarm about 10 m west.
    scen = get_scenario("rigid_swarm_shift")
    attacked = simulate(scen.sim, scen.attack).stack("true_pos")
    clean = simulate(scen.sim, AttackConfig()).stack("true_pos")
    np.testing.assert_allclose((attacked - clean)[-1].mean(axis=0), [-10.0, 0.0], atol=0.2)


def _attack_effect(centre_gain: float) -> np.ndarray:
    """True positions with the single_hijack attack minus a clean run (same seed)."""
    scen = get_scenario("single_hijack", centre_gain=centre_gain)
    attacked = simulate(scen.sim, scen.attack).stack("true_pos")
    clean = simulate(scen.sim, AttackConfig()).stack("true_pos")
    return attacked - clean


def test_closed_loop_single_spoof_propagates() -> None:
    # centre_gain = 0: pure consensus formation keeping. The spoofed drone is
    # pushed by -D(1 - 1/N) and every other drone permanently by +D/N = 1 m east.
    effect = _attack_effect(0.0)
    np.testing.assert_allclose(effect[-1, 3], [-9.0, 0.0], atol=0.2)
    others_east = np.delete(effect[-1], 3, axis=0)[:, 0]
    np.testing.assert_allclose(others_east, 1.0, atol=0.1)
    # centre_gain > 0 (default): the path term pulls the swarm back, so the
    # others are only nudged briefly, while the spoofed drone is dragged the
    # full 10 m west.
    effect = _attack_effect(0.5)
    others_east = np.delete(effect, 3, axis=1)[:, :, 0].mean(axis=1)
    assert others_east.max() > 0.2 and abs(others_east[-1]) < 0.05
    np.testing.assert_allclose(effect[-1, 3], [-10.0, 0.0], atol=0.2)


def test_performance_600_steps_30_drones() -> None:
    cfg = SimConfig(n_drones=30)
    start = time.perf_counter()
    simulate(cfg, AttackConfig(type="swarm_wide"))
    assert time.perf_counter() - start < 1.0
