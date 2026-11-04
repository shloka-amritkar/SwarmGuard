"""Phase 2 tests: detectors A, B, C, metrics and calibration."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from analysis.calibration import (
    CalibrationResult,
    calibrate,
    calibration_key,
    calibration_seeds,
    load_calibration,
    load_or_default,
    save_calibration,
)
from analysis.metrics import RunMetrics, aggregate, compute_metrics
from attacks.scenarios import get_scenario
from config import DetectorConfig, ManoeuvreEvent, SimConfig
from core.schema import DroneStatus, ObservedFrame
from core.utils import rotation_matrix
from detection import (
    AnchorDetector,
    DetectionRun,
    GeometryDetector,
    MotionDetector,
    PersistenceFilter,
    StepResult,
    build_detectors,
    persistence_over_time,
)
from detection.detector_a_geometry import geometry_residuals, vote
from detection.detector_b_motion import cusum_update
from detection.detector_c_anchor import aggregate_residuals
from simulation.swarm import simulate

EVAL_SEED = 5000


def _run_scenario(key: str, seed: int = EVAL_SEED, det: DetectorConfig | None = None):
    """Simulate a scenario and run A, B, C. Returns (SimResult, {short: DetectionRun})."""
    scenario = get_scenario(key, seed=seed)
    result = simulate(scenario.sim, scenario.attack)
    runs = {d.short: d.run(result.observed()) for d in build_detectors(det or DetectorConfig(), scenario.sim)}
    return result, runs


def _with_gps(obs: ObservedFrame, gps: np.ndarray) -> ObservedFrame:
    """Copy of an ObservedFrame with different GPS positions."""
    return ObservedFrame(obs.step, obs.t, gps, obs.vel_meas, obs.ranges, obs.range_mask,
                         obs.anchor_pos, obs.anchor_mask)


# ------------------------------------------------------------------ base ----
@pytest.mark.parametrize("make", [
    lambda: GeometryDetector(DetectorConfig()),
    lambda: MotionDetector(DetectorConfig(), 1.0, 0.1),
    lambda: AnchorDetector(DetectorConfig()),
])
def test_detectors_refuse_truth(make) -> None:
    result = simulate(SimConfig(duration=1.0))
    detector = make()
    with pytest.raises(TypeError, match="must not see truth"):
        detector.update(result.frames[0])
    with pytest.raises(TypeError):
        detector.run(result.frames)


def test_persistence_filter_m_of_n() -> None:
    pf = PersistenceFilter(3, 5, 1)
    outputs = [bool(pf.update(np.array([v]))[0]) for v in [1, 0, 1, 0, 1, 0, 0, 0, 1]]
    #      history sums (last 5): 1  1  2  2  3  2  2  1  1
    assert outputs == [False, False, False, False, True, False, False, False, False]


def test_vectorised_persistence_matches_filter() -> None:
    rng = np.random.default_rng(7)
    raw = rng.random((300, 6)) < 0.3
    pf = PersistenceFilter(3, 5, 6)
    stepwise = np.stack([pf.update(row) for row in raw])
    assert np.array_equal(stepwise, persistence_over_time(raw, 3, 5))


def test_detection_run_queries_and_dataframe() -> None:
    result, runs = _run_scenario("single_hijack")
    run = runs["A"]
    df = run.to_dataframe()
    assert list(df.columns) == ["step", "t", "drone", "flag", "raw_flag", "score", "status",
                                "swarm_flag", "swarm_score"]
    assert len(df) == result.n_steps * result.n_drones
    first = run.first_alarm_time(3)
    assert first is not None and first >= 20.0
    assert 3 in run.alarm_drones()


def test_detectors_are_deterministic() -> None:
    _, first = _run_scenario("slow_drift")
    _, second = _run_scenario("slow_drift")
    for key in "ABC":
        assert np.array_equal(first[key].flags(), second[key].flags())
        assert np.allclose(first[key].scores(), second[key].scores(), equal_nan=True)


# ------------------------------------------------------------- detector A ----
def test_common_shift_and_rotation_cancel_in_geometry() -> None:
    """Proof check: (g_i + D) - (g_j + D) = g_i - g_j, and rotations keep lengths."""
    obs = simulate(SimConfig(duration=1.0)).frames[3].observed
    base, _ = geometry_residuals(obs)
    shifted, _ = geometry_residuals(_with_gps(obs, obs.gps_pos + np.array([10.0, -4.0])))
    centre = obs.gps_pos.mean(axis=0)
    rotated_gps = (obs.gps_pos - centre) @ rotation_matrix(math.radians(30)).T + centre
    rotated, _ = geometry_residuals(_with_gps(obs, rotated_gps))
    assert np.allclose(base, shifted, equal_nan=True)
    assert np.allclose(base, rotated, equal_nan=True)


def test_vote_scores_majority() -> None:
    """Drone 0 disagrees with all partners; the others with only one."""
    n = 5
    residual = np.full((n, n), 0.5)
    residual[0, :] = residual[:, 0] = 10.0
    np.fill_diagonal(residual, np.nan)
    valid = ~np.eye(n, dtype=bool)
    scores, raw, has, bad = vote(residual, valid, tau_a=4.3, vote_fraction=0.5)
    assert scores[0] == 1.0 and raw[0]
    assert np.allclose(scores[1:], 0.25) and not raw[1:].any()
    assert has.all() and bad[0] == 4


def test_a_identifies_single_hijack() -> None:
    _, runs = _run_scenario("single_hijack")
    assert runs["A"].alarm_drones(after=20.0) == {3}


def test_a_blind_but_b_and_c_see_rigid_shift() -> None:
    result, runs = _run_scenario("rigid_swarm_shift")
    assert not runs["A"].flags().any()
    assert compute_metrics(result, runs["B"]).detected
    assert compute_metrics(result, runs["C"]).detected


def test_a_no_evidence_with_too_few_pairs() -> None:
    obs = simulate(SimConfig(duration=1.0)).frames[0].observed
    mask = np.zeros_like(obs.range_mask)
    lonely = ObservedFrame(0, 0.0, obs.gps_pos, obs.vel_meas, np.where(mask, obs.ranges, np.nan),
                           mask, obs.anchor_pos, obs.anchor_mask)
    step = GeometryDetector(DetectorConfig()).update(lonely)
    assert all(s == DroneStatus.NO_EVIDENCE for s in step.status)
    assert np.isnan(step.scores).all()


# ------------------------------------------------------------- detector B ----
def _straight_frames(vel_reported: np.ndarray, steps: int = 80) -> list[ObservedFrame]:
    """Noise-free frames of 4 drones flying 2 m/s east; vel_meas is what is reported."""
    n, dt = 4, 0.1
    start = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    true_vel = np.tile([2.0, 0.0], (n, 1))
    frames = []
    for k in range(steps):
        pos = start + true_vel * dt * k
        ranges = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        mask = ~np.eye(n, dtype=bool)
        frames.append(ObservedFrame(k, k * dt, pos, vel_reported, ranges, mask,
                                    np.full((n, 2), np.nan), np.zeros(n, dtype=bool)))
    return frames


def test_b_uses_velocity_input_f7() -> None:
    """With the velocity input the prediction is exact; ignoring it would alarm."""
    moving = _straight_frames(np.tile([2.0, 0.0], (4, 1)))
    run = MotionDetector(DetectorConfig(), gps_sigma=0.3, vel_sigma=0.05).run(moving)
    assert not run.flags().any()
    assert np.nanmax(run.steps[-1].diagnostics["nis"]) < 1e-6
    ignored = _straight_frames(np.zeros((4, 2)))
    run_bad = MotionDetector(DetectorConfig(), gps_sigma=0.3, vel_sigma=0.05).run(ignored)
    assert run_bad.flags().all(axis=1)[-1]


def test_cusum_update_rule() -> None:
    s = cusum_update(np.array([0.0, 5.0, 0.0]), np.array([4.5, 0.0, np.nan]), k=1.0)
    # 0 + 4.5 - 3 = 1.5 ; max(0, 5 + 0 - 3) = 2 ; NaN adds nothing
    assert np.allclose(s, [1.5, 2.0, 0.0])


def test_b_cusum_catches_slow_drift_before_10m() -> None:
    result, runs = _run_scenario("slow_drift")
    run = runs["B"]
    first = run.first_alarm_time(6, after=15.0)
    assert first is not None
    offset_at_alarm = 0.5 * (first - 15.0)            # drift_rate * elapsed time
    assert offset_at_alarm < 10.0
    change_points = run.steps[-1].diagnostics["change_points"]
    assert 6 in change_points


def test_b_latches_after_coast_limit() -> None:
    _, runs = _run_scenario("single_hijack")
    steps = runs["B"].steps
    reinit_steps = [s for s in steps if s.diagnostics["reinit"][3]]
    assert reinit_steps, "a 40 s spoof should exceed the 10 s coast limit"
    after = [s for s in steps if s.t >= reinit_steps[0].t]
    assert all(s.flags[3] for s in after)


# ------------------------------------------------------------- detector C ----
def test_c_zero_anchors_is_no_evidence_f8() -> None:
    sim = SimConfig(anchor_ratio=0.0, duration=3.0)
    run = AnchorDetector(DetectorConfig()).run(simulate(sim).observed())
    for step in run.steps:
        assert all(s == DroneStatus.NO_EVIDENCE for s in step.status)
        assert math.isnan(step.swarm_score) and not step.swarm_flag
        assert step.diagnostics["swarm_status"] == "no_evidence"


def test_c_median_worked_example() -> None:
    r = np.array([9.8, 10.1, 10.2, 100.0, np.nan])
    assert aggregate_residuals(r, "median") == pytest.approx(10.15)
    assert aggregate_residuals(r, "mean") == pytest.approx(32.525)
    assert aggregate_residuals(r, "max") == pytest.approx(100.0)
    assert math.isnan(aggregate_residuals(np.array([np.nan, np.nan]), "median"))


def test_c_detects_hijack_only_if_anchored() -> None:
    for seed in range(EVAL_SEED, EVAL_SEED + 6):
        result, runs = _run_scenario("single_hijack", seed=seed)
        anchored = bool(result.frames[0].anchored[3])
        assert compute_metrics(result, runs["C"]).detected == anchored


# ---------------------------------------------------------------- metrics ----
def _fake_run(result, alarms: dict[int, float]) -> DetectionRun:
    """DetectionRun where drone i is in ALARM from time alarms[i] onward."""
    n = result.n_drones
    steps = []
    for frame in result.frames:
        flags = np.array([i in alarms and frame.t >= alarms[i] for i in range(n)])
        steps.append(StepResult(frame.step, frame.t, flags, flags, flags.astype(float),
                                [DroneStatus.NORMAL] * n, bool(flags.any()), 0.0))
    return DetectionRun("fake", "X", steps)


def test_metrics_on_hand_built_run() -> None:
    scenario = get_scenario("single_hijack", seed=EVAL_SEED)
    result = simulate(scenario.sim, scenario.attack)
    m = compute_metrics(result, _fake_run(result, {3: 21.0, 5: 10.0}))
    assert m.detected and m.detection_delay == pytest.approx(1.0)
    assert m.early_alarms
    assert m.false_alarm_drones == {5}
    assert m.id_precision == pytest.approx(0.5)      # alarmed after onset: {3, 5}
    assert m.id_recall == pytest.approx(1.0)
    assert m.time_to_full_identification == pytest.approx(1.0)
    assert m.swarm_level_detected


def test_metrics_clean_run_and_undefined_precision() -> None:
    result = simulate(SimConfig(duration=5.0))
    quiet = compute_metrics(result, _fake_run(result, {}))
    noisy = compute_metrics(result, _fake_run(result, {2: 1.0}))
    assert not quiet.clean_false_alarm and noisy.clean_false_alarm
    assert math.isnan(quiet.id_precision)


def test_aggregate_rates_and_wilson() -> None:
    items = [RunMetrics("A", s, {1}, 20.0, detected=s < 8, detection_delay=1.0 if s < 8 else None)
             for s in range(10)]
    items += [RunMetrics("A", 100 + s, clean_false_alarm=(s == 0)) for s in range(5)]
    df = aggregate(items)
    row = df.iloc[0]
    assert row["detection_rate"] == pytest.approx(0.8)
    assert row["detection_ci_low"] < 0.8 < row["detection_ci_high"]
    assert row["clean_false_alarm_rate"] == pytest.approx(0.2)
    assert row["mean_delay_s"] == pytest.approx(1.0)


# ------------------------------------------------------------ calibration ----
def test_calibration_seed_ranges_never_overlap_evaluation() -> None:
    assert calibration_seeds(3) == [1000, 1001, 1002]
    with pytest.raises(ValueError):
        calibration_seeds(4001)


def test_calibration_key_ignores_seed_and_manoeuvres() -> None:
    base = SimConfig()
    assert calibration_key(base) == calibration_key(replace(base, seed=9))
    assert calibration_key(base) == calibration_key(
        replace(base, manoeuvres=[ManoeuvreEvent("rotate", 1, 2, 10)]))
    assert calibration_key(base) != calibration_key(replace(base, n_drones=12))


def test_small_calibration_round_trip_and_f3(tmp_path) -> None:
    sim = SimConfig(duration=20.0)
    assert load_or_default(sim, tmp_path) == DetectorConfig()   # nothing saved yet: defaults
    fractions: list[float] = []
    result = calibrate(sim, n_seeds=6, target_fa=0.2, progress=fractions.append)
    assert fractions[-1] == 1.0
    assert result.seed_start == 1000 and result.seed_end == 1005
    assert set(result.achieved) == {"A", "B", "C"}
    assert all(result.achieved[k]["rate"] <= 0.2 or result.notes for k in "AC")
    save_calibration(result, tmp_path)
    loaded = load_calibration(sim, tmp_path)
    assert isinstance(loaded, CalibrationResult)
    assert loaded.thresholds == result.thresholds
    assert load_or_default(sim, tmp_path) == result.thresholds
    assert len(loaded.table()) == 3
    # F3: DetectorConfig defaults are untouched by calibrating and loading.
    assert DetectorConfig().tau_a == 4.3
