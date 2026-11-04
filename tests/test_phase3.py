"""Phase 3 tests: fusion, recovery, experiments, evasion search, propagation."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from analysis.calibration import load_or_default
from analysis.evasion_search import bisect_threshold, default_params
from analysis.experiments import apply_param, run_pipeline, run_trial, summarise, sweep
from analysis.propagation import min_separation, pagerank, sensitivity_matrix
from attacks.scenarios import get_scenario
from config import AttackConfig, DetectorConfig, SimConfig
from core.schema import DroneStatus, ObservedFrame, Verdict
from core.utils import pairwise_distances, rotation_matrix
from detection.base import DetectionRun, StepResult
from detection.fusion import (
    agreement_matrix,
    dempster,
    describe_pattern,
    masses_for_step,
    pattern_key,
    verdict_from_masses,
)
from recovery.mds_alignment import (
    RecoveryError,
    RecoveryStatus,
    classical_mds,
    fill_missing_distances,
    kabsch,
    ransac_align,
    recover_positions,
    recover_run,
)
from recovery.multilateration import repair_all, repair_drone
from simulation.swarm import simulate

EVAL_SEED = 5000


@pytest.fixture(scope="module")
def shift_pipeline():
    """Full pipeline on rigid_swarm_shift (evaluation seed), shared by several tests."""
    sc = get_scenario("rigid_swarm_shift", seed=EVAL_SEED)
    return run_pipeline(sc.sim, sc.attack, load_or_default(sc.sim))


# ---------------------------------------------------------------- fusion ----
def test_dempster_masses_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    raw = rng.random((200, 2, 3))
    m = raw / raw.sum(axis=-1, keepdims=True)
    out, _ = dempster(m[:, 0], m[:, 1])
    assert np.allclose(out.sum(axis=-1), 1.0)
    assert (out >= 0).all()


def test_total_conflict_gives_ignorance() -> None:
    out, high = dempster(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert high and np.allclose(out, [0.0, 0.0, 1.0])


def test_no_evidence_is_neutral() -> None:
    m = np.array([0.42, 0.33, 0.25])
    out, high = dempster(m, np.array([0.0, 0.0, 1.0]))
    assert not high and np.allclose(out, m)


def test_dempster_worked_example() -> None:
    """A says normal, B says attack: K = 0.7 * 0.8 = 0.56."""
    out, _ = dempster(np.array([0.0, 0.7, 0.3]), np.array([0.8, 0.0, 0.2]))
    assert np.allclose(out, [0.24 / 0.44, 0.14 / 0.44, 0.06 / 0.44])


@pytest.mark.parametrize("masses, expected", [
    ((0.05, 0.75, 0.20), Verdict.NORMAL),
    ((0.80, 0.10, 0.10), Verdict.COMPROMISED),
    ((0.05, 0.05, 0.90), Verdict.INSUFFICIENT_EVIDENCE),
    ((0.40, 0.30, 0.30), Verdict.SUSPICIOUS),
])
def test_verdict_rules(masses, expected) -> None:
    assert verdict_from_masses(np.array(masses)) == expected


def test_masses_for_no_evidence_drones() -> None:
    step = StepResult(0, 0.0, np.zeros(2, bool), np.zeros(2, bool), np.array([np.nan, 5.0]),
                      [DroneStatus.NO_EVIDENCE, DroneStatus.ALARM], False, 5.0)
    m = masses_for_step("C", step, DetectorConfig())
    assert np.allclose(m[0], [0, 0, 1])
    assert np.isclose(m[1].sum(), 1.0) and m[1, 0] > 0
    m_low = masses_for_step("C", step, DetectorConfig(), c_outliers=True)
    assert np.isclose(m_low[1, 2], 0.4)            # reliability drops to 0.6


def test_rigid_shift_pattern_and_fusion(shift_pipeline) -> None:
    key = pattern_key(shift_pipeline.runs)
    assert key == "A_silent_B_C_alarm"
    assert "swarm shift" in describe_pattern(key)
    table = agreement_matrix(shift_pipeline.runs)
    assert set(table["A"]) == {"silent"} and set(table["B"]) == {"alarm"}
    df = shift_pipeline.fusion.to_dataframe()
    assert np.allclose(df[["m_attack", "m_normal", "m_uncertain"]].sum(axis=1), 1.0)
    assert (df["verdict"] == "compromised").any()


def test_pattern_key_for_quiet_runs() -> None:
    sc = get_scenario("clean_baseline", seed=EVAL_SEED)
    pipe = run_pipeline(sc.sim, sc.attack, load_or_default(sc.sim), include_recovery=False)
    assert pattern_key(pipe.runs) == "A_B_C_silent"


# -------------------------------------------------------------- recovery ----
def _points() -> np.ndarray:
    return np.array([[0, 0], [10, 0], [3, 7], [-4, 5], [8, 9], [-6, -3]], dtype=float)


def test_noiseless_mds_reconstructs_geometry() -> None:
    pts = _points()
    x, filled = classical_mds(pairwise_distances(pts))
    assert not filled.any()
    for mirrored in (False, True):
        src = x * np.array([1.0, -1.0]) if mirrored else x
        rot, t = kabsch(src, pts)
        err = np.abs(src @ rot.T + t - pts).max()
        if err < 1e-6:
            break
    assert err < 1e-6


def test_kabsch_recovers_known_transform_and_never_reflects() -> None:
    pts = _points()
    rot_true = rotation_matrix(math.radians(37))
    moved = pts @ rot_true.T + np.array([5.0, -2.0])
    rot, t = kabsch(pts, moved)
    assert np.allclose(rot, rot_true) and np.allclose(t, [5.0, -2.0])
    mirrored = pts * np.array([1.0, -1.0])
    rot_m, _ = kabsch(pts, mirrored)
    assert np.isclose(np.linalg.det(rot_m), 1.0)       # a rotation, never a mirror


def test_floyd_warshall_fill_and_disconnected() -> None:
    d = np.array([[0, 1, np.nan], [1, 0, 2], [np.nan, 2, 0]], dtype=float)
    full, filled = fill_missing_distances(d)
    assert full[0, 2] == 3 and filled[0, 2] and filled[2, 0]
    split = np.array([[0, 1, np.nan], [1, 0, np.nan], [np.nan, np.nan, 0]], dtype=float)
    with pytest.raises(RecoveryError):
        fill_missing_distances(split)


def test_ransac_rejects_one_faulty_anchor_of_five() -> None:
    pts = _points()[:5]
    targets = pts @ rotation_matrix(0.4).T + np.array([20.0, 3.0])
    targets[2] += np.array([15.0, 0.0])              # faulty anchor
    res = ransac_align(pts, np.arange(5), targets)
    assert res.status == RecoveryStatus.OK
    assert res.inliers.tolist() == [True, True, False, True, True]


def test_ransac_needs_three_anchors() -> None:
    res = ransac_align(_points(), np.arange(2), _points()[:2])
    assert res.status == RecoveryStatus.INSUFFICIENT_ANCHORS


def test_recovery_estimates_rigid_shift(shift_pipeline) -> None:
    summary = shift_pipeline.recovery.summary()
    vec = np.array(summary["attack_vector"])
    true_offset = shift_pipeline.sim_result.frames[-1].attack_offset[0]   # truth: check only
    assert np.linalg.norm(vec - true_offset) < 2.0


def test_ransac_marks_faulty_anchors() -> None:
    sc = get_scenario("faulty_anchor_shift", seed=EVAL_SEED)
    result = simulate(sc.sim, sc.attack)
    frame = result.frames[-1]
    rec = recover_positions(frame.observed)
    assert set(rec.outlier_anchors) == set(np.flatnonzero(frame.faulty_anchor))


def test_recovery_refuses_truth() -> None:
    frame = simulate(SimConfig(duration=1.0)).frames[0]
    with pytest.raises(TypeError):
        recover_run([frame])
    with pytest.raises(TypeError):
        repair_drone(0, frame, np.zeros(10, bool))


def _square_obs(gps: np.ndarray, truth: np.ndarray) -> ObservedFrame:
    """Frame with exact ranges from `truth` and the given GPS readings."""
    n = len(truth)
    ranges = pairwise_distances(truth)
    mask = ~np.eye(n, dtype=bool)
    return ObservedFrame(0, 0.0, gps, np.zeros((n, 2)), ranges, mask,
                         np.full((n, 2), np.nan), np.zeros(n, bool))


def test_multilateration_excludes_flagged_neighbours_f1() -> None:
    truth = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5]], dtype=float)
    gps = truth.copy()
    gps[4] += [8.0, 0.0]          # drone 4 spoofed
    gps[3] += [6.0, 6.0]          # drone 3 also spoofed (flagged)
    flagged = np.array([False, False, False, True, True])
    rep = repair_drone(4, _square_obs(gps, truth), flagged)
    assert rep.status == RecoveryStatus.OK and 3 not in rep.neighbours
    assert np.linalg.norm(rep.position - truth[4]) < 1e-3


def test_multilateration_insufficient_neighbours() -> None:
    truth = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    flagged = np.array([False, True, True, False])
    reps = repair_all(_square_obs(truth, truth), flagged)
    assert set(reps) == {1, 2}
    assert all(r.status == RecoveryStatus.INSUFFICIENT_NEIGHBOURS and r.position is None
               for r in reps.values())


# ------------------------------------------------------------ experiments ----
def test_apply_param_validates() -> None:
    sc = get_scenario("coordinated_pair")
    _, atk = apply_param(sc.sim, sc.attack, "n_targets", 3)
    assert atk.targets == [0, 1, 2]
    single = get_scenario("single_hijack")
    with pytest.raises(ValueError):
        apply_param(single.sim, single.attack, "n_targets", 2)
    with pytest.raises(ValueError):
        apply_param(single.sim, single.attack, "bogus", 1)


def test_run_trial_is_deterministic() -> None:
    sc = get_scenario("single_hijack")
    det = load_or_default(sc.sim)
    a = run_trial(sc.sim, sc.attack, det, EVAL_SEED)
    b = run_trial(sc.sim, sc.attack, det, EVAL_SEED)
    assert set(a["metrics"]) == {"A", "B", "C", "F"}
    assert a["metrics"]["A"] == b["metrics"]["A"]
    assert a["recovery_error_m"] == pytest.approx(b["recovery_error_m"])
    assert a["recovery_error_m"] < a["gps_error_m"]


def test_small_sweep_rises_with_magnitude() -> None:
    df = sweep("magnitude", [1.0, 12.0], "single_hijack", n_seeds=3, include_recovery=False)
    assert len(df) == 2 * 3 * 4
    table = summarise(df)
    a = table[table["detector"] == "A"].set_index("value")["detection_rate"]
    assert a[12.0] >= a[1.0] and a[12.0] == 1.0
    with pytest.raises(ValueError):
        sweep("magnitude", [1.0], "single_hijack", n_seeds=1, seed_start=1000)


# ---------------------------------------------------------------- evasion ----
def test_bisection_on_monotone_function() -> None:
    rate = lambda x: min(1.0, max(0.0, (x - 2.0) / 4.0))  # noqa: E731
    assert bisect_threshold(rate, 0.5, 0.0, 10.0, 0.01) == pytest.approx(4.0, abs=0.01)
    assert bisect_threshold(rate, 0.5, 5.0, 10.0, 0.01) == 5.0
    assert bisect_threshold(lambda x: 0.0, 0.5, 0.0, 10.0, 0.01) is None


def test_default_params_follow_attack_type() -> None:
    assert default_params(get_scenario("slow_drift")) == ("drift_rate",)
    assert default_params(get_scenario("rigid_swarm_drift")) == ("drift_rate",)
    assert default_params(get_scenario("single_hijack")) == ("magnitude",)
    with pytest.raises(ValueError):
        default_params(get_scenario("clean_baseline"))


# ------------------------------------------------------------ propagation ----
def test_pagerank_hand_graph() -> None:
    # Edges: 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 0.  W[j, i] = weight of i -> j.
    w = np.zeros((3, 3))
    w[1, 0] = w[2, 0] = w[2, 1] = w[0, 2] = 1.0
    r = pagerank(w)
    assert np.isclose(r.sum(), 1.0)
    assert r[2] > r[1]                      # node 2 has two incoming links
    dangling = pagerank(np.zeros((4, 4)))
    assert np.allclose(dangling, 0.25)


def test_propagation_rejects_open_loop() -> None:
    with pytest.raises(ValueError, match="closed"):
        sensitivity_matrix(SimConfig(loop_mode="open"))


def test_sensitivity_matrix_centre_gain_zero_spreads_d_over_n() -> None:
    res = sensitivity_matrix(replace(SimConfig(), centre_gain=0.0), magnitude=6.0)
    off = res.matrix[~np.eye(10, dtype=bool)]
    assert off.mean() == pytest.approx(6.0 / 10, abs=0.1)        # D / N
    assert np.allclose(np.diag(res.matrix), 0.0)
    assert np.isclose(res.influence.sum(), 1.0)


def test_min_separation_counts_one_event_per_pass() -> None:
    result = simulate(SimConfig(duration=30.0),
                      AttackConfig(type="constant_offset", targets=[2], magnitude_m=6.0, start_time=20.0))
    series, events = min_separation(result)
    assert series.shape == (result.n_steps,) and series.min() >= 0
    pairs = [(e.i, e.j) for e in events]
    assert len(pairs) == len(set(pairs))     # hysteresis: no flicker double-counting here
