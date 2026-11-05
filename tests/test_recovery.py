"""Recovery: MDS + Kabsch, RANSAC alignment (F4), multilateration (F1)."""

from __future__ import annotations

import math

import numpy as np

from core.schema import ObservedFrame
from core.utils import pairwise_distances, rotation_matrix
from recovery.mds_alignment import RecoveryStatus, classical_mds, fit_with_reflection, kabsch, ransac_align
from recovery.multilateration import repair_all, repair_drone

POINTS = np.array([[0, 0], [10, 0], [3, 7], [-4, 5], [8, 9], [-6, -3]], dtype=float)


def test_noiseless_mds_reconstructs_geometry() -> None:
    """MDS is only defined up to rotation, translation and mirror; align and compare."""
    x, filled = classical_mds(pairwise_distances(POINTS))
    assert not filled.any()
    fit, residuals = fit_with_reflection(x, POINTS)
    assert residuals.max() < 1e-6
    assert np.abs(fit.apply(x) - POINTS).max() < 1e-6


def test_kabsch_recovers_known_transform() -> None:
    rot_true = rotation_matrix(math.radians(37))
    moved = POINTS @ rot_true.T + np.array([5.0, -2.0])
    rot, t = kabsch(POINTS, moved)
    assert np.allclose(rot, rot_true) and np.allclose(t, [5.0, -2.0])


def test_reflection_is_handled() -> None:
    """Kabsch itself never mirrors (det = +1); the mirrored fit handles reflections."""
    mirrored = POINTS * np.array([1.0, -1.0]) @ rotation_matrix(0.3).T
    rot, _ = kabsch(POINTS, mirrored)
    assert np.isclose(np.linalg.det(rot), 1.0)
    fit, residuals = fit_with_reflection(POINTS, mirrored)
    assert fit.mirrored and residuals.max() < 1e-6


def test_ransac_rejects_one_faulty_anchor_of_five() -> None:
    pts = POINTS[:5]
    targets = pts @ rotation_matrix(0.4).T + np.array([20.0, 3.0])
    targets[2] += np.array([15.0, 0.0])
    res = ransac_align(pts, np.arange(5), targets)
    assert res.status == RecoveryStatus.OK
    assert res.inliers.tolist() == [True, True, False, True, True]


def _obs(gps: np.ndarray, truth: np.ndarray) -> ObservedFrame:
    n = len(truth)
    return ObservedFrame(0, 0.0, gps, np.zeros((n, 2)), pairwise_distances(truth), ~np.eye(n, dtype=bool),
                         np.full((n, 2), np.nan), np.zeros(n, bool))


def test_multilateration_excludes_flagged_neighbours_f1() -> None:
    truth = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5]], dtype=float)
    gps = truth.copy()
    gps[3] += [6.0, 6.0]
    gps[4] += [8.0, 0.0]
    flagged = np.array([False, False, False, True, True])
    rep = repair_drone(4, _obs(gps, truth), flagged)
    assert rep.status == RecoveryStatus.OK
    assert 3 not in rep.neighbours and set(rep.neighbours) == {0, 1, 2}
    assert np.linalg.norm(rep.position - truth[4]) < 1e-3


def test_multilateration_insufficient_neighbours() -> None:
    truth = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    reps = repair_all(_obs(truth, truth), np.array([False, True, True, False]))
    assert all(r.status == RecoveryStatus.INSUFFICIENT_NEIGHBOURS and r.position is None
               for r in reps.values())
