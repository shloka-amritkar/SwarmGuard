"""Detector C (absolute anchors): sees rigid shifts, honest about no data, robust median."""

from __future__ import annotations

import math

import numpy as np

from analysis.metrics import compute_metrics
from config import DetectorConfig, SimConfig
from core.schema import DroneStatus, ObservedFrame
from detection import AnchorDetector
from simulation.swarm import simulate


def test_rigid_shift_detected_with_anchor_ratio_0_3(scenario_runs) -> None:
    result, runs = scenario_runs("rigid_swarm_shift")
    assert result.config.anchor_ratio == 0.3
    assert compute_metrics(result, runs["C"]).detected
    assert runs["C"].swarm_flags().any()


def test_zero_anchors_is_no_evidence_f8() -> None:
    run = AnchorDetector(DetectorConfig()).run(simulate(SimConfig(anchor_ratio=0.0, duration=3.0)).observed())
    for step in run.steps:
        assert all(s == DroneStatus.NO_EVIDENCE for s in step.status)
        assert math.isnan(step.swarm_score) and not step.swarm_flag
        assert np.isnan(step.scores).all()          # never a "clean" 0


def _four_anchor_frame(fault: float) -> ObservedFrame:
    """6 drones, 4 anchored, GPS exact; anchor index 2 reports `fault` metres east."""
    n = 6
    gps = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [15, 5]], dtype=float)
    anchors = np.full((n, 2), np.nan)
    mask = np.array([True, True, True, True, False, False])
    anchors[mask] = gps[mask]
    anchors[2] += [fault, 0.0]
    ranges = np.linalg.norm(gps[:, None] - gps[None], axis=-1)
    return ObservedFrame(0, 0.0, gps, np.zeros((n, 2)), ranges, ~np.eye(n, dtype=bool), anchors, mask)


def test_median_robust_to_one_faulty_anchor_of_four() -> None:
    cfg = DetectorConfig()
    step = AnchorDetector(cfg).update(_four_anchor_frame(fault=15.0))
    # residuals 0, 0, 15, 0 -> median 0, while the mean would be 3.75
    assert step.swarm_score == 0.0
    assert step.swarm_score < cfg.tau_c
    assert not step.diagnostics["swarm_raw_flag"]
    assert step.scores[2] == 15.0                     # the faulty reading itself is still visible
