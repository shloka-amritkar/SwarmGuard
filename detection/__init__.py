"""Detectors: each sees only ObservedFrame data (rule R1).

    A  GeometryDetector   GPS distances vs. radio ranges   (blind to rigid shifts)
    B  MotionDetector     GPS vs. own velocity sensor      (Kalman + CUSUM)
    C  AnchorDetector     GPS vs. absolute anchor fixes    (anchored drones only)
"""

from __future__ import annotations

from config import DetectorConfig, SimConfig
from detection.base import (
    DetectionRun,
    Detector,
    PersistenceFilter,
    StepResult,
    persistence_over_time,
    status_from_flags,
)
from detection.detector_a_geometry import GeometryDetector
from detection.detector_b_motion import MotionDetector
from detection.detector_c_anchor import AnchorDetector


def build_detectors(det_cfg: DetectorConfig, sim_cfg: SimConfig) -> list[Detector]:
    """Create detectors A, B and C with the given thresholds.

    Only sim_cfg.gps_sigma and sim_cfg.vel_sigma are read: they are the
    sensor datasheet noise levels Detector B's Kalman filter needs. No truth
    and no attack information is passed.
    """
    return [
        GeometryDetector(det_cfg),
        MotionDetector(det_cfg, gps_sigma=sim_cfg.gps_sigma, vel_sigma=sim_cfg.vel_sigma),
        AnchorDetector(det_cfg),
    ]


__all__ = [
    "AnchorDetector",
    "DetectionRun",
    "Detector",
    "GeometryDetector",
    "MotionDetector",
    "PersistenceFilter",
    "StepResult",
    "build_detectors",
    "persistence_over_time",
    "status_from_flags",
]
