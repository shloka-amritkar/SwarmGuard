"""Dempster-Shafer fusion: valid masses, total conflict, and neutral NO_EVIDENCE."""

from __future__ import annotations

import numpy as np

from config import DetectorConfig
from core.schema import DroneStatus
from detection.base import StepResult
from detection.fusion import dempster, masses_for_step


def test_masses_sum_to_one() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        a, b = rng.random(3), rng.random(3)
        out, _ = dempster(a / a.sum(), b / b.sum())
        assert np.isclose(out.sum(), 1.0) and (out >= 0).all()


def test_total_conflict_gives_ignorance() -> None:
    """One source certain 'normal', the other certain 'compromised': K = 1 -> (0, 0, 1)."""
    out, high_conflict = dempster(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert high_conflict
    assert np.allclose(out, [0.0, 0.0, 1.0])


def test_no_evidence_is_neutral() -> None:
    m = np.array([0.5, 0.2, 0.3])
    out, _ = dempster(m, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(out, m)
    step = StepResult(0, 0.0, np.zeros(1, bool), np.zeros(1, bool), np.array([np.nan]),
                      [DroneStatus.NO_EVIDENCE], False, float("nan"))
    masses = masses_for_step("C", step, DetectorConfig())
    assert np.allclose(masses[0], [0.0, 0.0, 1.0])     # all mass on "don't know"
