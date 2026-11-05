"""Detector B (motion consistency): fast on jumps, CUSUM on drifts, gating keeps evidence."""

from __future__ import annotations

import numpy as np


def test_10m_step_detected_within_1s(scenario_runs) -> None:
    _, runs = scenario_runs("single_hijack")          # 10 m jump on drone index 3 at 20 s
    first = runs["B"].first_alarm_time(3, after=20.0)
    assert first is not None and first - 20.0 <= 1.0


def test_half_metre_per_second_drift_detected_within_15s(scenario_runs) -> None:
    _, runs = scenario_runs("slow_drift")             # 0.5 m/s on drone index 6 from 15 s
    first = runs["B"].first_alarm_time(6, after=15.0)
    assert first is not None and first - 15.0 <= 15.0


def test_gating_keeps_innovation_persistent(scenario_runs) -> None:
    """A gated fix is NOT absorbed into the estimate, so the 10 m innovation stays visible."""
    _, runs = scenario_runs("single_hijack")
    window = [s for s in runs["B"].steps if 20.0 <= s.t < 25.0]
    assert all(s.diagnostics["coasting"][3] for s in window)
    innovation = np.array([np.linalg.norm(s.diagnostics["innovation"][3]) for s in window])
    assert innovation.min() > 5.0
    assert all(s.flags[3] for s in window)
