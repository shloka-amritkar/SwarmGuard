"""Shared pytest fixtures for the per-module test files (Phase 6).

WHY analytical default thresholds (DetectorConfig()) and not the saved
calibration: data/ is never committed, so a fresh clone has no calibration.
Tests must give the same answer on every machine.
WHY session-scoped caching: several files need the same scenario run;
simulating it once keeps the whole suite fast.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from attacks.scenarios import get_scenario
from config import DetectorConfig
from core.schema import SimResult
from detection import build_detectors
from detection.base import DetectionRun
from simulation.swarm import simulate

EVAL_SEED = 5000

ScenarioRunner = Callable[..., tuple[SimResult, dict[str, DetectionRun]]]


@pytest.fixture(scope="session")
def scenario_runs() -> ScenarioRunner:
    """Function (key, seed=5000) -> (SimResult, {"A": run, "B": run, "C": run}), cached."""
    cache: dict[tuple[str, int], tuple[SimResult, dict[str, DetectionRun]]] = {}

    def run(key: str, seed: int = EVAL_SEED) -> tuple[SimResult, dict[str, DetectionRun]]:
        if (key, seed) not in cache:
            scenario = get_scenario(key, seed=seed)
            result = simulate(scenario.sim, scenario.attack)
            runs = {d.short: d.run(result.observed())
                    for d in build_detectors(DetectorConfig(), scenario.sim)}
            cache[(key, seed)] = (result, runs)
        return cache[(key, seed)]

    return run
