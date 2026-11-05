"""Rule R2: same config + seed gives identical output; another seed differs."""

from __future__ import annotations

import numpy as np

from attacks.scenarios import get_scenario
from simulation.swarm import simulate

ARRAYS = ("gps_pos", "vel_meas", "ranges", "anchor_pos", "true_pos", "attack_offset")


def _run(seed: int):
    scenario = get_scenario("single_hijack", seed=seed)
    scenario.sim.duration = 25.0
    return simulate(scenario.sim, scenario.attack)


def test_same_seed_identical() -> None:
    a, b = _run(7), _run(7)
    for name in ARRAYS:
        assert np.array_equal(a.stack(name), b.stack(name), equal_nan=True), name


def test_different_seeds_differ() -> None:
    a, b = _run(7), _run(8)
    assert not np.allclose(a.stack("gps_pos"), b.stack("gps_pos"))
    assert not np.allclose(a.stack("ranges"), b.stack("ranges"), equal_nan=True)
