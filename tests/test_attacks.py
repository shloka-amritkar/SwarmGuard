"""Simulated GPS attacks: rigid attacks keep shape, labels persist (F5), ramps go 0 -> 1."""

from __future__ import annotations

import numpy as np
import pytest

from attacks.gps_attacks import ramp_progress, ramp_shape
from attacks.scenarios import get_scenario
from config import VALID_RAMP_PROFILES
from core.utils import pairwise_distances
from simulation.swarm import simulate


@pytest.mark.parametrize("key", ["rigid_swarm_shift", "rigid_swarm_drift", "rigid_rotation"])
def test_rigid_attacks_preserve_pairwise_gps_distances(key: str) -> None:
    """With GPS noise switched off, GPS distances equal true distances at every step."""
    scenario = get_scenario(key, seed=5000)
    scenario.sim.gps_sigma = 0.0
    scenario.sim.duration = 40.0
    result = simulate(scenario.sim, scenario.attack)
    assert np.abs(result.stack("attack_offset")).max() > 1.0        # the attack really acted
    for frame in result.frames:
        gps = pairwise_distances(frame.observed.gps_pos)
        assert np.allclose(gps, pairwise_distances(frame.true_pos), atol=1e-9)


def test_labels_persist_for_whole_window_f5() -> None:
    """Intermittent pulses switch the offset off, but the label stays 1."""
    scenario = get_scenario("intermittent_pulses", seed=5000)
    result = simulate(scenario.sim, scenario.attack)
    target = scenario.attack.targets[0]
    times = result.times()
    labels = result.stack("attack_labels")[:, target]
    offsets = np.linalg.norm(result.stack("attack_offset")[:, target], axis=1)
    active = times >= scenario.attack.start_time
    if scenario.attack.end_time is not None:
        active &= times < scenario.attack.end_time
    assert np.array_equal(labels, active)
    assert (offsets[active] == 0).any(), "pulses should have 'off' moments inside the window"
    others = np.delete(result.stack("attack_labels"), target, axis=1)
    assert not others.any()


@pytest.mark.parametrize("profile", [p for p in VALID_RAMP_PROFILES if p != "step"])
def test_ramps_go_from_0_to_1(profile: str) -> None:
    assert ramp_shape(0.0, profile) == pytest.approx(0.0, abs=1e-12)
    assert ramp_shape(1.0, profile) == pytest.approx(1.0)
    values = [ramp_shape(p, profile) for p in np.linspace(0, 1, 21)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))      # never shrinks


def test_ramp_progress_clips() -> None:
    assert ramp_progress(5.0, start=10.0, ramp_duration=4.0) == 0.0
    assert ramp_progress(12.0, start=10.0, ramp_duration=4.0) == 0.5
    assert ramp_progress(30.0, start=10.0, ramp_duration=4.0) == 1.0
