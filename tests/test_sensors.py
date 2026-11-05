"""Radio ranging sensor: noise really stored (F2), symmetric, diagonal 0, NaN out of range."""

from __future__ import annotations

import numpy as np
import pytest

from config import SimConfig
from core.utils import pairwise_distances
from simulation.swarm import simulate


def test_stored_range_noise_matches_range_sigma_f2() -> None:
    sim = SimConfig(duration=20.0, range_sigma=0.3)
    result = simulate(sim)
    errors = []
    for frame in result.frames:
        true_dist = pairwise_distances(frame.true_pos)
        upper = np.triu(frame.observed.range_mask, k=1)
        errors.append((frame.observed.ranges - true_dist)[upper])
    err = np.concatenate(errors)
    assert err.std() == pytest.approx(0.3, rel=0.05)
    assert abs(err.mean()) < 0.02


def test_ranges_symmetric_with_zero_diagonal() -> None:
    obs = simulate(SimConfig(duration=2.0)).frames[-1].observed
    assert np.array_equal(obs.ranges, obs.ranges.T, equal_nan=True)
    assert np.all(np.diag(obs.ranges) == 0.0)


def test_nan_beyond_comm_range() -> None:
    # Circle of radius 8 m: opposite drones are 16 m apart, beyond a 12 m radio.
    result = simulate(SimConfig(duration=2.0, radius=8.0, comm_range=12.0))
    for frame in result.frames:
        far = pairwise_distances(frame.true_pos) > 12.0
        assert far.any()
        assert np.isnan(frame.observed.ranges[far]).all()
        assert not frame.observed.range_mask[far].any()
