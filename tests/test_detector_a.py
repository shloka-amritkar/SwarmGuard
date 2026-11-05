"""Detector A (relative geometry): finds a lone liar, blind to a rigid shift."""

from __future__ import annotations


def test_flags_index_3_in_single_hijack(scenario_runs) -> None:
    _, runs = scenario_runs("single_hijack")
    assert runs["A"].alarm_drones(after=20.0) == {3}


def test_never_alarms_in_rigid_swarm_shift(scenario_runs) -> None:
    """(g_i + D) - (g_j + D) = g_i - g_j: a common offset is invisible to A."""
    _, runs = scenario_runs("rigid_swarm_shift")
    assert not runs["A"].flags().any()
    assert not runs["A"].swarm_flags().any()


def test_two_attackers_flagged_no_honest_drone(scenario_runs) -> None:
    result, runs = scenario_runs("coordinated_pair")
    attacked = result.attacked_drones()
    assert len(attacked) == 2
    assert runs["A"].alarm_drones() == attacked
