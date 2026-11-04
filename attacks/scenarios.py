"""Ready-made scenarios: a flight, an attack, and what each detector should do.

Each Scenario bundles a SimConfig and an AttackConfig with a short
description and an `expected` dictionary:

    detector -> "detect"   the detector should raise an alarm
                "blind"    the detector is mathematically unable to see it
                "partial"  it sees only some drones, or only sometimes

Scenarios with no attack have an empty `expected` dict: every detector
should stay quiet, and any alarm there counts as a false alarm.

WHY presets: the same named situations are used in the CLI, tests, the
experiments page and the viva demo, so everyone talks about identical
flights (idea inspired by ZYRNAV Resilience Lab's scenario presets).

WHY builder functions instead of stored objects: configs are mutable
dataclasses. Building a fresh copy on every get_scenario call means a
caller that edits its copy can never change the preset for anyone else.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from config import AttackConfig, ManoeuvreEvent, SimConfig

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """A named, documented (SimConfig, AttackConfig) pair."""

    key: str
    title: str
    description: str
    sim: SimConfig
    attack: AttackConfig
    expected: dict[str, str] = field(default_factory=dict)


def _clean_baseline() -> Scenario:
    return Scenario(
        "clean_baseline", "Clean baseline",
        "Ten drones fly a straight line with no attack. This measures the false-alarm "
        "rate: every detector should report no anomaly detected.",
        SimConfig(), AttackConfig(),
    )


def _single_hijack() -> Scenario:
    return Scenario(
        "single_hijack", "Single drone hijack",
        "Drone 04's GPS jumps 10 m east at 20 s. Relative geometry (A) and motion "
        "consistency (B) should both catch it; anchors (C) only help if that drone "
        "carries an anchor.",
        SimConfig(),
        AttackConfig(type="constant_offset", targets=[3], magnitude_m=10.0, start_time=20.0),
        {"A": "detect", "B": "detect", "C": "partial"},
    )


def _slow_drift() -> Scenario:
    return Scenario(
        "slow_drift", "Slow single-drone drift",
        "Drone 07's GPS creeps east at 0.5 m/s from 15 s, too slowly for a jump test. "
        "CUSUM in Detector B accumulates the small errors; A fires once the drift "
        "exceeds its distance threshold.",
        SimConfig(),
        AttackConfig(type="gradual_drift", targets=[6], drift_rate=0.5, start_time=15.0),
        {"A": "detect", "B": "detect", "C": "partial"},
    )


def _coordinated_pair() -> Scenario:
    return Scenario(
        "coordinated_pair", "Coordinated pair",
        "Drones 02 and 06 are spoofed at 20 s by about 8 m each, in different seeded "
        "directions. Tests whether neighbour voting copes with more than one liar.",
        SimConfig(),
        AttackConfig(type="multi_drone", targets=[1, 5], magnitude_m=8.0, start_time=20.0),
        {"A": "detect", "B": "detect", "C": "partial"},
    )


def _rigid_swarm_shift() -> Scenario:
    return Scenario(
        "rigid_swarm_shift", "Rigid swarm shift",
        "Every drone's GPS jumps 10 m east at 25 s. All pairwise distances are "
        "unchanged, so relative geometry (A) is blind (the RigidShift blind spot); the "
        "sudden jump is visible to motion consistency (B) and to anchors (C).",
        SimConfig(),
        AttackConfig(type="swarm_wide", magnitude_m=10.0, start_time=25.0),
        {"A": "blind", "B": "detect", "C": "detect"},
    )


def _rigid_swarm_drift() -> Scenario:
    return Scenario(
        "rigid_swarm_drift", "Rigid swarm drift",
        "All GPS readings drift together at 0.3 m/s from 20 s. A stays blind because "
        "the formation looks perfect; B must rely on CUSUM to see the slow drift, "
        "while C sees the growing absolute error.",
        SimConfig(),
        AttackConfig(type="swarm_wide", drift_rate=0.3, ramp_profile="linear", start_time=20.0),
        {"A": "blind", "B": "detect", "C": "detect"},
    )


def _rigid_rotation() -> Scenario:
    return Scenario(
        "rigid_rotation", "Rigid rotation",
        "The whole reported formation rotates about its centre at 1 deg/s from 20 s. "
        "Rotation preserves every distance, so A is blind; the error is small near "
        "the centre, so B and C see it only partially.",
        SimConfig(),
        AttackConfig(type="rigid_rotation", rotation_rate_deg_s=1.0, magnitude_m=0.0,
                     start_time=20.0),
        {"A": "blind", "B": "partial", "C": "partial"},
    )


def _faulty_anchor_shift() -> Scenario:
    return Scenario(
        "faulty_anchor_shift", "Rigid shift with faulty anchors",
        "A 10 m swarm-wide shift at 25 s while a third of the anchors are biased by "
        "15 m. Tests whether Detector C stays reliable when its trusted references are "
        "themselves wrong.",
        SimConfig(anchor_ratio=0.5, anchor_fault_ratio=0.34),
        AttackConfig(type="swarm_wide", magnitude_m=10.0, start_time=25.0),
        {"A": "blind", "B": "detect", "C": "partial"},
    )


def _intermittent_pulses() -> Scenario:
    return Scenario(
        "intermittent_pulses", "Intermittent pulses",
        "Drone 03's GPS is pushed 6 m off for 2 s out of every 4 s from 20 s. Each "
        "on/off edge is a jump for B; A sees the pulses; labels stay attacked for the "
        "whole window (F5).",
        SimConfig(),
        AttackConfig(type="intermittent", targets=[2], magnitude_m=6.0, pulse_period=4.0,
                     pulse_duty=0.5, start_time=20.0),
        {"A": "detect", "B": "detect", "C": "partial"},
    )


def _manoeuvre_only() -> Scenario:
    return Scenario(
        "manoeuvre_only", "Legitimate manoeuvres only",
        "No attack: the formation rotates 90 deg over 20-30 s and grows to 1.5x over "
        "35-40 s. Any alarm here is a false alarm caused by legitimate motion.",
        SimConfig(manoeuvres=[
            ManoeuvreEvent("rotate", 20.0, 30.0, 90.0),
            ManoeuvreEvent("scale", 35.0, 40.0, 1.5),
        ]),
        AttackConfig(),
    )


def _light_show_pune() -> Scenario:
    return Scenario(
        "light_show_pune", "Pune light show",
        "Twenty drones fly a loop over Pune, changing circle, star, heart, grid, circle. "
        "From 30 s all GPS readings drift together at 0.3 m/s, so the whole show is "
        "dragged off course while the formation still looks perfect to relative "
        "geometry (A).",
        SimConfig(n_drones=20, radius=15.0, path="loop", loop_mode="closed",
                  shape_sequence=[(0.0, "circle"), (12.0, "star"), (24.0, "heart"),
                                  (36.0, "grid"), (48.0, "circle")]),
        AttackConfig(type="swarm_wide", drift_rate=0.3, ramp_profile="linear", start_time=30.0),
        {"A": "blind", "B": "detect", "C": "detect"},
    )


_BUILDERS: dict[str, Callable[[], Scenario]] = {
    "clean_baseline": _clean_baseline,
    "single_hijack": _single_hijack,
    "slow_drift": _slow_drift,
    "coordinated_pair": _coordinated_pair,
    "rigid_swarm_shift": _rigid_swarm_shift,
    "rigid_swarm_drift": _rigid_swarm_drift,
    "rigid_rotation": _rigid_rotation,
    "faulty_anchor_shift": _faulty_anchor_shift,
    "intermittent_pulses": _intermittent_pulses,
    "manoeuvre_only": _manoeuvre_only,
    "light_show_pune": _light_show_pune,
}


def scenario_keys() -> list[str]:
    """Return all scenario keys in display order."""
    return list(_BUILDERS)


def get_scenario(
    key: str,
    seed: int | None = None,
    centre_gain: float | None = None,
) -> Scenario:
    """Return a fresh copy of the named scenario, with optional overrides.

    seed:        replaces both the simulation seed and the attack seed, so
                 the same scenario can be re-run with different random noise.
    centre_gain: replaces SimConfig.centre_gain (closed-loop path pull,
                 1/s). 0.0 gives pure consensus formation keeping, where a
                 single-drone spoof spreads offset / N to every drone; the
                 default 0.5 drags the swarm with a swarm-wide spoof. See
                 simulation/swarm.py. Used by propagation analysis to run
                 the same flight under both control laws.
    """
    if key not in _BUILDERS:
        raise ValueError(f"unknown scenario {key!r}; choose from {scenario_keys()}")
    scenario = _BUILDERS[key]()
    if seed is not None:
        scenario.sim.seed = int(seed)
        scenario.attack.seed = int(seed)
    if centre_gain is not None:
        scenario.sim.centre_gain = centre_gain
    if seed is not None or centre_gain is not None:
        # WHY re-validate: fields were changed after construction, so the
        # same checks (e.g. centre_gain >= 0) must run again.
        scenario.sim.__post_init__()
        scenario.attack.__post_init__()
    return scenario


def list_scenarios() -> list[Scenario]:
    """Return a fresh copy of every scenario, in display order."""
    return [builder() for builder in _BUILDERS.values()]
