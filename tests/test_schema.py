"""Rule R1: detectors can never see the simulation's ground truth."""

from __future__ import annotations

from dataclasses import fields

import pytest

from config import DetectorConfig, SimConfig
from core.schema import TRUTH_FIELDS, Frame, ObservedFrame
from detection import build_detectors
from simulation.swarm import simulate


def test_observed_frame_has_no_truth_fields() -> None:
    observed = {f.name for f in fields(ObservedFrame)}
    assert TRUTH_FIELDS, "the list of truth fields must not be empty"
    assert observed.isdisjoint(TRUTH_FIELDS)
    assert observed.isdisjoint({"true_pos", "true_vel", "attack_offset", "attack_labels"})


def test_frame_holds_truth_separately() -> None:
    frame = simulate(SimConfig(duration=1.0)).frames[0]
    assert isinstance(frame, Frame) and isinstance(frame.observed, ObservedFrame)
    assert not isinstance(frame, ObservedFrame)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_update_with_frame_raises_type_error(index: int) -> None:
    sim = SimConfig(duration=1.0)
    frame = simulate(sim).frames[0]
    detector = build_detectors(DetectorConfig(), sim)[index]
    with pytest.raises(TypeError):
        detector.update(frame)
    detector.update(frame.observed)          # the observed part is accepted
