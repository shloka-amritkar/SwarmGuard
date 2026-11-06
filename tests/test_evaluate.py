"""Tests for the scenario evaluation behind the README results tables (cli.py evaluate)."""

from __future__ import annotations

import pytest

import cli
from analysis.experiments import (
    EvalCase,
    case_kind,
    default_eval_cases,
    evaluate_scenarios,
    evaluation_table,
    format_rate,
    markdown_rows,
)
from attacks.scenarios import list_scenarios


def test_default_cases_cover_every_scenario_plus_attack_removed() -> None:
    cases = default_eval_cases()
    assert [c.scenario for c in cases[:-1]] == [s.key for s in list_scenarios()]
    assert cases[-1] == EvalCase("faulty_anchor_shift", remove_attack=True)
    assert cases[-1].label == "faulty_anchor_shift swarm, attack removed"


def test_case_kind() -> None:
    assert case_kind(EvalCase("single_hijack")) == "detection"
    assert case_kind(EvalCase("clean_baseline")) == "false_alarm"
    assert case_kind(EvalCase("single_hijack", remove_attack=True)) == "false_alarm"


def test_format_rate_shows_count_rate_and_wilson_interval() -> None:
    assert format_rate(50, 50) == "50/50 = 100.0 % (92.9-100.0 %)"
    assert format_rate(14, 50) == "14/50 = 28.0 % (17.5-41.7 %)"


def test_evaluation_rejects_calibration_seeds() -> None:
    with pytest.raises(ValueError, match="5000"):
        evaluate_scenarios([EvalCase("single_hijack")], 1, seed_start=1000,
                           require_calibration=False)


def test_small_evaluation_table_and_rows() -> None:
    cases = [EvalCase("single_hijack"), EvalCase("single_hijack", remove_attack=True)]
    df = evaluate_scenarios(cases, 2, require_calibration=False)
    assert len(df) == 2 * 2 * 4  # cases x seeds x detectors (A, B, C, F)
    assert set(df["seed"]) == {5000, 5001}
    table = evaluation_table(df)
    assert (table["n"] == 2).all()
    assert ((table["ci_low"] <= table["rate"]) & (table["rate"] <= table["ci_high"])).all()
    det_rows = markdown_rows(table, "detection")
    fa_rows = markdown_rows(table, "false_alarm")
    assert len(det_rows) == 1 and det_rows[0].startswith("| single_hijack | 2 (5000-5001) |")
    assert len(fa_rows) == 1 and "attack removed" in fa_rows[0]
    # A 10 m hijack on one drone is caught by Detector A on both seeds.
    a = table[(table["case"] == "single_hijack") & (table["detector"] == "A")].iloc[0]
    assert a["k"] == 2


def test_evaluation_is_deterministic() -> None:
    cases = [EvalCase("clean_baseline")]
    first = evaluate_scenarios(cases, 1, require_calibration=False)
    second = evaluate_scenarios(cases, 1, require_calibration=False)
    assert first.equals(second)


def test_cli_rejects_unknown_scenario() -> None:
    assert cli.main(["evaluate", "--scenarios", "not_a_scenario", "--seeds", "1"]) == 2
