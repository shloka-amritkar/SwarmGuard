"""Calibrated thresholds really give the target false-alarm rate on NEW clean flights.

Procedure (kept short so the test is fast; 10 s flights):
    1. calibrate on calibration seeds 1000..1199 with target 5 %
    2. fly 50 fresh clean evaluation seeds 5000..5049, built exactly like the
       calibration flights (odd seeds get the manoeuvre schedule, which
       starts at 20 s, so a 10 s flight is straight-line clean flight)
    3. per detector, count runs with ANY alarm: k out of 50
    4. pass if the target is not significantly exceeded, i.e. the lower end
       of the 95 % Wilson interval of k/50 is <= the target.
WHY a Wilson bound and not "k/50 <= 5 %": with 50 runs, 3 or 4 false alarms
can happen by chance even when the true rate is exactly 5 %. The interval
says what the data can and cannot rule out.
WHY 200 calibration seeds (the production default) and not fewer: picking
"the smallest threshold that meets the target" on a small sample is
optimistic. Measured on 20 s flights: 60 calibration seeds chose
tau_c = 2.9 m, which then alarmed on 14 % of new clean flights; 200 seeds
chose 3.1 m, which alarmed on 3 %. Short flights keep 200 seeds affordable.
"""

from __future__ import annotations

import pytest

from analysis.calibration import EVALUATION_SEED_START, calibrate, clean_sim_for_seed
from config import AttackConfig, SimConfig
from core.utils import wilson_interval
from detection import build_detectors
from simulation.swarm import simulate

TARGET = 0.05
N_CALIBRATION = 200
N_EVALUATION = 50


@pytest.fixture(scope="module")
def evaluation_counts() -> dict[str, int]:
    base = SimConfig(duration=10.0)
    thresholds = calibrate(base, n_seeds=N_CALIBRATION, target_fa=TARGET).thresholds
    counts = {"A": 0, "B": 0, "C": 0}
    for seed in range(EVALUATION_SEED_START, EVALUATION_SEED_START + N_EVALUATION):
        sim = clean_sim_for_seed(base, seed, include_manoeuvres=True)
        frames = simulate(sim, AttackConfig()).observed()
        for detector in build_detectors(thresholds, sim):
            run = detector.run(frames)
            counts[detector.short] += bool(run.flags().any() or run.swarm_flags().any())
    return counts


@pytest.mark.parametrize("short", ["A", "B", "C"])
def test_false_alarm_rate_within_wilson_bound_of_target(evaluation_counts, short: str) -> None:
    k = evaluation_counts[short]
    low, high = wilson_interval(k, N_EVALUATION)
    assert low <= TARGET, f"{short}: {k}/{N_EVALUATION} clean runs alarmed, CI [{low:.3f}, {high:.3f}]"
