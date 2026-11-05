"""Experimental Detector D: group split (F6), labels (F5), features, detector, hash-checked model.

Uses a tiny dataset (12 short runs) and a 20-tree forest so it runs in seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from attacks.scenarios import get_scenario
from config import DetectorConfig
from detection.detector_d_ml import (
    FEATURE_NAMES,
    MLDataset,
    MLDetector,
    build_dataset,
    group_split,
    load_model,
    run_ml,
    save_model,
    train,
    wrap_angle,
)
from simulation.swarm import simulate

SMALL_RF = {"n_estimators": 20, "n_jobs": 1}
SCENARIOS = ["clean_baseline", "single_hijack", "rigid_swarm_shift", "slow_drift"]


@pytest.fixture(scope="module")
def dataset() -> MLDataset:
    return build_dataset(12, scenarios=SCENARIOS, step_stride=4)


@pytest.fixture(scope="module")
def trained(dataset: MLDataset):
    return train(dataset.X, dataset.y, dataset.groups, dataset.attack_type, rf_params=SMALL_RF)


def test_no_group_in_both_train_and_test_f6(dataset: MLDataset, trained) -> None:
    train_idx, test_idx = group_split(dataset.groups)
    assert set(dataset.groups[train_idx]).isdisjoint(dataset.groups[test_idx])
    assert set(trained.train_groups).isdisjoint(trained.test_groups)
    assert set(trained.train_groups) | set(trained.test_groups) == set(range(12))


def test_feature_matrix_shape_and_no_nan(dataset: MLDataset) -> None:
    assert dataset.X.shape[1] == len(FEATURE_NAMES) == 11
    assert not np.isnan(dataset.X).any()
    assert len(dataset.X) == len(dataset.y) == len(dataset.groups) == len(dataset.attack_type)
    has_anchor = dataset.X[:, FEATURE_NAMES.index("has_anchor")]
    c_res = dataset.X[:, FEATURE_NAMES.index("c_residual")]
    assert np.all(c_res[has_anchor == 0] == -1.0)            # sentinel when there is no anchor


def test_labels_stay_one_for_whole_attack_window_f5(dataset: MLDataset) -> None:
    """Run 1 is single_hijack (drone 3 from 20 s): every kept step after onset is labelled 1."""
    y = dataset.y[dataset.groups == 1].reshape(-1, 10)       # (kept steps, drones)
    times = np.arange(y.shape[0]) * 0.4                      # step_stride 4 x dt 0.1
    assert np.all(y[times >= 20.0 + 1e-9, 3] == 1)
    assert np.all(y[times < 20.0 - 1e-9] == 0)
    assert np.all(np.delete(y, 3, axis=1) == 0)


def test_metrics_and_importances(trained) -> None:
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= trained.metrics[name] <= 1.0
    assert set(trained.feature_importances) == set(FEATURE_NAMES)
    assert sum(trained.feature_importances.values()) == pytest.approx(1.0)
    assert np.sum(trained.confusion) == trained.n_test


def test_wrap_angle() -> None:
    assert wrap_angle(np.array([np.radians(358)]))[0] == pytest.approx(np.radians(-2))
    assert abs(wrap_angle(np.array([3 * np.pi]))[0]) == pytest.approx(np.pi)


def test_ml_detector_refuses_truth_and_matches_stepwise(trained) -> None:
    sc = get_scenario("single_hijack", seed=5000)
    sc.sim.duration = 5.0
    result = simulate(sc.sim, sc.attack)
    det = MLDetector(trained.model, DetectorConfig(), sc.sim)
    with pytest.raises(TypeError):
        det.update(result.frames[0])
    batch = det.run(result.observed())
    stepwise = MLDetector(trained.model, DetectorConfig(), sc.sim)
    scores = np.stack([stepwise.update(obs).scores for obs in result.observed()])
    assert np.allclose(batch.scores(), scores)
    assert batch.short == "D"


def test_save_load_round_trip_and_hash_check(tmp_path: Path, trained) -> None:
    assert load_model(tmp_path) is None                       # nothing trained yet
    path = save_model(trained, {"n_runs": 12}, tmp_path)
    loaded = load_model(tmp_path)
    assert loaded is not None and loaded.info["metrics"] == trained.metrics
    sc = get_scenario("clean_baseline", seed=5000)
    sc.sim.duration = 3.0
    run = run_ml(simulate(sc.sim).observed(), loaded, DetectorConfig(), sc.sim)
    assert run.scores().shape == (30, 10)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="changed since training"):
        load_model(tmp_path)


# --------------------------------------------------------------------------
# The "Experimental ML detector" toggle in the real app (AppTest)
# --------------------------------------------------------------------------
def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: Path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SWARMGUARD_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("SWARMGUARD_KEYS", str(tmp_path / "keys"))
    monkeypatch.setenv("SWARMGUARD_PACKAGES", str(tmp_path / "packages"))
    monkeypatch.setenv("SWARMGUARD_MODELS", str(models))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    return at


def test_toggle_without_model_shows_train_button(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path, monkeypatch, tmp_path / "no_models")
    at.switch_page("views/experiments.py").run()
    assert len(at.tabs) == 4                                   # toggle off: no ML tab
    at.toggle(key="exp-ml").set_value(True).run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert any(b.key == "sg-empty-ml-experiments" for b in at.button)


def test_toggle_with_model_shows_detector_d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trained) -> None:
    models = tmp_path / "models"
    save_model(trained, {"n_runs": 12, "seed_start": 100000, "seed_end": 100011}, models)
    at = _app(tmp_path, monkeypatch, models)
    at.switch_page("views/detection.py").run()
    at.button(key="sg-empty-detection").click().run()          # quick-start run
    at.toggle(key="det-ml").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any(b.key == "sg-empty-ml-detection" for b in at.button)
    at.switch_page("views/experiments.py").run()
    at.toggle(key="exp-ml").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("sha256" in c.value for c in at.caption)
