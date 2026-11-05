"""Detector D (EXPERIMENTAL): a Random Forest that learns from our simulator.

    FeatureBuilder  11 numbers per drone per step, from ObservedFrame + A, B, C
    extract_features  a whole run of features at once, shape (T, N, 11)
    build_dataset   many simulated runs -> X, y, groups (+ attack type per row)
    train           group split by run (F6) + RandomForestClassifier + metrics
    save_model / load_model   rf.joblib with its SHA-256 and a config JSON
    MLDetector      predict_proba > 0.5, then the usual m-of-n persistence

WHY "experimental": the forest learns only from OUR simulator, so it can be
no more realistic than the simulation itself. A high score here says "the
model has learned the simulator's attack patterns", not "this would work on
real drones". Its main value is the feature-importance chart: it shows which
measurements carry the evidence.

Rule R1: features come only from an ObservedFrame and the outputs of
Detectors A, B and C. The attack labels (truth) are used ONLY as the target
y while building the training set, never as an input feature.
All numbers produced here are simulation results.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from analysis.calibration import load_or_default
from attacks.scenarios import get_scenario, scenario_keys
from config import DetectorConfig, SimConfig
from core.schema import ObservedFrame
from detection import build_detectors
from detection.base import (
    DetectionRun,
    Detector,
    PersistenceFilter,
    StepResult,
    status_from_flags,
)
from forensics.hashing import sha256_file
from simulation.swarm import simulate

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float], None] | None

FEATURE_NAMES: tuple[str, ...] = (
    "nis",
    "cusum",
    "nis_mean_w",
    "a_vote_fraction",
    "a_max_pair_residual",
    "c_residual",
    "has_anchor",
    "gps_speed",
    "gps_vel_mismatch",
    "gps_accel_norm",
    "heading_change",
)
WINDOW = 10
ML_THRESHOLD = 0.5
# WHY 100,000: calibration uses seeds 1000+, evaluation 5000+. Training
# seeds sit far above both, so the model never trains on a flight that is
# later used to calibrate or to evaluate.
TRAINING_SEED_START = 100_000
MODEL_DIR = Path("data") / "models"
MODEL_FILE = "rf.joblib"
HASH_FILE = "rf.sha256"
CONFIG_FILE = "rf_config.json"
RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
# WHY a minimum speed for headings: atan2 of a near-zero vector is pure
# noise, so a hovering drone would show random "turns" of up to 180 deg.
MIN_SPEED_FOR_HEADING = 0.5


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi):  wrapped = (a + pi) mod 2*pi - pi.

    WHY: a turn from +179 deg to -179 deg is a 2 deg turn, not 358 deg.
    """
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
class FeatureBuilder:
    """Turn one step of observations + A, B, C outputs into an (N, 11) matrix.

    It keeps a little history per drone: the last w NIS values, the previous
    GPS position, the previous GPS-derived velocity and its heading.

    GPS motion (finite differences, dt = time since the previous step):
        gps_vel_k   = (gps_k - gps_{k-1}) / dt
        gps_speed   = |gps_vel_k|
        mismatch    = |gps_vel_k - vel_meas_{k-1}|   (D4: vel_meas[k-1] is the
                      velocity flown during [t_{k-1}, t_k])
        accel_norm  = |gps_vel_k - gps_vel_{k-1}| / dt
        heading     = atan2(vy, vx);  heading_change = wrap(h_k - h_{k-1})

    WHY one class for training AND detection: the features used to train
    the model and those used later to apply it are computed by exactly the
    same code, so they can never quietly drift apart.
    """

    def __init__(self, n: int, window: int = WINDOW) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.n = n
        self.window = window
        self._nis_hist: deque[np.ndarray] = deque(maxlen=window)
        self._prev_t: float | None = None
        self._prev_gps: np.ndarray | None = None
        self._prev_vel_meas: np.ndarray | None = None
        self._prev_gps_vel: np.ndarray | None = None
        self._prev_heading: np.ndarray | None = None

    def _motion(self, obs: ObservedFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (gps_speed, mismatch, accel_norm, heading_change), each (N,)."""
        zeros = np.zeros(self.n)
        if self._prev_gps is None or self._prev_t is None:
            # First frame: no difference yet, so every motion feature is 0.
            return zeros, zeros.copy(), zeros.copy(), zeros.copy()
        dt = obs.t - self._prev_t
        if dt <= 0:
            raise ValueError(f"frames must be in time order (dt = {dt})")
        gps_vel = (obs.gps_pos - self._prev_gps) / dt
        speed = np.linalg.norm(gps_vel, axis=1)
        mismatch = np.linalg.norm(gps_vel - self._prev_vel_meas, axis=1)
        if self._prev_gps_vel is None:
            accel = zeros.copy()
        else:
            accel = np.linalg.norm(gps_vel - self._prev_gps_vel, axis=1) / dt
        heading = np.arctan2(gps_vel[:, 1], gps_vel[:, 0])
        if self._prev_heading is None:
            turn = zeros.copy()
        else:
            turn = wrap_angle(heading - self._prev_heading)
            turn[speed < MIN_SPEED_FOR_HEADING] = 0.0
        self._prev_gps_vel = gps_vel
        self._prev_heading = heading
        return speed, mismatch, accel, turn

    def update(
        self,
        obs: ObservedFrame,
        res_a: StepResult,
        res_b: StepResult,
        res_c: StepResult,
    ) -> np.ndarray:
        """Return the (N, 11) feature matrix for this step (columns = FEATURE_NAMES)."""
        if obs.n != self.n:
            raise ValueError(f"frame has {obs.n} drones, builder was made for {self.n}")
        # B: NIS is NaN on the first step (no prediction yet) -> 0.
        nis = np.nan_to_num(np.asarray(res_b.diagnostics["nis"], dtype=float), nan=0.0)
        cusum = np.asarray(res_b.diagnostics["cusum"], dtype=float)
        self._nis_hist.append(nis)
        nis_mean = np.mean(np.stack(self._nis_hist), axis=0)
        # A: vote fraction NaN (too few pairs) -> 0; max residual over valid pairs.
        vote = np.nan_to_num(res_a.scores, nan=0.0)
        resid = np.asarray(res_a.diagnostics["residuals"], dtype=float)
        has_pair = ~np.all(np.isnan(resid), axis=1)
        max_pair = np.zeros(self.n)
        max_pair[has_pair] = np.nanmax(resid[has_pair], axis=1)
        # C: residual where the drone has an anchor, else the sentinel -1.
        has_anchor = obs.anchor_mask.astype(float)
        c_res = np.where(obs.anchor_mask, np.nan_to_num(res_c.scores, nan=-1.0), -1.0)
        speed, mismatch, accel, turn = self._motion(obs)

        self._prev_t = obs.t
        self._prev_gps = obs.gps_pos.copy()
        self._prev_vel_meas = obs.vel_meas.copy()
        return np.column_stack([nis, cusum, nis_mean, vote, max_pair, c_res, has_anchor,
                                speed, mismatch, accel, turn])


def extract_features(
    frames: Sequence[ObservedFrame],
    det_cfg: DetectorConfig,
    sim_cfg: SimConfig,
) -> np.ndarray:
    """Run A, B, C over a whole flight and return features of shape (T, N, 11).

    Only the sensor noise levels are read from sim_cfg (the same datasheet
    values Detector B already uses); no truth enters.
    """
    if not frames:
        raise ValueError("extract_features needs at least one frame")
    for obs in frames:
        if not isinstance(obs, ObservedFrame):
            raise TypeError(f"features must not see truth: expected ObservedFrame, got {type(obs).__name__}")
    detectors = build_detectors(det_cfg, sim_cfg)
    builder = FeatureBuilder(frames[0].n)
    rows = []
    for obs in frames:
        res_a, res_b, res_c = (d.update(obs) for d in detectors)
        rows.append(builder.update(obs, res_a, res_b, res_c))
    return np.stack(rows)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
@dataclass
class MLDataset:
    """Training table: one row per (run, kept step, drone).

    X            (rows, 11) features
    y            (rows,) 1 while that drone is under attack (F5: the whole
                 attack window stays 1, not only the first step)
    groups       (rows,) run id; used for the group split (F6)
    attack_type  (rows,) the run's attack type ("none" for clean runs)
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    attack_type: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    config: dict[str, Any] = field(default_factory=dict)


def build_dataset(
    n_runs: int,
    scenarios: Sequence[str] | None = None,
    seed_start: int = TRAINING_SEED_START,
    step_stride: int = 2,
    progress: ProgressFn = None,
) -> MLDataset:
    """Simulate n_runs flights and turn them into a labelled table.

    Run i uses scenario scenarios[i mod len] with seed seed_start + i. The
    default scenario list is ALL scenarios, which includes the clean
    baseline and legitimate manoeuvres, so the model also sees plenty of
    "no attack" data, including hard negatives (turns that are not attacks).

    WHY step_stride (default 2): consecutive steps are 0.1 s apart and
    almost identical, so keeping every second one halves training time with
    almost no loss. Rolling features are still updated at EVERY step.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")
    if step_stride < 1:
        raise ValueError(f"step_stride must be >= 1, got {step_stride}")
    keys = list(scenarios) if scenarios else scenario_keys()
    xs, ys, gs, ts = [], [], [], []
    for i in range(n_runs):
        scenario = get_scenario(keys[i % len(keys)], seed=seed_start + i)
        result = simulate(scenario.sim, scenario.attack)
        det = load_or_default(scenario.sim)
        feats = extract_features(result.observed(), det, scenario.sim)[::step_stride]
        # WHY truth here is allowed: labels are the ANSWER the model learns
        # to predict. They are never fed in as an input feature.
        labels = result.stack("attack_labels")[::step_stride]
        xs.append(feats.reshape(-1, len(FEATURE_NAMES)))
        ys.append(labels.reshape(-1).astype(int))
        gs.append(np.full(labels.size, i, dtype=int))
        ts.append(np.full(labels.size, scenario.attack.type, dtype=object))
        if progress is not None:
            progress((i + 1) / n_runs)
    config = {"n_runs": n_runs, "scenarios": keys, "seed_start": seed_start,
              "seed_end": seed_start + n_runs - 1, "step_stride": step_stride, "window": WINDOW}
    logger.info("built ML dataset: %d runs, %d rows", n_runs, sum(len(y) for y in ys))
    return MLDataset(np.concatenate(xs), np.concatenate(ys), np.concatenate(gs),
                     np.concatenate(ts), FEATURE_NAMES, config)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainResult:
    """A fitted forest plus its held-out (test) metrics. Simulation results."""

    model: RandomForestClassifier
    metrics: dict[str, float]
    confusion: list[list[int]]
    feature_importances: dict[str, float]
    per_type_recall: dict[str, float]
    clean_false_positive_rate: float
    train_groups: list[int]
    test_groups: list[int]
    n_train: int
    n_test: int
    rf_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Everything except the model itself, ready for JSON."""
        return {
            "metrics": self.metrics, "confusion": self.confusion,
            "feature_importances": self.feature_importances,
            "per_type_recall": self.per_type_recall,
            "clean_false_positive_rate": self.clean_false_positive_rate,
            "train_groups": self.train_groups, "test_groups": self.test_groups,
            "n_train": self.n_train, "n_test": self.n_test, "rf_params": self.rf_params,
        }


def group_split(groups: np.ndarray, test_size: float = 0.25, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split ROW indices so every run lands entirely in train OR in test (F6).

    WHY: rows from one flight are near-copies of each other (0.1 s apart).
    A row-level split would put almost the same row in both sets, and the
    test score would measure memory, not generalisation.
    """
    if len(np.unique(groups)) < 2:
        raise ValueError("need at least 2 runs (groups) to make a train/test split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx


def _per_type_recall(y: np.ndarray, pred: np.ndarray, types: np.ndarray) -> dict[str, float]:
    """Recall on attacked rows, separately for each attack type in the test set."""
    out: dict[str, float] = {}
    for t in sorted({str(v) for v in types[y == 1]}):
        mask = (y == 1) & (types == t)
        out[t] = float(pred[mask].mean())
    return out


def train(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    attack_type: np.ndarray | None = None,
    test_size: float = 0.25,
    rf_params: dict[str, Any] | None = None,
) -> TrainResult:
    """Group-split by run, fit the forest on train, score it on test.

    Metrics (test set only):
        accuracy  = (TP + TN) / all       precision = TP / (TP + FP)
        recall    = TP / (TP + FN)        F1 = 2 P R / (P + R)
        ROC-AUC   = chance a random attacked row scores above a random clean row
    class_weight="balanced" weights each class by 1 / its frequency, so the
    many clean rows do not drown out the fewer attacked ones.
    """
    X, y, groups = np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(groups)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError(f"X, y and groups must have the same length, got {len(X)}, {len(y)}, {len(groups)}")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("training data needs both clean (0) and attacked (1) rows")
    params = {**RF_PARAMS, **(rf_params or {})}
    train_idx, test_idx = group_split(groups, test_size)
    model = RandomForestClassifier(**params).fit(X[train_idx], y[train_idx])
    y_test = y[test_idx]
    proba = model.predict_proba(X[test_idx])[:, list(model.classes_).index(1)]
    pred = (proba > ML_THRESHOLD).astype(int)
    two_classes = len(np.unique(y_test)) == 2
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        # WHY NaN: ROC-AUC is undefined if the test set has only one class.
        "roc_auc": float(roc_auc_score(y_test, proba)) if two_classes else float("nan"),
    }
    types = np.asarray(attack_type, dtype=object)[test_idx] if attack_type is not None \
        else np.full(len(test_idx), "unknown", dtype=object)
    clean_rows = types == "none"
    return TrainResult(
        model=model,
        metrics=metrics,
        confusion=confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        feature_importances={n: float(v) for n, v in zip(FEATURE_NAMES, model.feature_importances_)},
        per_type_recall=_per_type_recall(y_test, pred, types),
        clean_false_positive_rate=float(pred[clean_rows].mean()) if clean_rows.any() else float("nan"),
        train_groups=sorted(int(g) for g in np.unique(groups[train_idx])),
        test_groups=sorted(int(g) for g in np.unique(groups[test_idx])),
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        rf_params=params,
    )


# --------------------------------------------------------------------------
# Saving and loading (hash-checked)
# --------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """A loaded forest plus the information saved next to it."""

    model: RandomForestClassifier
    info: dict[str, Any]
    sha256: str
    path: Path


def save_model(result: TrainResult, dataset_config: dict[str, Any], directory: Path = MODEL_DIR) -> Path:
    """Write rf.joblib, rf.sha256 and rf_config.json; return the model path.

    WHY store the SHA-256: a .joblib file is a pickle, and loading a pickle
    can run code. load_model() re-hashes the file and refuses it if it has
    changed since training (the same idea as our evidence packages).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILE
    joblib.dump(result.model, path)
    digest = sha256_file(path)
    (directory / HASH_FILE).write_text(f"{digest}  {MODEL_FILE}\n", encoding="utf-8")
    info = {
        "label": "EXPERIMENTAL - simulation result",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": digest, "feature_names": list(FEATURE_NAMES), "threshold": ML_THRESHOLD,
        "sklearn_version": sklearn.__version__, "dataset": dataset_config, **result.to_dict(),
    }
    (directory / CONFIG_FILE).write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    logger.info("saved ML model to %s (sha256 %s)", path, digest)
    return path


def load_model(directory: Path = MODEL_DIR) -> TrainedModel | None:
    """Load the saved model, or None if none has been trained yet.

    Raises ValueError if the file's SHA-256 does not match the recorded one.
    """
    directory = Path(directory)
    path, hash_path, cfg_path = directory / MODEL_FILE, directory / HASH_FILE, directory / CONFIG_FILE
    if not (path.exists() and hash_path.exists() and cfg_path.exists()):
        return None
    expected = hash_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"model file {path} has changed since training "
                         f"(sha256 {actual[:12]}... != recorded {expected[:12]}...); retrain it")
    info = json.loads(cfg_path.read_text(encoding="utf-8"))
    if info.get("sklearn_version") != sklearn.__version__:
        logger.warning("model trained with scikit-learn %s, running %s",
                       info.get("sklearn_version"), sklearn.__version__)
    model = joblib.load(path)
    # WHY n_jobs=1: at detection time we predict small batches; starting a
    # pool of worker processes for each call would cost more than it saves.
    model.set_params(n_jobs=1)
    return TrainedModel(model, info, actual, path)


def train_and_save(
    n_runs: int,
    seed_start: int = TRAINING_SEED_START,
    directory: Path = MODEL_DIR,
    rf_params: dict[str, Any] | None = None,
    progress: ProgressFn = None,
) -> tuple[TrainResult, Path]:
    """build_dataset -> train -> save_model, used by the CLI and the app."""
    data = build_dataset(n_runs, seed_start=seed_start, progress=progress)
    result = train(data.X, data.y, data.groups, data.attack_type, rf_params=rf_params)
    return result, save_model(result, data.config, directory)


# --------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------
def positive_proba(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """P(attacked) for each row of X."""
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


class MLDetector(Detector):
    """Detector D (experimental): raw flag = P(attacked) > 0.5, then m-of-n persistence.

    Score = P(attacked) per drone. Every drone always has evidence, because
    the GPS and velocity features exist for every drone at every step.
    It runs its own private copies of A, B and C to build its features.
    """

    name = "Experimental ML"
    short = "D"

    def __init__(self, model: RandomForestClassifier, det_cfg: DetectorConfig,
                 sim_cfg: SimConfig, threshold: float = ML_THRESHOLD) -> None:
        super().__init__()
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.model = model
        self.det_cfg = det_cfg
        self.sim_cfg = sim_cfg
        self.threshold = threshold

    def _reset_state(self, n: int) -> None:
        self._inner = build_detectors(self.det_cfg, self.sim_cfg)
        self._features = FeatureBuilder(n)
        self._persist = PersistenceFilter(self.det_cfg.persist_m, self.det_cfg.persist_n, n)

    def _result(self, obs: ObservedFrame, features: np.ndarray, proba: np.ndarray) -> StepResult:
        raw = proba > self.threshold
        flags = self._persist.update(raw)
        return StepResult(
            step=obs.step, t=obs.t, flags=flags, raw_flags=raw, scores=proba,
            status=status_from_flags(flags, raw, np.ones(obs.n, dtype=bool)),
            swarm_flag=bool(flags.any()), swarm_score=float(np.max(proba)),
            diagnostics={"probability": proba, "features": features},
        )

    def _step(self, obs: ObservedFrame) -> StepResult:
        res_a, res_b, res_c = (d.update(obs) for d in self._inner)
        features = self._features.update(obs, res_a, res_b, res_c)
        return self._result(obs, features, positive_proba(self.model, features))

    def run(self, frames: Sequence[ObservedFrame]) -> DetectionRun:
        """Same answer as calling update() frame by frame, but much faster.

        WHY override: asking a 200-tree forest for 10 predictions, 600 times,
        is dominated by per-call overhead. Here all features are built
        first and the forest is asked ONCE for all T x N rows.
        """
        feats = extract_features(frames, self.det_cfg, self.sim_cfg)
        t_count, n, k = feats.shape
        proba = positive_proba(self.model, feats.reshape(-1, k)).reshape(t_count, n)
        self.reset(n)
        steps = [self._result(obs, feats[i], proba[i]) for i, obs in enumerate(frames)]
        return DetectionRun(name=self.name, short=self.short, steps=steps)


def run_ml(frames: Sequence[ObservedFrame], trained: TrainedModel,
           det_cfg: DetectorConfig, sim_cfg: SimConfig) -> DetectionRun:
    """Convenience: run Detector D over one flight with a loaded model."""
    return MLDetector(trained.model, det_cfg, sim_cfg, trained.info.get("threshold", ML_THRESHOLD)).run(frames)
