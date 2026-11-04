"""Core data model: what the simulator produces and what detectors may see.

The single most important rule of the project (R1, ground-truth separation)
is enforced here by the TYPES themselves:

    ObservedFrame  - only what a real drone could measure at one time step:
                     GPS, velocity sensor, radio ranges, anchor readings.
                     It has NO field for true position, true velocity or
                     attack labels.
    Frame          - one simulated time step for evaluation. It CONTAINS an
                     ObservedFrame (frame.observed) plus the hidden truth.

Detectors, fusion and recovery are written to accept an ObservedFrame. If a
detector could read true positions, detection would be trivial and every
accuracy number would be meaningless; real drones never know their true
position.

WHY composition (Frame has-an ObservedFrame) and not inheritance (Frame
is-an ObservedFrame): with inheritance, a Frame would also count as an
ObservedFrame, so passing a whole Frame (truth included) to a detector would
be accepted silently. With composition the caller must write frame.observed,
which makes the separation visible in the code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from enum import Enum

import numpy as np
import pandas as pd

from config import AttackConfig, SimConfig

logger = logging.getLogger(__name__)


class DroneStatus(str, Enum):
    """Per-drone, per-step output of a single detector.

    NO_EVIDENCE means the detector had no usable data for this drone (for
    example Detector C on a drone without an anchor). It is deliberately
    different from NORMAL (rule R4): "I could not check" is not "I checked
    and it looked fine".
    """

    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ALARM = "alarm"
    NO_EVIDENCE = "no_evidence"


class Verdict(str, Enum):
    """Fused verdict for one drone after combining all detectors."""

    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    COMPROMISED = "compromised"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AttackType(str, Enum):
    """Names of the simulated GPS attacks (same strings as AttackConfig.type)."""

    NONE = "none"
    CONSTANT_OFFSET = "constant_offset"
    GRADUAL_DRIFT = "gradual_drift"
    MULTI_DRONE = "multi_drone"
    SWARM_WIDE = "swarm_wide"
    SUBSET_RIGID = "subset_rigid"
    RIGID_ROTATION = "rigid_rotation"
    INTERMITTENT = "intermittent"


def _readonly_copy(array: np.ndarray, dtype: type) -> np.ndarray:
    """Return a private, read-only copy of an array.

    WHY copy: the simulator keeps updating its own position arrays every
    step; a copy guarantees a stored frame never changes afterwards.
    WHY read-only: a detector that accidentally writes into shared data
    (e.g. gps_pos -= bias) would corrupt the input of every other detector.
    NumPy raises an error instead.
    """
    out = np.array(array, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def _check_shape(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    """Raise ValueError if array does not have the expected shape."""
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")


@dataclass(frozen=True)
class ObservedFrame:
    """Everything a real swarm could measure at one time step (N drones).

    step         integer step index k
    t            time in seconds, t = k * dt
    gps_pos      (N, 2) reported GPS positions (may be spoofed), metres
    vel_meas     (N, 2) velocity sensor readings, m/s. vel_meas[k] measures
                 the velocity flown during [t_k, t_k + dt], so
                 true_pos[k+1] = true_pos[k] + true_vel[k] * dt.
    ranges       (N, N) radio range between each pair, metres; symmetric,
                 diagonal 0, NaN where the pair is out of radio range
    range_mask   (N, N) True where ranges[i, j] is a valid measurement
    anchor_pos   (N, 2) absolute anchor position fix; NaN rows where the
                 drone had no reading this step
    anchor_mask  (N,) True where the drone has an anchor reading this step
    """

    step: int
    t: float
    gps_pos: np.ndarray
    vel_meas: np.ndarray
    ranges: np.ndarray
    range_mask: np.ndarray
    anchor_pos: np.ndarray
    anchor_mask: np.ndarray

    def __post_init__(self) -> None:
        n = np.asarray(self.gps_pos).shape[0]
        specs = {
            "gps_pos": ((n, 2), np.float64),
            "vel_meas": ((n, 2), np.float64),
            "ranges": ((n, n), np.float64),
            "range_mask": ((n, n), np.bool_),
            "anchor_pos": ((n, 2), np.float64),
            "anchor_mask": ((n,), np.bool_),
        }
        for name, (shape, dtype) in specs.items():
            arr = _readonly_copy(getattr(self, name), dtype)
            _check_shape(f"ObservedFrame.{name}", arr, shape)
            # WHY object.__setattr__: the dataclass is frozen, so normal
            # assignment is blocked; this is the standard way to set fields
            # once, inside __post_init__.
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "t", float(self.t))

    @property
    def n(self) -> int:
        """Number of drones N."""
        return int(self.gps_pos.shape[0])


@dataclass(frozen=True)
class Frame:
    """One simulated time step: observations plus hidden truth.

    observed       the ObservedFrame (the only part detectors receive)
    --- EVALUATION ONLY: never passed to detectors, fusion or recovery ---
    true_pos       (N, 2) true positions, metres
    true_vel       (N, 2) true velocity flown during [t, t + dt], m/s
    attack_offset  (N, 2) false GPS offset added by the attack, metres
    attack_labels  (N,) True for every drone affected by an active attack
    faulty_anchor  (N,) True where the drone's anchor is biased (faulty)
    --- known plan information ---
    slot_pos       (N, 2) intended formation slot of each drone
    anchored       (N,) which drones carry an anchor sensor
    """

    observed: ObservedFrame
    true_pos: np.ndarray
    true_vel: np.ndarray
    attack_offset: np.ndarray
    attack_labels: np.ndarray
    faulty_anchor: np.ndarray
    slot_pos: np.ndarray
    anchored: np.ndarray

    def __post_init__(self) -> None:
        n = self.observed.n
        specs = {
            "true_pos": ((n, 2), np.float64),
            "true_vel": ((n, 2), np.float64),
            "attack_offset": ((n, 2), np.float64),
            "attack_labels": ((n,), np.bool_),
            "faulty_anchor": ((n,), np.bool_),
            "slot_pos": ((n, 2), np.float64),
            "anchored": ((n,), np.bool_),
        }
        for name, (shape, dtype) in specs.items():
            arr = _readonly_copy(getattr(self, name), dtype)
            _check_shape(f"Frame.{name}", arr, shape)
            object.__setattr__(self, name, arr)

    @property
    def step(self) -> int:
        """Step index (taken from the observed part)."""
        return self.observed.step

    @property
    def t(self) -> float:
        """Time in seconds (taken from the observed part)."""
        return self.observed.t


# Names of the truth fields. Tests use this to prove ObservedFrame has none.
TRUTH_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(Frame) if f.name not in ("observed", "slot_pos", "anchored")
)


@dataclass
class SimResult:
    """The full output of one simulation run.

    config   the SimConfig used
    attack   the AttackConfig used
    frames   one Frame per time step, in time order
    """

    config: SimConfig
    attack: AttackConfig
    frames: list[Frame]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("SimResult needs at least one frame")

    @property
    def n_drones(self) -> int:
        """Number of drones N."""
        return self.frames[0].observed.n

    @property
    def n_steps(self) -> int:
        """Number of stored time steps T."""
        return len(self.frames)

    def times(self) -> np.ndarray:
        """Return the (T,) array of step times in seconds."""
        return np.array([f.t for f in self.frames], dtype=float)

    def observed(self) -> list[ObservedFrame]:
        """Return only the observation part of every frame (detector input)."""
        return [f.observed for f in self.frames]

    def stack(self, name: str) -> np.ndarray:
        """Stack one per-frame array over time, e.g. stack("true_pos") -> (T, N, 2).

        Looks the name up on the Frame first, then on the ObservedFrame.
        """
        first = self.frames[0]
        if hasattr(first, name) and isinstance(getattr(first, name), np.ndarray):
            return np.stack([getattr(f, name) for f in self.frames])
        if hasattr(first.observed, name):
            return np.stack([getattr(f.observed, name) for f in self.frames])
        raise ValueError(f"unknown frame array {name!r}")

    def _index_columns(self) -> dict[str, np.ndarray]:
        """Return step, t and drone columns for long-format tables (T*N rows)."""
        t_count, n = self.n_steps, self.n_drones
        steps = np.array([f.step for f in self.frames])
        return {
            "step": np.repeat(steps, n),
            "t": np.repeat(self.times(), n),
            "drone": np.tile(np.arange(n), t_count),
        }

    def to_observed_dataframe(self) -> pd.DataFrame:
        """Long table of observations: one row per (step, drone).

        Columns: step, t, drone, gps_x, gps_y, vel_x, vel_y, anchor_x,
        anchor_y, has_anchor. Ranges are (N, N) per step and are available
        separately through ranges_array().
        """
        gps = self.stack("gps_pos").reshape(-1, 2)
        vel = self.stack("vel_meas").reshape(-1, 2)
        anc = self.stack("anchor_pos").reshape(-1, 2)
        cols = self._index_columns()
        cols.update(
            gps_x=gps[:, 0], gps_y=gps[:, 1],
            vel_x=vel[:, 0], vel_y=vel[:, 1],
            anchor_x=anc[:, 0], anchor_y=anc[:, 1],
            has_anchor=self.stack("anchor_mask").reshape(-1),
        )
        return pd.DataFrame(cols)

    def to_truth_dataframe(self) -> pd.DataFrame:
        """Long table of hidden truth (EVALUATION ONLY): one row per (step, drone).

        Columns: step, t, drone, true_x, true_y, true_vx, true_vy, off_x,
        off_y, attacked, slot_x, slot_y.
        """
        pos = self.stack("true_pos").reshape(-1, 2)
        vel = self.stack("true_vel").reshape(-1, 2)
        off = self.stack("attack_offset").reshape(-1, 2)
        slot = self.stack("slot_pos").reshape(-1, 2)
        cols = self._index_columns()
        cols.update(
            true_x=pos[:, 0], true_y=pos[:, 1],
            true_vx=vel[:, 0], true_vy=vel[:, 1],
            off_x=off[:, 0], off_y=off[:, 1],
            attacked=self.stack("attack_labels").reshape(-1),
            slot_x=slot[:, 0], slot_y=slot[:, 1],
        )
        return pd.DataFrame(cols)

    def ranges_array(self) -> np.ndarray:
        """Return all range matrices stacked over time, shape (T, N, N)."""
        return self.stack("ranges")

    def attack_onset(self) -> float | None:
        """Return the first time any drone is labelled attacked, or None."""
        for frame in self.frames:
            if frame.attack_labels.any():
                return frame.t
        return None

    def attacked_drones(self) -> set[int]:
        """Return the set of drone indices that were attacked at any time."""
        labels = self.stack("attack_labels")
        return {int(i) for i in np.flatnonzero(labels.any(axis=0))}
