"""Configuration dataclasses for SwarmGuard.

Every run of the platform is fully described by four small configuration
objects:

    SimConfig        how the swarm flies and how noisy its sensors are
    AttackConfig     what (simulated) GPS spoofing is applied, if any
    DetectorConfig   detector thresholds (changed only by calibration, F3)
    ExperimentConfig how many random seeds a statistical experiment uses

WHY dataclasses: they give named, typed fields with defaults, a readable
repr, and easy conversion to a dictionary (dataclasses.asdict). That makes
every run reproducible: the same config plus the same seed always gives the
same output (rule R2), and the config can be saved next to results.

Every config can be turned into canonical JSON and hashed with SHA-256:

    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    config_hash = SHA-256(canonical)

WHY canonical JSON: sorting the keys and removing optional whitespace means
two equal configs always produce byte-identical JSON, so equal configs have
equal hashes. The hash is later stored in incident records and evidence
packages to prove which settings produced a result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Self

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Allowed values for string fields. Kept as tuples so they cannot be changed.
# --------------------------------------------------------------------------
VALID_FORMATIONS: tuple[str, ...] = ("circle", "line", "grid", "star", "heart", "custom")
VALID_PATHS: tuple[str, ...] = ("straight", "loop", "waypoints")
VALID_LOOP_MODES: tuple[str, ...] = ("open", "closed")
VALID_MANOEUVRES: tuple[str, ...] = ("rotate", "scale")
VALID_ATTACK_TYPES: tuple[str, ...] = (
    "none",
    "constant_offset",
    "gradual_drift",
    "multi_drone",
    "swarm_wide",
    "subset_rigid",
    "rigid_rotation",
    "intermittent",
)
VALID_RAMP_PROFILES: tuple[str, ...] = ("step", "linear", "exponential", "polynomial", "sigmoid")
VALID_AGGREGATES: tuple[str, ...] = ("median", "mean", "max")

# Attack types that act on exactly one listed target drone.
SINGLE_TARGET_ATTACKS: tuple[str, ...] = ("constant_offset", "gradual_drift", "intermittent")
# Attack types that act on one or more listed target drones.
MULTI_TARGET_ATTACKS: tuple[str, ...] = ("multi_drone", "subset_rigid")

MIN_DRONES = 4
MAX_DRONES = 30


# --------------------------------------------------------------------------
# Small validation helpers (used by every config class)
# --------------------------------------------------------------------------
def _as_float(name: str, value: Any) -> float:
    """Convert a numeric value to float, or raise a clear ValueError.

    WHY: storing 10 and 10.0 differently would give two different JSON
    strings, and therefore two different config hashes, for the same
    setting. Converting every real-valued field to float avoids that.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _as_int(name: str, value: Any) -> int:
    """Convert a whole-number value to int, or raise a clear ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def _check_choice(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    """Raise ValueError unless value is one of the allowed strings."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")
    return str(value)


def _check_range(name: str, value: float, low: float, high: float) -> None:
    """Raise ValueError unless low <= value <= high."""
    if not (low <= value <= high):
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")


def _as_point(name: str, value: Any) -> tuple[float, float]:
    """Convert a 2-element sequence (from Python or JSON) to (x, y) floats."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be an (x, y) pair, got {value!r}")
    return (_as_float(f"{name}.x", value[0]), _as_float(f"{name}.y", value[1]))


class _ConfigMixin:
    """Shared dict / JSON / hash behaviour for every config dataclass.

    Each config class validates and normalises itself in __post_init__, so
    from_dict only needs to reject unknown keys and call the constructor.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary (nested dataclasses become dicts)."""
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a config from a dictionary, rejecting unknown keys.

        WHY reject unknown keys: a typo such as "gps_sigmaa" would otherwise
        be silently ignored and the run would use the default value.
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"{cls.__name__}.from_dict expects a mapping, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown field(s) {unknown}")
        return cls(**dict(data))

    def to_json(self, indent: int | None = None) -> str:
        """Return JSON with sorted keys (readable when indent is given)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Build a config from a JSON string produced by to_json."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{cls.__name__}: invalid JSON ({exc})") from exc
        return cls.from_dict(data)

    def config_hash(self) -> str:
        """Return the SHA-256 hex digest of the canonical JSON form.

        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Simulation configuration
# --------------------------------------------------------------------------
@dataclass
class ManoeuvreEvent(_ConfigMixin):
    """A legitimate formation change: a rotation or a scaling.

    kind  = "rotate": the formation turns by `value` degrees in total,
            progressing linearly from `start` to `end` (seconds).
    kind  = "scale":  the formation size is multiplied by a factor that
            grows linearly from 1 at `start` to `value` at `end`.

    WHY: legitimate formation changes create GPS motion that is not an
    attack. Including them lets us measure false alarms honestly instead of
    only testing perfectly boring flights.
    """

    kind: str
    start: float
    end: float
    value: float

    def __post_init__(self) -> None:
        self.kind = _check_choice("ManoeuvreEvent.kind", self.kind, VALID_MANOEUVRES)
        self.start = _as_float("ManoeuvreEvent.start", self.start)
        self.end = _as_float("ManoeuvreEvent.end", self.end)
        self.value = _as_float("ManoeuvreEvent.value", self.value)
        if self.start < 0:
            raise ValueError(f"ManoeuvreEvent.start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"ManoeuvreEvent.end ({self.end}) must be greater than start ({self.start})"
            )
        if self.kind == "scale" and self.value <= 0:
            raise ValueError(f"scale ManoeuvreEvent.value must be > 0, got {self.value}")


@dataclass
class SimConfig(_ConfigMixin):
    """How the swarm flies and how its sensors behave.

    Units: metres, seconds, metres/second, degrees. Coordinates use a local
    East-North frame (x = east, y = north), see core/utils.py.

    Sensor noise model (each axis independent, Gaussian):
        GPS       sigma = gps_sigma     (m)
        ranging   sigma = range_sigma   (m, on each pairwise distance)
        velocity  sigma = vel_sigma     (m/s)
        anchors   sigma = anchor_sigma  (m)

    Control:
        loop_mode "open":   drones steer using their TRUE position.
        loop_mode "closed": drones steer using their (possibly spoofed) GPS,
                            so attacks can change the true flight path.
        centre_gain: how strongly (1/s) the closed-loop swarm pulls its
                     GPS-estimated centre back onto the planned path. Set it
                     to 0 to get pure formation keeping with no path
                     correction. See simulation/swarm.py for the equations.
    """

    n_drones: int = 10
    formation: str = "circle"
    radius: float = 8.0
    duration: float = 60.0
    dt: float = 0.1
    center_speed: float = 2.0
    path: str = "straight"
    heading_deg: float = 0.0
    loop_radius: float = 40.0
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    max_speed: float = 6.0
    kp: float = 1.2
    loop_mode: str = "closed"
    centre_gain: float = 0.5
    comm_range: float = 50.0
    gps_sigma: float = 1.0
    range_sigma: float = 0.3
    vel_sigma: float = 0.1
    anchor_ratio: float = 0.3
    anchor_sigma: float = 0.5
    anchor_dropout: float = 0.0
    anchor_fault_ratio: float = 0.0
    anchor_fault_bias: float = 15.0
    manoeuvres: list[ManoeuvreEvent] = field(default_factory=list)
    shape_sequence: list[tuple[float, str]] = field(default_factory=list)
    custom_points: list[tuple[float, float]] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self) -> None:
        self._validate_swarm()
        self._validate_path()
        self._validate_sensors()
        self._validate_events()
        self.seed = _as_int("seed", self.seed)
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    @property
    def n_steps(self) -> int:
        """Number of simulation steps: round(duration / dt). 60 / 0.1 = 600."""
        return int(round(self.duration / self.dt))

    def _validate_swarm(self) -> None:
        """Check swarm size, formation and controller settings."""
        self.n_drones = _as_int("n_drones", self.n_drones)
        if not (MIN_DRONES <= self.n_drones <= MAX_DRONES):
            raise ValueError(
                f"n_drones must be between {MIN_DRONES} and {MAX_DRONES}, got {self.n_drones}"
            )
        self.formation = _check_choice("formation", self.formation, VALID_FORMATIONS)
        self.custom_points = [
            _as_point(f"custom_points[{i}]", p) for i, p in enumerate(self.custom_points)
        ]
        if self.formation == "custom" and len(self.custom_points) != self.n_drones:
            raise ValueError(
                f"formation 'custom' needs exactly n_drones={self.n_drones} custom_points, "
                f"got {len(self.custom_points)}"
            )
        for name in ("radius", "max_speed", "kp"):
            value = _as_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
            setattr(self, name, value)
        self.centre_gain = _as_float("centre_gain", self.centre_gain)
        if self.centre_gain < 0:
            raise ValueError(f"centre_gain must be >= 0, got {self.centre_gain}")
        self.loop_mode = _check_choice("loop_mode", self.loop_mode, VALID_LOOP_MODES)

    def _validate_path(self) -> None:
        """Check timing and the swarm-centre path settings."""
        self.duration = _as_float("duration", self.duration)
        self.dt = _as_float("dt", self.dt)
        if self.duration <= 0:
            raise ValueError(f"duration must be > 0, got {self.duration}")
        if self.dt <= 0 or self.dt > self.duration:
            raise ValueError(f"dt must be > 0 and <= duration, got {self.dt}")
        self.center_speed = _as_float("center_speed", self.center_speed)
        if self.center_speed < 0:
            raise ValueError(f"center_speed must be >= 0, got {self.center_speed}")
        if self.center_speed > self.max_speed:
            raise ValueError(
                f"center_speed ({self.center_speed}) cannot exceed max_speed ({self.max_speed})"
            )
        self.path = _check_choice("path", self.path, VALID_PATHS)
        self.heading_deg = _as_float("heading_deg", self.heading_deg)
        self.loop_radius = _as_float("loop_radius", self.loop_radius)
        if self.loop_radius <= 0:
            raise ValueError(f"loop_radius must be > 0, got {self.loop_radius}")
        self.waypoints = [_as_point(f"waypoints[{i}]", p) for i, p in enumerate(self.waypoints)]
        if self.path == "waypoints" and len(self.waypoints) < 2:
            raise ValueError("path 'waypoints' needs at least 2 waypoints")

    def _validate_sensors(self) -> None:
        """Check sensor noise, communication and anchor settings."""
        for name in ("comm_range",):
            value = _as_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
            setattr(self, name, value)
        for name in ("gps_sigma", "range_sigma", "vel_sigma", "anchor_sigma", "anchor_fault_bias"):
            value = _as_float(name, getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
            setattr(self, name, value)
        for name in ("anchor_ratio", "anchor_dropout", "anchor_fault_ratio"):
            value = _as_float(name, getattr(self, name))
            _check_range(name, value, 0.0, 1.0)
            setattr(self, name, value)

    def _validate_events(self) -> None:
        """Normalise manoeuvres and shape changes (JSON lists -> objects)."""
        events: list[ManoeuvreEvent] = []
        for i, event in enumerate(self.manoeuvres):
            if isinstance(event, ManoeuvreEvent):
                events.append(event)
            elif isinstance(event, Mapping):
                events.append(ManoeuvreEvent.from_dict(event))
            else:
                raise ValueError(f"manoeuvres[{i}] must be a ManoeuvreEvent or dict, got {event!r}")
        self.manoeuvres = events

        sequence: list[tuple[float, str]] = []
        for i, item in enumerate(self.shape_sequence):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"shape_sequence[{i}] must be a (time, shape) pair, got {item!r}")
            time = _as_float(f"shape_sequence[{i}].time", item[0])
            shape = _check_choice(f"shape_sequence[{i}].shape", item[1], VALID_FORMATIONS)
            if shape == "custom" and len(self.custom_points) != self.n_drones:
                raise ValueError("shape 'custom' in shape_sequence needs n_drones custom_points")
            if sequence and time <= sequence[-1][0]:
                raise ValueError("shape_sequence times must be strictly increasing")
            if time < 0:
                raise ValueError(f"shape_sequence[{i}] time must be >= 0, got {time}")
            sequence.append((time, shape))
        self.shape_sequence = sequence


# --------------------------------------------------------------------------
# Attack configuration
# --------------------------------------------------------------------------
@dataclass
class AttackConfig(_ConfigMixin):
    """Simulated GPS spoofing: false offsets added to REPORTED GPS only.

    This is pure arithmetic on numbers inside the simulation. Nothing here
    generates radio signals or interacts with any real receiver.

    Fields:
        type           attack name, see VALID_ATTACK_TYPES
        targets        0-based drone indices (the UI shows index 0 as "Drone 01")
        magnitude_m    size of the false offset (m)
        direction_deg  direction of the offset, 0 = east, counter-clockwise
        start_time     when the attack begins (s)
        end_time       when it stops (s); None = until the end of the run
        drift_rate     speed of drifting attacks (m/s)
        drift_cap      if True, a drift stops growing at magnitude_m
        ramp_profile   how a step attack grows from 0 to full size
        ramp_duration  time (s) to reach full size for non-step profiles
        rotation_rate_deg_s  rigid_rotation speed (degrees per second)
        pulse_period, pulse_duty  intermittent attack on/off cycle
        seed           seeds random parts of the attack (multi_drone)
    """

    type: str = "none"
    targets: list[int] = field(default_factory=list)
    magnitude_m: float = 10.0
    direction_deg: float = 0.0
    start_time: float = 20.0
    end_time: float | None = None
    drift_rate: float = 0.5
    drift_cap: bool = False
    ramp_profile: str = "step"
    ramp_duration: float = 10.0
    rotation_rate_deg_s: float = 1.0
    pulse_period: float = 4.0
    pulse_duty: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        self.type = _check_choice("type", self.type, VALID_ATTACK_TYPES)
        self.ramp_profile = _check_choice("ramp_profile", self.ramp_profile, VALID_RAMP_PROFILES)
        self.targets = [_as_int(f"targets[{i}]", t) for i, t in enumerate(self.targets)]
        if any(t < 0 for t in self.targets):
            raise ValueError(f"targets must be >= 0, got {self.targets}")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(f"targets must not repeat, got {self.targets}")
        if self.type in SINGLE_TARGET_ATTACKS and len(self.targets) != 1:
            raise ValueError(f"attack '{self.type}' needs exactly 1 target, got {self.targets}")
        if self.type in MULTI_TARGET_ATTACKS and len(self.targets) < 1:
            raise ValueError(f"attack '{self.type}' needs at least 1 target")
        self._validate_numbers()

    def _validate_numbers(self) -> None:
        """Check every numeric attack parameter."""
        for name in ("magnitude_m", "start_time", "drift_rate"):
            value = _as_float(name, getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
            setattr(self, name, value)
        for name in ("ramp_duration", "pulse_period"):
            value = _as_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
            setattr(self, name, value)
        self.direction_deg = _as_float("direction_deg", self.direction_deg)
        self.rotation_rate_deg_s = _as_float("rotation_rate_deg_s", self.rotation_rate_deg_s)
        self.pulse_duty = _as_float("pulse_duty", self.pulse_duty)
        if not (0.0 < self.pulse_duty <= 1.0):
            raise ValueError(f"pulse_duty must be in (0, 1], got {self.pulse_duty}")
        if self.end_time is not None:
            self.end_time = _as_float("end_time", self.end_time)
            if self.end_time <= self.start_time:
                raise ValueError(
                    f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
                )
        if not isinstance(self.drift_cap, bool):
            raise ValueError(f"drift_cap must be True or False, got {self.drift_cap!r}")
        self.seed = _as_int("seed", self.seed)
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")


# --------------------------------------------------------------------------
# Detector configuration
# --------------------------------------------------------------------------
@dataclass
class DetectorConfig(_ConfigMixin):
    """Detector thresholds. Changed ONLY by an explicit calibration load (F3).

    WHY F3: earlier projects silently overwrote thresholds while running,
    so reported accuracy did not match the thresholds a reader saw. Here a
    threshold changes only when the user deliberately loads a calibration.

    The defaults below are analytical starting values. Phase 2 calibration
    replaces them with values measured on clean simulated flights.

    tau_a = 4.3 m (Detector A, pair distance error):
        GPS-derived distance minus radio range is approximately the
        projection of (n_i - n_j) onto the line between the drones, plus
        range noise. Per-axis GPS sigma = 1 m, so the difference of two
        independent errors has variance 1 + 1 = 2 m^2. Adding range
        variance 0.3^2 = 0.09 gives sigma = sqrt(2.09) = 1.45 m.
        3 sigma = 4.3 m.

    nis_threshold = gate_nis = 9.21 (Detector B, Kalman innovation):
        The normalised innovation squared (NIS) of a 2D Gaussian innovation
        follows a chi-square distribution with 2 degrees of freedom, whose
        tail is P(NIS > x) = exp(-x / 2). Setting this to 0.01 gives
        x = -2 ln(0.01) = 9.21.

    tau_c = 4.2 m (Detector C, GPS minus anchor):
        Per-axis sigma = sqrt(gps_sigma^2 + anchor_sigma^2)
                       = sqrt(1 + 0.25) = 1.118 m.
        The length of a 2D vector of two independent N(0, s^2) parts is
        Rayleigh distributed; its 99.9 % quantile is
        s * sqrt(-2 ln(0.001)) = 1.118 * 3.717 = 4.16 m, rounded to 4.2.

    Other fields: vote_fraction (share of a drone's pairs that must disagree),
    persist_m / persist_n (m alarms within the last n steps), cusum_k /
    cusum_h (CUSUM slack and alarm level), max_coast_s (longest time the
    filter may run without accepting GPS), c_aggregate (how Detector C
    combines anchor errors: median | mean | max).
    """

    tau_a: float = 4.3
    vote_fraction: float = 0.5
    persist_m: int = 3
    persist_n: int = 5
    nis_threshold: float = 9.21
    cusum_k: float = 1.0
    cusum_h: float = 25.0
    gate_nis: float = 9.21
    max_coast_s: float = 10.0
    tau_c: float = 4.2
    c_aggregate: str = "median"

    def __post_init__(self) -> None:
        for name in ("tau_a", "nis_threshold", "cusum_h", "gate_nis", "max_coast_s", "tau_c"):
            value = _as_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
            setattr(self, name, value)
        self.cusum_k = _as_float("cusum_k", self.cusum_k)
        if self.cusum_k < 0:
            raise ValueError(f"cusum_k must be >= 0, got {self.cusum_k}")
        self.vote_fraction = _as_float("vote_fraction", self.vote_fraction)
        if not (0.0 < self.vote_fraction <= 1.0):
            raise ValueError(f"vote_fraction must be in (0, 1], got {self.vote_fraction}")
        self.persist_m = _as_int("persist_m", self.persist_m)
        self.persist_n = _as_int("persist_n", self.persist_n)
        if not (1 <= self.persist_m <= self.persist_n):
            raise ValueError(
                f"need 1 <= persist_m <= persist_n, got m={self.persist_m}, n={self.persist_n}"
            )
        self.c_aggregate = _check_choice("c_aggregate", self.c_aggregate, VALID_AGGREGATES)


# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------
@dataclass
class ExperimentConfig(_ConfigMixin):
    """Settings for statistical experiments over many random seeds.

    Seeds used: seed_start, seed_start + 1, ..., seed_start + n_seeds - 1.
    WHY seed_start = 5000: experiment seeds are kept far away from the
    seeds used for calibration and demos, so a threshold is never tested on
    the same random flights it was tuned on.
    """

    n_seeds: int = 100
    seed_start: int = 5000
    workers: int = 1

    def __post_init__(self) -> None:
        self.n_seeds = _as_int("n_seeds", self.n_seeds)
        self.seed_start = _as_int("seed_start", self.seed_start)
        self.workers = _as_int("workers", self.workers)
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        if self.seed_start < 0:
            raise ValueError(f"seed_start must be >= 0, got {self.seed_start}")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")

    def seeds(self) -> list[int]:
        """Return the list of seeds this experiment runs."""
        return list(range(self.seed_start, self.seed_start + self.n_seeds))
