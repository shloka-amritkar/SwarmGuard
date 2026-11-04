############################################################################
# PHASE 1 — CORE DATA MODEL, SWARM SIMULATION, SENSORS, ATTACKS, CLI
# Files: config.py, core/schema.py, core/utils.py, simulation/formation.py,
#        simulation/swarm.py, simulation/sensors.py, attacks/gps_attacks.py,
#        attacks/scenarios.py, cli.py, requirements.txt, .gitignore,
#        all needed __init__.py files
############################################################################

=== 1.1 config.py ===
Dataclasses with defaults, validation in __post_init__, and to_dict,
from_dict, to_json, from_json, config_hash (SHA-256 of canonical JSON:
sort_keys=True, separators=(",", ":")).

SimConfig
  n_drones: int = 10                 (valid 4..30)
  formation: str = "circle"          (circle | line | grid | star | heart | custom)
  radius: float = 8.0                (m)
  duration: float = 60.0             (s)
  dt: float = 0.1                    (s) -> 600 steps
  center_speed: float = 2.0          (m/s)
  path: str = "straight"             (straight | loop | waypoints)
  heading_deg: float = 0.0           (0 = east, counter-clockwise positive)
  loop_radius: float = 40.0
  waypoints: list[tuple[float, float]] = empty
  max_speed: float = 6.0             (m/s)
  kp: float = 1.2                    (1/s formation-keeping gain)
  loop_mode: str = "closed"          (open | closed)
  comm_range: float = 50.0           (m)
  gps_sigma: float = 1.0             (m per axis)
  range_sigma: float = 0.3           (m)
  vel_sigma: float = 0.1             (m/s per axis)
  anchor_ratio: float = 0.3
  anchor_sigma: float = 0.5          (m per axis)
  anchor_dropout: float = 0.0        (per-step probability of a missing reading)
  anchor_fault_ratio: float = 0.0
  anchor_fault_bias: float = 15.0    (m)
  manoeuvres: list[ManoeuvreEvent] = empty
  shape_sequence: list[tuple[float, str]] = empty   (time, shape)
  seed: int = 0

ManoeuvreEvent: kind (rotate | scale), start, end, value
  rotate: total degrees rotated over [start, end]
  scale: final scale factor reached at end
  WHY: legitimate formation changes create GPS motion that is not an
  attack. These events let us measure false alarms honestly.

AttackConfig
  type: str = "none"
  targets: list[int] = []            (0-based; UI shows "Drone 01")
  magnitude_m: float = 10.0
  direction_deg: float = 0.0
  start_time: float = 20.0
  end_time: float | None = None
  drift_rate: float = 0.5            (m/s)
  ramp_profile: str = "step"         (step | linear | exponential | polynomial | sigmoid)
  ramp_duration: float = 10.0
  rotation_rate_deg_s: float = 1.0
  pulse_period: float = 4.0
  pulse_duty: float = 0.5
  seed: int = 0

DetectorConfig (changed only by an explicit calibration load, F3)
  tau_a = 4.3, vote_fraction = 0.5, persist_m = 3, persist_n = 5,
  nis_threshold = 9.21, cusum_k = 1.0, cusum_h = 25.0, gate_nis = 9.21,
  max_coast_s = 10.0, tau_c = 4.2, c_aggregate = "median"
  WHY tau_a = 4.3: the pair distance error ≈ projection of (n_i - n_j);
  per-axis GPS sigma 1 m gives variance 2 m^2 for the difference, plus range
  variance 0.09 -> sigma ≈ sqrt(2.09) ≈ 1.45 m; 3 sigma ≈ 4.3 m.
  WHY nis_threshold = 9.21: NIS for a 2D Gaussian innovation is chi-square
  with 2 dof; P(NIS > x) = exp(-x/2) = 0.01 gives x = -2 ln(0.01) = 9.21.
  WHY tau_c ≈ 4.2: GPS minus anchor has per-axis sigma sqrt(1 + 0.25) ≈ 1.118;
  its norm is Rayleigh; the 99.9% quantile is 1.118 * sqrt(-2 ln 0.001)
  ≈ 4.16.
  These are starting values. Phase 2 calibration replaces them
  empirically. Put these derivations in the docstrings.

ExperimentConfig: n_seeds = 100, seed_start = 5000, workers = 1

=== 1.2 core/schema.py ===
Enums: DroneStatus {NORMAL, SUSPICIOUS, ALARM, NO_EVIDENCE};
       Verdict {NORMAL, SUSPICIOUS, COMPROMISED, INSUFFICIENT_EVIDENCE};
       AttackType (all attack names).

@dataclass(frozen=True) ObservedFrame
  step: int; t: float
  gps_pos (N,2) float64; vel_meas (N,2)
  ranges (N,N) symmetric, diagonal 0, NaN where invalid
  range_mask (N,N) bool
  anchor_pos (N,2), NaN rows where no reading
  anchor_mask (N,) bool  (reading available this step)
  property n -> N

@dataclass(frozen=True) Frame
  observed: ObservedFrame
  true_pos (N,2), true_vel (N,2), attack_offset (N,2),
  attack_labels (N,) bool, faulty_anchor (N,) bool   EVALUATION ONLY
  slot_pos (N,2)   intended formation slots
  anchored (N,) bool   which drones carry anchors
  WHY composition, not inheritance: if Frame inherited from ObservedFrame,
  a Frame could be passed to a detector by accident.

@dataclass SimResult
  config, attack, frames: list[Frame]
  observed() -> list[ObservedFrame]
  to_observed_dataframe(): step, t, drone, gps_x, gps_y, vel_x, vel_y,
      anchor_x, anchor_y, has_anchor
  to_truth_dataframe(): step, t, drone, true_x, true_y, true_vx, true_vy,
      off_x, off_y, attacked, slot_x, slot_y
  ranges_array() -> (T,N,N); attack_onset() -> float | None;
  attacked_drones() -> set[int]

=== 1.3 core/utils.py ===
  enu_to_latlon, latlon_to_enu, rotation_matrix(theta),
  pairwise_distances(P) via broadcasting,
  clip_norm(V, max_norm)  # WHY: no physically impossible true motion
  chi2_2dof_quantile(p) = -2 ln(1 - p)
  rayleigh_quantile(sigma, p) = sigma * sqrt(-2 ln(1 - p))
  wilson_interval(k, n, z=1.96):
     centre = (p + z^2/(2n)) / (1 + z^2/n)
     half = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)
     WHY: 50/50 detections should not be reported as "100%, certain".

=== 1.4 simulation/formation.py ===
Offsets (N,2) around the centre, max radius ≈ config.radius:
  circle (angles 2πk/n), line, grid (near-square), star (5-point polygon,
  inner radius 0.4r, equal arc-length sampling), heart (x = 16 sin^3 t,
  y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t, equal arc-length sampling,
  normalised), custom(points), get_shape(name, n, r).
Morphing: assign_slots(current, target) with
  scipy.optimize.linear_sum_assignment to minimise total travel.
  WHY: without it, drones cross paths during light-show shape changes and
  near-collisions appear that no attack caused.
  morph_offsets(t, shape_sequence, n, r): switch at given times with 2 s
  linear blending.
apply_manoeuvres(offsets, t, events): cumulative rotation and scaling with
  linear progress inside each event window.

=== 1.5 simulation/sensors.py ===
SensorSuite(config, rng_bundle) with separate generators for gps, range,
vel and anchor.
  gps(true_pos, offset) = true_pos + offset + N(0, gps_sigma^2 I)
  velocity(true_vel) = true_vel + N(0, vel_sigma^2 I)       (NOT spoofable)
  ranges(true_pos): true distances + N(0, range_sigma^2); upper triangle
    generated and mirrored; diagonal 0; > comm_range -> NaN; clipped at 0;
    returns (ranges, mask). The noisy values are what is stored (F2).
  anchors(true_pos, anchored, faulty): anchored drones get
    true_pos + N(0, anchor_sigma^2 I); faulty ones add anchor_fault_bias in
    a fixed seeded direction; dropout -> NaN + mask False.
  WHY ranging, velocity and anchors are not spoofed: they come from
  independent physical channels (UWB, IMU/optical flow, camera landmarks).
  The detection strategy is cross-checking GPS against channels the
  spoofer does not control. This assumption is stated in Methodology.

=== 1.6 attacks/gps_attacks.py ===
AttackModel(attack_config, n_drones).offsets(t, clean_reported=None)
  -> (offset (N,2), labels (N,) bool). Offsets apply to GPS only.
Direction u = (cos d, sin d). Ramp progress p = clip((t - start)/ramp_duration, 0, 1):
  step: 1 once t >= start; linear: p; exponential: (1 - e^{-5p})/(1 - e^{-5});
  polynomial: p^3; sigmoid: logistic(12(p - 0.5)) rescaled so s(0)=0, s(1)=1.
Active window: start <= t and (end is None or t < end).
Types:
  none
  constant_offset: one target, magnitude * u * s(p)
  gradual_drift: one target, drift_rate * (t - start) * u (capped at
    magnitude_m if magnitude_m > 0 and explicitly requested by a cap flag)
  multi_drone: each target has its own seeded direction and magnitude in
    [0.7, 1.3] * magnitude_m
  swarm_wide: same offset on ALL drones; step/ramp, or drift if drift_rate > 0
    and ramp_profile == "linear"
  subset_rigid: same offset on listed targets only
  rigid_rotation: rotate clean reported positions about their centroid c by
    θ(t) = rotation_rate * (t - start), plus magnitude * u * s(p)
  intermittent: one target, active when ((t - start) mod period) < duty * period
Labels: True for every affected drone for the whole active window (F5).
Docstring proves rotation preserves pairwise distances:
  |R(p_i - c) - R(p_j - c)| = |p_i - p_j|.

=== 1.7 simulation/swarm.py ===
SwarmSimulator(sim_config, attack_config).run() -> SimResult
Each step (t = k*dt):
  1. centre(t): straight / loop / waypoints; v_center by finite difference.
  2. formation offsets (morph + manoeuvres); slots = centre + offsets.
  3. attack offsets (rotation uses clean reported positions this step).
  4. sensors from the current true state.
  5. control:
     open loop: v_cmd = kp * (slot - true_pos) + v_center
       WHY: spoofing corrupts only measurements; cleanest detection tests.
     closed loop: own position estimate = own GPS; swarm centre estimate =
       mean of all reported GPS minus mean formation offset;
       v_cmd = kp * (centre_est + offset_i - gps_i) + v_center
       WHY: real drones steer by what they measure. Because the centre is
       a consensus of everyone's GPS, one spoofed drone shifts every
       target by offset/N: this is how attacks propagate (SwarmFuzz idea).
  6. v = clip_norm(v_cmd, max_speed); true_pos += v * dt; true_vel = v.
  7. store Frame.
Initial true_pos = slots at t = 0 + 0.2 m seeded jitter. Anchored drones:
round(anchor_ratio * N) chosen once per run; faulty subset among them.
Performance: 600 steps × 30 drones under 1 s.

=== 1.8 attacks/scenarios.py ===
Scenario: key, title, description (2–3 sentences: what it tests, which
detector should succeed or fail), sim, attack, expected dict
(detector -> "detect" | "blind" | "partial").
Presets:
  clean_baseline
  single_hijack: drone index 3, constant 10 m east at 20 s
  slow_drift: drone index 6, 0.5 m/s from 15 s
  coordinated_pair: indices 1 and 5, multi_drone 8 m at 20 s
  rigid_swarm_shift: swarm_wide 10 m east step at 25 s
      expected A blind, B detect, C detect
  rigid_swarm_drift: swarm_wide 0.3 m/s from 20 s
  rigid_rotation: 1 deg/s from 20 s
  faulty_anchor_shift: rigid shift, anchor_ratio 0.5, anchor_fault_ratio 0.34
  intermittent_pulses: index 2, 6 m, period 4 s, duty 0.5
  manoeuvre_only: rotate 90° over 20–30 s, scale to 1.5 over 35–40 s, no attack
  light_show_pune: n = 20, radius 15, path loop, shape_sequence
      [(0,"circle"), (12,"star"), (24,"heart"), (36,"grid"), (48,"circle")],
      swarm_wide drift 0.3 m/s from 30 s, closed loop.
      The demo scenario: the whole show is dragged off course while the
      formation still looks perfect to relative-geometry checks.
get_scenario(key), list_scenarios().

=== 1.9 cli.py (Phase 1 part) ===
argparse subcommands:
  list-scenarios
  run --scenario KEY [--seed N] [--plot] [--csv out.csv]
    prints steps, drones, attacked drones, onset, max offset, mean true
    distance from slots; --plot saves a matplotlib PNG to data/plots/:
    true paths solid, GPS paths dotted, attacked drones in red.

=== 1.10 requirements.txt and .gitignore ===
requirements with minimum versions for the stack in CLAUDE.md.
.gitignore: .venv/, data/, keys/, *.db, __pycache__/, .pytest_cache/, *.zip

ACCEPTANCE CHECKS (run them all)
  .venv\Scripts\python cli.py list-scenarios
  .venv\Scripts\python cli.py run --scenario single_hijack --plot
  .venv\Scripts\python cli.py run --scenario rigid_swarm_shift --plot
  .venv\Scripts\python cli.py run --scenario light_show_pune --plot
  A short Python check: in rigid_swarm_shift, the GPS pairwise distance
  matrices at t = 24 s and t = 30 s differ only by noise.
############################################################################