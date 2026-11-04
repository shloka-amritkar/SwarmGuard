# Phase 1 Explained: Data Model, Swarm Simulation, Sensors, Attacks, CLI

Phase 1 builds the "world" that every later phase works in. Nothing here
detects anything yet. It produces realistic, reproducible flights with or
without simulated GPS spoofing, and it makes sure detectors can never see the
answers.

How the pieces fit:

```
config.py ──► simulation/swarm.py ──► SimResult (list of Frames)
                 │   uses                    │
                 ├─ simulation/formation.py  ├─ frame.observed  → detectors (Phase 2)
                 ├─ simulation/sensors.py    └─ frame.true_pos … → evaluation only
                 └─ attacks/gps_attacks.py
attacks/scenarios.py: named presets (config + attack + expected outcome)
cli.py: run presets from the terminal, print summaries, save plots/CSV
```

---

## 1. `config.py`: every setting in one place

**What it does.** It defines five dataclasses: `SimConfig` (the flight and
sensors), `ManoeuvreEvent` (a legitimate rotate/scale), `AttackConfig` (the
simulated spoofing), `DetectorConfig` (thresholds) and `ExperimentConfig`
(seeds for statistics). Each one checks its own values in `__post_init__` and
raises a clear `ValueError` (for example "n_drones must be between 4 and 30,
got 3").

**Hashing.** Any config can be turned into *canonical JSON*, which means sorted
keys and no extra spaces, and then hashed with SHA-256. Equal configs always
produce the same text, so they get the same hash. Later phases put this hash
into evidence packages to prove which settings produced a result. Numbers are
always stored as floats, so `radius=8` and `radius=8.0` hash the same.

**Where the default thresholds come from (plain words):**
- `tau_a = 4.3 m`. Detector A compares the GPS distance between two drones
  with their radio range. Each GPS has about 1 m of error per axis, so the
  difference of two GPS errors has variance 1 + 1 = 2 m². The radio adds
  0.3² = 0.09. The total standard deviation is √2.09 ≈ 1.45 m, and three of
  those is ≈ 4.3 m.
- `nis_threshold = 9.21`. A 2-D "surprise score" (NIS) follows a chi-square
  distribution with 2 degrees of freedom. The chance it exceeds x is e^(−x/2).
  Setting that chance to 1 % gives x = −2 ln 0.01 = 9.21.
- `tau_c = 4.2 m`. GPS minus anchor has a per-axis sigma of √(1 + 0.25) =
  1.118 m. The length of such a 2-D error follows a Rayleigh distribution, and
  only 0.1 % of lengths exceed 1.118·√(−2 ln 0.001) ≈ 4.16 m.

**Additions beyond the spec, and why:**
- `drift_cap` (AttackConfig). The spec mentions "a cap flag" for
  gradual drift. When it is True, a drift stops growing at `magnitude_m`.
- `centre_gain` (SimConfig, default 0.5). This controls closed-loop path
  correction (see swarm.py). Setting it to 0 gives the spec's pure
  formation-keeping law.
- `custom_points` (SimConfig). The coordinates for the "custom" formation.

## 2. `core/utils.py`: small maths helpers

- `enu_to_latlon` / `latlon_to_enu` convert metres east and north of Pune to
  latitude and longitude, and back. A degree of longitude is shorter than a
  degree of latitude by cos(latitude), which is 0.948 at Pune, so the formula
  divides by it. These are used **only for maps**. All detector maths is in
  metres.
- `rotation_matrix(θ)` is the standard 2×2 matrix that turns a point by θ
  counter-clockwise.
- `pairwise_distances(P)` returns every drone-to-drone distance at once.
  It uses NumPy "broadcasting" instead of a double loop.
- `clip_norm(V, max)` shortens velocities that are too fast while keeping
  their direction, so no drone flies faster than its motors allow.
- `chi2_2dof_quantile` and `rayleigh_quantile` are the threshold formulas above.
- `wilson_interval(k, n)` is a confidence range for a success rate. The naive
  formula says 50 detections out of 50 is "100 % ± 0". Wilson gives about
  93–100 %, which honestly shows that 50 tries cannot prove perfection.
- `round_half_up` exists because Python's `round(2.5)` is 2 (banker's rounding),
  which would be surprising when choosing how many drones get anchors.

## 3. `core/schema.py`: the data model (rule R1 lives here)

- **`ObservedFrame`** holds only what real drones could measure at one time
  step: GPS, the velocity sensor, the radio-range matrix with its mask, and
  anchor fixes with their mask. It has **no** truth fields at all.
- **`Frame`** *contains* an ObservedFrame (`frame.observed`) plus the hidden
  truth: true position and velocity, attack offset, attack labels and faulty
  anchors. It also holds plan information: slots and which drones carry anchors.
- **Composition instead of inheritance.** If `Frame` were a subclass of
  `ObservedFrame`, a whole Frame (truth included) could be handed to a detector
  by accident and Python would accept it. With composition the programmer must
  write `frame.observed`, so the separation is visible.
- **Read-only copies.** Each array is copied and locked (`writeable = False`).
  If a detector tried `gps_pos -= 5`, NumPy would raise an error instead of
  silently corrupting the data every other detector uses.
- **`SimResult`** is the list of frames plus the configs. It also has helpers:
  `observed()` (the detector input), `stack("true_pos")` (a T×N×2 array), long
  tables via `to_observed_dataframe` and `to_truth_dataframe`,
  `attack_onset()` and `attacked_drones()`.
- **Enums.** `DroneStatus` includes `NO_EVIDENCE`, which is different from
  `NORMAL`: "I couldn't check" is not "I checked and it was fine" (rule R4).

## 4. `simulation/formation.py`: shapes and shape changes

- A formation is a list of offsets from the swarm centre. The built-in shapes
  are centred on (0, 0) and scaled so the farthest drone is exactly `radius`
  metres out.
- **Star:** a 10-corner outline alternating radius 1 and 0.4.
  **Heart:** x = 16 sin³t, y = 13 cos t − 5 cos 2t − 2 cos 3t − cos 4t.
- **Equal arc-length sampling.** The code measures the curve's length and
  places drones at equal distances along it. Sampling equal steps of t would
  bunch drones up where the curve moves slowly.
- **`assign_slots`** decides which drone takes which new slot when the shape
  changes, minimising total travel. It builds a cost table C[i, j] (the
  distance from drone i to slot j) and solves it with
  `scipy.optimize.linear_sum_assignment`, the Hungarian algorithm. Without
  this, drones would cross paths and near-collisions would appear that no
  attack caused.
- **`ShapeSchedule`** precomputes every shape change once and blends linearly
  over 2 s: `offsets = (1 − α)·old + α·new`, with α going from 0 to 1.
- **`apply_manoeuvres`** adds legitimate rotations and scalings. Progress
  inside each event window is linear. These exist so false alarms can be
  measured honestly.

## 5. `simulation/sensors.py`: what the drones measure

| Sensor | Formula | Spoofable? |
|---|---|---|
| GPS | true + attack offset + N(0, 1²) | **yes** (the only one) |
| Velocity | true velocity + N(0, 0.1²) | no (IMU / optical flow) |
| Radio range | true distance + N(0, 0.3²), clipped ≥ 0, NaN beyond 50 m | no (UWB) |
| Anchor | true + N(0, 0.5²), plus a 15 m fixed-direction bias if faulty, possible dropout | no (camera landmarks) |

- **One random generator per sensor**, spawned with `SeedSequence.spawn`.
  Turning on anchor dropout must not change the GPS noise. With a shared
  generator, every later number would shift.
- **Fix F2.** The noisy range values are what gets stored. An earlier project
  computed the noise and then saved the clean distances.
- Range noise is drawn for the upper triangle and mirrored, so i→j and j→i
  share one measurement.

## 6. `attacks/gps_attacks.py`: simulated spoofing

An attack is just an offset added to reported GPS numbers. There are no radio
signals and no real-world use. Terms: u = (cos d, sin d) is the attack
direction, p is ramp progress from 0 to 1, and s(p) is the ramp shape.

| Type | Offset |
|---|---|
| constant_offset | one drone: M·s(p)·u |
| gradual_drift | one drone: drift_rate·(t − start)·u (optionally capped) |
| multi_drone | each target: its own seeded direction, size 0.7–1.3 × M |
| swarm_wide | all drones: the same D (step/ramp, or drift) |
| subset_rigid | listed drones: the same M·s(p)·u |
| rigid_rotation | rotate everyone's reported positions about their centroid |
| intermittent | one drone, "on" for duty × period of each period |

- **Why swarm-wide is invisible to Detector A:**
  (g_i + D) − (g_j + D) = g_i − g_j. The common offset cancels in every
  difference.
- **Why rotation is invisible too:** R(q_i − c) − R(q_j − c) = R(q_i − q_j),
  and a rotation never changes a vector's length.
- **Fix F5.** Labels stay True for the *whole* attack window, including the
  "off" half of intermittent pulses and the early part of a ramp.

## 7. `simulation/swarm.py`: the simulator loop

For each step (dt = 0.1 s, 600 steps):

1. The planned centre c(t) follows a straight line, a loop or waypoints.
   Its velocity comes from a forward difference.
2. Slots = c(t) + formation offsets.
3. GPS noise is drawn, then the attack offset is computed. Rotation needs the
   clean noisy readings, so the noise comes first.
4. Ranges and anchors are read from the current true positions.
5. Control:
   - **Open loop:** `v = kp·(slot − true) + v_c`. Drones steer by truth, so
     spoofing never moves them. This is the cleanest test of detection.
   - **Closed loop:** `c_est = mean(GPS) − mean(offsets)`, then
     `v = kp·(c_est + o_i − gps_i) + kc·(c(t) − c_est) + v_c`.
     The first term keeps formation using GPS. Because c_est is an average
     of everyone's GPS, one spoofed drone moves everyone's target. The second
     term (kc = `centre_gain`) pulls the GPS-estimated centre back onto the
     planned path, so a swarm-wide spoof physically drags the real swarm off
     course.
6. The commanded speed is capped at `max_speed`. The resulting velocity is
   stored as `true_vel`, and the velocity sensor measures it. The rule is
   `true_pos[k+1] = true_pos[k] + true_vel[k]·dt`, which Phase 2's Detector B
   relies on.
7. The frame is stored and the position is updated.

**Why the pure consensus formula ignores a swarm-wide spoof (the
cancellation).** Look at the closed-loop law with `centre_gain = 0`. It
contains only the formation term:

1. *The formation term never moves the swarm as a whole.* Add it up over all
   drones: Σ kp·(c_est + o_i − gps_i) = kp·(N·c_est + Σo − Σgps). Since
   c_est = mean(gps) − mean(o), this sum is exactly 0. The term can only
   rearrange drones inside the formation. The swarm's average is moved only
   by v_c, which comes from the plan, not from GPS.
2. *A shared offset cancels.* If every GPS reading gets the same D, then
   c_est also grows by D, and each drone's command becomes
   (c_est + D) + o_i − (gps_i + D) = c_est + o_i − gps_i. That is identical
   to the command without the attack. The real swarm keeps flying the plan,
   and only the reported positions are wrong. (This is the same cancellation
   that blinds Detector A.)

The path term `centre_gain·(c(t) − c_est)` breaks the cancellation. c_est
now includes +D, so the whole swarm is steered by −D until its GPS-estimated
centre is back on the plan. In other words, the real swarm is dragged D
off course while the formation still looks perfect. A test checks both
cases: at `centre_gain = 0` the attacked and clean true flights are
identical, and at 0.5 the swarm ends up about 10 m away.

**Measured propagation** (single_hijack: 10 m on Drone 04 with N = 10,
compared with a clean run on the same seed):

| centre_gain | Other 9 drones | Drone 04 |
|---|---|---|
| 0 (pure consensus) | permanently **1.0 m** east (= D/N, the SwarmFuzz propagation effect) | ends about 9 m west |
| 0.5 (default) | nudged at most about **0.35 m**, then pulled back to 0 | dragged the full 10 m west |

The default stays at 0.5 because it shows the real danger of a swarm-wide
spoof. Any scenario can be rerun at 0 with `get_scenario(key,
centre_gain=0.0)` or `python cli.py run --scenario KEY --centre-gain 0`.
Phase 3 propagation analysis runs both values.

**Speed:** 600 steps × 30 drones took about 0.06 s (the target was under 1 s).

## 8. `attacks/scenarios.py`: named presets

Eleven ready-made situations (clean baseline, single hijack, slow drift,
coordinated pair, rigid shift, rigid drift, rigid rotation, faulty anchors,
intermittent pulses, manoeuvres only, Pune light show). Each has a 2–3
sentence description and an `expected` table, for example rigid_swarm_shift:
A = blind, B = detect, C = detect. No-attack scenarios have an empty table:
any alarm there is a false alarm. Every call builds a fresh copy, so editing
one copy never changes the preset.

## 9. `cli.py`: the command line

- `python cli.py list-scenarios` lists the presets.
- `python cli.py run --scenario KEY [--seed N] [--plot] [--csv out.csv]`
  prints a summary. `--plot` saves `data/plots/KEY_seedN.png`, with true
  paths solid, GPS dotted and attacked drones red.

## 10. Tests (`tests/test_phase1.py`, 59 checks)

The tests cover config validation, the JSON/hash round-trip, unit
conversions, Wilson intervals, shape radii, slot assignment, range-matrix
properties (F2), anchor faults and dropout, ramp shapes, intermittent labels
(F5), rotation preserving distances, no truth fields in ObservedFrame,
read-only frames, same seed giving the same output, velocity being consistent
with motion, the rigid shift being invisible to pairwise distances,
propagation, and speed.

**Acceptance result (rigid_swarm_shift, t = 24 s vs 30 s):** the 45
pair-distance changes average −0.09 m (std 1.79 m, largest 3.36 m), even
though every GPS jumped 10 m in between. That is pure noise. Relative geometry
cannot see the attack.

---

## VIVA NOTES

**Q1. Why can't detectors just use true positions?**
Real drones never know their true position; if they did, GPS spoofing would
not matter. A detector that sees the truth would score 100 % and prove nothing.
We enforce this with types: detectors receive an `ObservedFrame`, which has no
truth fields at all.

**Q2. Why does `Frame` contain an `ObservedFrame` instead of inheriting from it?**
With inheritance, a Frame would also count as an ObservedFrame, so the whole
truth could be passed to a detector by mistake and Python would accept it.
With composition, the programmer has to write `frame.observed`, so the
separation is explicit.

**Q3. How is the simulation reproducible?**
All randomness comes from `np.random.default_rng` objects created from a
`SeedSequence(seed)`. Each sensor gets its own child stream through `spawn`.
There is no global random state, so the same config and seed give
bit-identical output. A test checks this.

**Q4. Why does each sensor have its own random generator?**
That way changing one sensor never alters another's noise. For example,
enabling anchor dropout draws extra random numbers. With a shared generator,
every GPS value after that would change, and two runs could no longer be
compared fairly.

**Q5. Prove that a swarm-wide shift is invisible to Detector A.**
Detector A uses distances between GPS positions. If every reading gets the
same offset D, then (g_i + D) − (g_j + D) = g_i − g_j, so every distance is
unchanged. Our acceptance check confirms it: the distance changes across a
10 m shift averaged −0.09 m, which is just noise.

**Q6. Why is a rotation attack also invisible to it?**
After rotating about the centroid c, q_i′ − q_j′ = R(q_i − q_j). A rotation
matrix satisfies RᵀR = I, so it never changes a vector's length. All pairwise
distances stay the same while absolute positions are wrong.

**Q7. Where does tau_a = 4.3 m come from?**
Two independent GPS errors with 1 m sigma per axis differ with variance
2 m², and the radio adds 0.09 m², giving a sigma of about 1.45 m. We alarm
beyond three sigma, about 4.3 m. Phase 2 replaces this with a value
calibrated on clean flights.

**Q8. What are fixes F2 and F5?**
F2: the noisy radio ranges are what is stored. A previous project computed
noise but saved clean distances, which made detection unrealistically easy.
F5: a drone stays labelled "attacked" for the whole attack window, including
the off-phases of a pulsing attack, so ML training and metrics aren't told a
paused attack is normal.

**Q9. What is the difference between open-loop and closed-loop control?**
In open loop the drones steer by their true position, so spoofing only
corrupts the measurements. That is the cleanest test of detection. In closed
loop they steer by GPS, as real drones do, so spoofing changes where they
actually fly and an attack on one drone can affect its neighbours.

**Q10. How does an attack on one drone affect others?**
In closed loop the swarm centre is estimated as the average of everyone's
GPS. A 10 m offset on one of 10 drones moves that average by 1 m, so every
drone's target moves 1 m. With `centre_gain = 0` we measured exactly 1.0 m of
lasting displacement on every other drone. With the default 0.5 the others
peak at about 0.35 m and recover.

**Q10b. Why doesn't the pure consensus formula react to a swarm-wide spoof?**
The common offset D is added to every GPS reading and also to the estimated
centre, so it cancels in each command: (c_est + D) + o_i − (gps_i + D) =
c_est + o_i − gps_i. The drones fly exactly as if there were no attack. The
`centre_gain` path term compares c_est with the planned path, which breaks
the cancellation and drags the real swarm off course.

**Q11. Why use the Hungarian algorithm for shape changes?**
When the circle becomes a star, any drone could take any star point. Solving
a minimum-total-distance assignment (`linear_sum_assignment`) gives each
drone a nearby point. Otherwise drones cross paths, and we would see
near-collisions that no attack caused.

**Q12. Why include legitimate manoeuvres in a security project?**
A detector that alarms whenever the formation moves would be useless. The
manoeuvre_only scenario has rotations and scalings but no attack, so any
alarm there is a measurable false alarm.

**Q13. Why the Wilson interval?**
With few trials, the simple formula claims 50/50 is "100 %, zero
uncertainty". Wilson gives about 93–100 % for 50/50, which honestly reflects
how much 50 tests can prove.

**Q14. Why are frame arrays read-only?**
Every detector receives the same arrays. If one detector changed them in
place, every later detector would get corrupted input, which is a very hard
bug to find. Locking the arrays makes NumPy raise an error immediately
instead.

**Q15. Does this project spoof real GPS?**
No. The "attack" is arithmetic: numbers added to simulated GPS readings
inside a Python program. Nothing generates radio signals or touches a real
receiver or drone.
