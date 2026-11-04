# Phase 2 Explained: Detectors A, B, C, Metrics and Calibration

Phase 1 built the world: simulated flights, with or without simulated GPS
spoofing. Phase 2 adds the three detectors that try to notice the spoofing
using only what a real drone could measure, the metrics that score them
against the hidden truth, and calibration, which measures how often they raise
false alarms on clean flights and picks thresholds from that.

This is the first phase that shows the project's central idea:

| Attack                     | A (geometry) | B (motion) | C (anchors)        |
|----------------------------|--------------|------------|--------------------|
| One drone hijacked         | detects      | detects    | only if anchored   |
| Whole swarm shifted 10 m   | **blind**    | detects    | detects            |
| One drone drifting slowly  | detects late | detects (CUSUM) | only if anchored |

How the pieces fit:

```
SimResult ──► result.observed() ──► detection/  A, B, C ──► DetectionRun
   │            (ObservedFrame only)                          │
   └── truth ─────────────────────────► analysis/metrics.py ◄─┘  (evaluation)
analysis/calibration.py: many clean SimResults ──► thresholds ──► data/calibration/<key>.json
cli.py: `detect` and `calibrate` commands
```

---

## 1. `detection/base.py`: shared building blocks

**StepResult.** One detector's answer at one time step: per-drone `flags`
(ALARM after persistence), `raw_flags` (the single-step test), `scores`,
`status`, plus a swarm-level flag/score and a `diagnostics` dictionary of
extra numbers for the UI.

**Status rules.** These are checked in order:
1. The detector had no usable data for this drone → `NO_EVIDENCE`. It is never
   `NORMAL`, because "I could not check" is not the same as "I checked and it
   looked fine" (rule R4).
2. Persistent flag → `ALARM`.
3. Single-step flag only → `SUSPICIOUS`.
4. Otherwise → `NORMAL`.

**PersistenceFilter (m of n).** A drone is flagged only if at least m = 3 of
its last n = 5 raw flags were True. Why: a run has 600 steps × 10 drones =
6,000 tests. Even a 1 % chance of a noise spike per test would give about 60
false alarms in every clean flight. Random spikes rarely happen 3 times in 5
steps, but a real attack keeps going, so it still gets through. Each drone's
history is kept in a `deque(maxlen=5)`, which drops the oldest value
automatically.

`persistence_over_time` gives the same answer for a whole run at once,
using a rolling sum: `window_sum[k] = cumsum[k] − cumsum[k−n]`. Calibration
uses it to try 56 thresholds without re-running anything. A test checks that
it matches the step-by-step filter exactly.

**Detector (abstract class).** The public `update(obs)` method first checks
that `obs` is an `ObservedFrame`. If someone passes a `Frame`, which contains
the truth, it raises `TypeError("detectors must not see truth")`. The check
lives in the base class, so no detector can forget it. Subclasses only write
`_reset_state` and `_step`.

**DetectionRun.** The list of StepResults for a whole flight. It provides
`flags()` as a (T, N) array, `first_alarm_time(drone)`, `alarm_drones()` and
`to_dataframe()` (one row per step and drone).

---

## 2. `detection/detector_a_geometry.py`: relative geometry

**Idea.** Two drones can measure the distance between them in two
independent ways: from their GPS positions, and from their radio ranging
(UWB). A GPS spoofer cannot change the radio. For each pair:

    residual R_ij = | |gps_i − gps_j| − range_ij |

If one drone's GPS is faked, every pair involving that drone has a large
residual.

**Majority vote.** A pair is suspicious if `R_ij > tau_a`. For each drone:

    score_i = suspicious pairs of i / valid pairs of i
    raw flag if score_i > 0.5   (strictly more than half)

A spoofed drone disagrees with all 9 partners (score ≈ 1). An honest drone
disagrees only with the liar (1/9 ≈ 0.11). So the vote points at the right
drone. A drone with fewer than 2 valid pairs gets `NO_EVIDENCE`.

**Swarm score** = 95th percentile of all residuals. The mean would hide the
9 bad pairs among the 36 good ones; the 95th percentile still shows them.

**The blind spot (must know for the viva).** If every GPS is shifted by the
same vector D:

    (g_i + D) − (g_j + D) = g_i − g_j

so every GPS distance stays the same and every residual stays the same.
Rotating every GPS by the same rotation also keeps all distances, because
rotations preserve length. So Detector A cannot see a swarm-wide shift or
rotation, however large. This is the RigidShift blind spot (Park & Yoo). It is
a mathematical fact, not a bug. A test (`test_common_shift_and_rotation_cancel_in_geometry`)
checks it numerically.

---

## 3. `detection/detector_b_motion.py`: motion consistency

**Idea.** Each drone has a velocity sensor (IMU / optical flow) that the
spoofer does not control. If we know where the drone was and how fast it
flew, we can predict where it should be. GPS that keeps disagreeing with
the prediction is suspicious. Each drone is checked against its *own*
physics, so B also sees swarm-wide attacks.

**Kalman filter (one per drone, in metres; fix F7).**
- Predict: `x_pred = x + u·dt`, where u is the velocity measured on the
  *previous* step. Design rule D4 says `vel_meas[k]` is the velocity flown
  during `[t_k, t_k + dt]`, so it carries the drone from step k to k + 1.
- Uncertainty grows: `P_pred = P + Q`, with `Q = (vel_sigma·dt)² I + 0.0001 I`.
- Innovation (surprise): `y = GPS − x_pred`, with expected spread
  `S = P_pred + R` and `R = gps_sigma² I`.
- NIS = `yᵀ S⁻¹ y`. This is the surprise measured in standard deviations,
  squared. On a clean flight it follows a chi-square distribution with 2 degrees
  of freedom, whose average is 2.
- Update: `K = P_pred S⁻¹`, `x = x_pred + K y`, `P = (I − K) P_pred`.

An earlier project worked in degrees and never used the velocity reading,
so its "prediction" was just the last GPS fix (F7). The test
`test_b_uses_velocity_input_f7` shows the difference. Drones flying at 2 m/s
with a correct velocity input give NIS ≈ 0. If the velocity input is ignored,
every drone alarms.

**Gating.** If NIS > `gate_nis` or the drone is already in ALARM, the GPS
reading is not used ("coasting"). Without gating, the filter slowly follows a
drift and the attack disappears into the estimate.

**Coast limit and latch.** Predicting from velocity alone slowly drifts, so
after 10 s of coasting the filter restarts from GPS. The ALARM stays *latched*
for the rest of the run, so the attacker is not "forgiven" by the restart.

**CUSUM (for slow drifts).** `S_t = max(0, S_{t−1} + NIS_t − 3)`. Clean NIS
averages 2, so each step subtracts about 1 and S stays near 0. A small but
steady problem (NIS ≈ 4.5) adds about 1.5 per step and crosses h = 25 in about
17 steps (1.7 s). The *change point* is the last step where S was 0. It
estimates when the attack started.

**Flags.** `flag = persistent(NIS > nis_threshold) OR S > cusum_h OR latched`.
CUSUM skips the persistence filter because it already accumulates over many
steps. Score = S.

**Sensor sigmas.** B needs `gps_sigma` and `vel_sigma` to build R and Q. These
are datasheet values a real drone would know. They are not truth.
`build_detectors` passes them in from `SimConfig`.

---

## 4. `detection/detector_c_anchor.py`: absolute anchors

**Idea.** Some drones have a trusted absolute position fix (for example a
camera recognising surveyed landmarks). For each drone with a reading,
`r_i = |gps_i − anchor_i|`, and the raw flag is `r_i > tau_c`. Drones without a
reading are `NO_EVIDENCE` with score NaN.

**Swarm level.** The median of the available r_i (configurable: median, mean
or max), flagged if above tau_c. With zero readings the swarm score is NaN and
the swarm status is "no_evidence". It is never 0 (fix F8: an earlier project
returned 0, which looks like "verified clean").

**Why the median (worked example).** Residuals 9.8, 10.1, 10.2 and 100 m, where
the last anchor is faulty. Mean = 32.5 m (pulled up by one bad anchor). Median
= 10.15 m (the real offset). If *most* anchors are faulty the median fails
too; Phase 3's RANSAC deals with that.

---

## 5. `analysis/metrics.py`: scoring against the truth

This module is allowed to read truth because it *evaluates* detectors. It
never feeds anything back into them.

`compute_metrics(sim_result, detection_run)` → `RunMetrics`:
- **detected**: ALARM on an attacked drone at or after onset.
- **detection_delay**: first such ALARM time − onset.
- **false_alarm_drones**: drones that were never attacked but alarmed.
- **early_alarms**: any ALARM before onset.
- **clean_false_alarm**: in a no-attack run, any ALARM or swarm flag.
- **id_precision** = correctly named drones / all drones named after onset.
  It is NaN when nothing was named, because 0/0 is undefined, not 0.
- **id_recall** = correctly named drones / all attacked drones.
- **time_to_full_identification**: when every attacked drone has been named.
- **swarm_level_detected**: swarm flag at or after onset.

Detection and identification are kept separate on purpose. Noticing that
something is wrong and naming the right drone are different abilities.

`aggregate(list)` makes one row per detector, with rates and 95 % **Wilson**
confidence intervals. The simple formula `p ± 1.96·sqrt(p(1−p)/n)` gives zero
width at 0 % or 100 %, which would claim certainty. Wilson does not.

---

## 6. `analysis/calibration.py`: measuring the real false-alarm rate

**Why.** The default thresholds in `config.py` come from textbook formulas
that assume independent tests. Our pipeline breaks that assumption in
several ways: persistence (3 of 5), voting, gating and closed-loop control.
So we *measure*.

**Steps (200 clean runs, seeds 1000–1199; odd seeds also fly the legitimate
rotate + scale manoeuvres):**
1. **A.** Try tau_a = 4.3, 4.4, … 8.0 through the full vote + persistence
   pipeline. Take the smallest value whose false-alarm rate per run is
   ≤ 1 %. Smallest means most sensitive while still acceptable. The search
   **starts at 4.3 m**, the theoretical 3-sigma value, so calibration can
   raise tau_a but never lower it (see "Why A has a floor" below).
2. **B.** Run with CUSUM alarms switched off and record each run's highest
   CUSUM value. `cusum_h` = 99th percentile × 1.05 (a 5 % safety margin). Then
   run the complete detector to check. *Our extension:* if the NIS path alone
   still alarms too often, `nis_threshold` is raised in steps of 0.5 and every
   step is written into `notes`.
3. **C.** Try tau_c = 2.0 … 8.0 in the same way as A.
4. Report the achieved rates with Wilson intervals and save JSON to
   `data/calibration/<key>.json`.

**Our actual result (simulation result, 200 runs):**

| Detector | Threshold                               | False-alarm runs | Rate (95 % CI)     |
|----------|-----------------------------------------|------------------|--------------------|
| A        | tau_a = 4.3 m                           | 0 / 200          | 0.0 % (0.0–1.9 %)  |
| B        | cusum_h = 37.0, nis_threshold = 10.71   | 1 / 200          | 0.5 % (0.1–2.8 %)  |
| C        | tau_c = 3.4 m                           | 2 / 200          | 1.0 % (0.3–3.6 %)  |

At the textbook value 9.21, `nis_threshold` gave 10/200 false-alarm runs. That
is why calibration matters: the theory said 1 %, but the real pipeline gave
5 %.

**Why A has a floor at 4.3 m.** Clean-run false alarms alone do not
constrain Detector A. Majority voting hides noise: an honest drone is flagged
only if *more than half* of its pairs look bad, and random noise almost never
does that. In an earlier version the search went down to 2.5 m and still met
the 1 % target, so calibration picked 2.5 m. During a multi-drone attack,
however, each honest drone already has one bad pair per liar (2 of 9 with two
liars). With such a low threshold, ordinary noise on a few more pairs tips it
over half, and A wrongly blames an honest drone. At 2.5 m this happened in
8 of 20 `coordinated_pair` runs; with the 4.3 m floor it happened in 0 of 20.
The lesson: a threshold must be judged on what it does under attack, not only
on clean flights.

**Light-show swarm (20 drones, loop path, shape changes).** This swarm is
calibrated separately (`calibrate --scenario light_show_pune`; the attack is
ignored because calibration always flies clean):

| Detector | Threshold                               | False-alarm runs | Rate (95 % CI)     |
|----------|-----------------------------------------|------------------|--------------------|
| A        | tau_a = 4.3 m                           | 0 / 200          | 0.0 % (0.0–1.9 %)  |
| B        | cusum_h = 44.3, nis_threshold = 11.21   | 2 / 200          | 1.0 % (0.3–3.6 %)  |
| C        | tau_c = 3.5 m                           | 0 / 200          | 0.0 % (0.0–1.9 %)  |

With 20 drones there are twice as many tests per flight, so B needed slightly
higher thresholds than the 10-drone swarm (the textbook 9.21 gave 14/200).

**The key.** The file name is a SHA-256 of the SimConfig with `seed` and
`manoeuvres` removed. So `clean_baseline` and `manoeuvre_only` share one
calibration. A different swarm (for example the 20-drone light show) needs
its own calibration and uses the defaults until it has one.

**Seed separation.** Calibration uses seeds 1000+ and evaluation uses 5000+.
`calibration_seeds()` refuses any request that would overlap. Testing on the
same flights used for tuning would inflate the results.

**F3 (no silent overwriting).** `DetectorConfig()` always holds the analytical
defaults. Calibrated values are used only when the caller explicitly calls
`load_calibration` / `load_or_default`, and the CLI prints which source it
used.

---

## 7. `cli.py`: new commands

```
.venv\Scripts\python cli.py detect --scenario rigid_swarm_shift --seed 5000
.venv\Scripts\python cli.py calibrate --seeds 200
.venv\Scripts\python cli.py calibrate --scenario light_show_pune --seeds 200
```

`detect` prints, for each detector: detected or not, the delay, the drones it
named, false-alarm drones, precision/recall and swarm-level alarm. It also
says whether the thresholds are calibrated or the analytical defaults.
`calibrate` prints the table above and saves the JSON. It uses only the
scenario's swarm settings; if the scenario has an attack, the attack is
ignored (with a printed note), because calibration always flies clean.

---

## 8. Tests (`tests/test_phase2.py`, 25 checks) and acceptance results

The tests cover: the truth guard (TypeError), persistence (both versions
agree), the shift/rotation cancellation proof, the majority vote, A finding
Drone 04, A blind but B and C detecting on the rigid shift, NO_EVIDENCE with
too few pairs or zero anchors (F8), the velocity input (F7), the CUSUM formula,
the latch after the coast limit, the median example, C detecting only when the
hijacked drone is anchored, metrics on hand-built runs, Wilson columns,
calibration seed ranges, the calibration key, and a small calibration
round-trip with defaults left untouched (F3).

Acceptance checks with calibrated thresholds, evaluation seeds 5000–5019
(simulation result):
- **single_hijack:** A and B detected 20/20, and A always named Drone 04. C
  detected exactly on the seeds where Drone 04 carries an anchor.
- **rigid_swarm_shift:** B and C detected 20/20. A stayed silent on 20/20
  (the blind spot).
- **slow_drift:** B's CUSUM alarmed when the fake offset was only 1.6–3.3 m,
  well before 10 m.
- **manoeuvre_only:** A and C silent on 20/20; B raised one false alarm
  (no alarm from any detector on 19/20 seeds).
- **coordinated_pair:** A detected 20/20 and flagged an honest drone in 0/20
  runs (8/20 before the 4.3 m floor).

---

## VIVA NOTES

**1. Why can't Detector A see a swarm-wide shift?**
A compares distances between drones. If every GPS moves by the same D, then
(g_i + D) − (g_j + D) = g_i − g_j, so no distance changes. A rigid rotation
also keeps all lengths. A cannot see it at any size. It is the RigidShift blind
spot, and the reason we need B and C.

**2. Why majority voting instead of flagging any drone with a bad pair?**
Every bad pair has two ends, and one of them is honest. The liar disagrees
with everyone (score ≈ 1), while an honest drone disagrees only with the liar
(≈ 0.11). Voting puts the blame on the right drone. It fails as the number of
liars approaches half the swarm.

**3. What is NIS and why is its average 2?**
NIS = yᵀS⁻¹y is the innovation (GPS minus prediction) measured in units of
its expected spread, squared. For a correct 2D Gaussian innovation it follows
a chi-square distribution with 2 degrees of freedom, whose mean is 2. Values
much larger than 2 mean the GPS does not match the drone's own motion.

**4. Why does B need CUSUM as well as an NIS threshold?**
A slow drift adds only a little error each step, so a single-step NIS test
never fires. CUSUM adds up (NIS − 3). On clean data that is negative on
average and S stays near 0. A small steady excess builds up until S crosses h.
In our runs it caught a 0.5 m/s drift when the offset was about 2–3 m.

**5. What is gating and why is it needed?**
If a GPS reading is too surprising (or the drone is already in alarm), the
filter ignores it and coasts on its prediction. Without this, the filter
gradually follows a drift, the innovation shrinks, and the attack hides
inside the estimate.

**6. Why latch the alarm after the coast limit?**
Coasting on velocity alone drifts, so after 10 s we restart the filter from
GPS. If that also cleared the alarm, an attacker would simply be "forgiven"
every 10 s. Latching keeps the record honest.

**7. What does fix F7 mean?**
The Kalman filter works in metres and actually uses the velocity sensor as
its control input: x_pred = x + u·dt. An earlier project worked in degrees and
ignored velocity, so it compared GPS with the previous GPS. A test shows that
ignoring velocity makes every moving drone alarm.

**8. Why does Detector C return NO_EVIDENCE rather than 0 with no anchors (F8)?**
A score of 0 reads as "checked and clean", but with no anchors nothing was
checked. NO_EVIDENCE with a NaN score is the honest answer, and later fusion
treats it as "unknown", not as support for "normal".

**9. Why the median in Detector C?**
One faulty anchor can wreck the mean: 9.8, 10.1, 10.2 and 100 average to
32.5, but the median is 10.15. The median ignores a minority of bad anchors.
If most anchors are faulty it fails too, which Phase 3's RANSAC addresses.

**10. Why calibrate instead of using the textbook thresholds?**
The formulas assume independent tests. Persistence, voting, gating and the
closed loop break that. For example, the textbook NIS threshold 9.21 gave 5 %
false-alarm runs, not 1 %. Calibration measures the real pipeline on 200
clean flights, so "below 1 %" is a measured claim, reported with a Wilson
interval.

**11. Why are calibration and evaluation seeds different?**
Thresholds tuned on some flights naturally look good on those same flights.
Calibrating on seeds 1000+ and evaluating on 5000+ means the reported rates
come from flights the thresholds have never seen, like a train/test split.

**12. Why separate detection from identification?**
"Something is wrong" and "Drone 04 is wrong" are different abilities. A
detector can have perfect detection and poor identification. Detector C on a
swarm shift shows this: it detects every time but names only the anchored 30 %
of drones (recall 0.30).

**13. Why Wilson confidence intervals?**
With 20 runs, 20/20 is not proof of 100 %. The simple formula gives a zero-width
interval at 0 % or 100 %. Wilson gives an honest range, for example 0.3–3.6 % for
2 false alarms in 200 runs.

**14. How do you guarantee detectors never see the truth?**
They receive only `ObservedFrame`, a type with no truth fields at all, and
`Detector.update()` raises TypeError if given a `Frame`. Truth is read only in
`analysis/metrics.py` and the CLI's labelled evaluation lines.

**15. Does a detector alarm prove which drone was attacked, or by whom?**
No. An alarm means the drone's GPS is inconsistent with independent
measurements. It does not identify an attacker and does not prove intent. With
no alarm, we say "no anomaly detected", never "GPS genuine".

**16. Why can't calibration lower tau_a below 4.3 m?**
Clean flights barely constrain A, because majority voting hides pair noise:
even a 2.5 m threshold met the 1 % clean target. Under a two-drone attack,
though, honest drones already have 2 bad pairs out of 9, so a low threshold
lets noise tip them over half and A blames them (8/20 runs at 2.5 m, 0/20 at
4.3 m). So 4.3 m, the theoretical 3-sigma value, is a floor that calibration
may raise but not lower.
