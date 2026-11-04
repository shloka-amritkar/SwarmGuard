############################################################################
# PHASE 2 — DETECTORS A, B, C + METRICS + CALIBRATION
# Files: detection/base.py, detection/detector_a_geometry.py,
#        detection/detector_b_motion.py, detection/detector_c_anchor.py,
#        analysis/metrics.py, analysis/calibration.py, cli.py additions
############################################################################

=== 2.1 detection/base.py ===
@dataclass StepResult
  step, t; flags (N,) bool (after persistence); raw_flags (N,) bool;
  scores (N,) float; status list[DroneStatus]; swarm_flag bool;
  swarm_score float; diagnostics dict
abstract class Detector: name, short ("A"|"B"|"C"|"D"); reset(n);
  update(obs: ObservedFrame) -> StepResult; run(list[ObservedFrame]) ->
  DetectionRun. update() raises TypeError if given a Frame
  ("detectors must not see truth").
DetectionRun: to_dataframe() long format (step, t, drone, flag, raw_flag,
  score, status, swarm_flag, swarm_score); first_alarm_time(drone);
  alarm_drones().
PersistenceFilter(m, n): per-drone deque of last n raw flags; flag if sum >= m.
  WHY: 600 steps × 10 drones = 6,000 tests per run; even 1% per test would
  give ~60 false alarms per clean run. Requiring m of n removes isolated
  noise spikes while real attacks, which persist, still pass.

=== 2.2 detection/detector_a_geometry.py ===
Per step:
  G = pairwise_distances(gps_pos)
  R = |G - ranges| on valid pairs, NaN elsewhere
  suspicious = R > tau_a
  score_i = bad_i / valid_i ; fewer than 2 valid pairs -> NO_EVIDENCE
  raw_flag_i = score_i > vote_fraction (strictly more than half)
  persistence -> flag; swarm_score = 95th percentile of valid residuals;
  swarm_flag = any(flag); diagnostics: residual matrix, bad counts.
WHY majority vote: a spoofed drone disagrees with every partner (score ≈ 1);
each honest drone has only one bad pair out of nine (≈ 0.11). With two
attackers, honest drones have 2/9. Voting breaks down near half the swarm
attacked; the experiments show this.
WHY 95th percentile: the mean dilutes a few large residuals among many small
ones.
Docstring includes the proof that a common offset cancels in G, so A cannot
detect swarm_wide or rigid_rotation. Expected behaviour, not a bug.

=== 2.3 detection/detector_b_motion.py ===
Per-drone 2D Kalman filter, velocity as a control input (F7).
  init: x = first gps, P = gps_sigma^2 I
  predict: x_pred = x + u*dt (u = vel_meas_i);
           P_pred = P + Q, Q = (vel_sigma*dt)^2 I + 1e-4 I
  innovation: y = z - x_pred; S = P_pred + R (R = gps_sigma^2 I);
              NIS = y^T S^-1 y
  update unless gated: K = P_pred S^-1; x = x_pred + K y; P = (I - K) P_pred
  gating: if NIS > gate_nis or the drone is in ALARM, coast (x = x_pred,
          P = P_pred).
  WHY gating: an ungated filter follows a slow drift almost entirely and
  the attack gets absorbed. Refusing suspicious readings keeps the
  innovation visible.
  coast limit: coasting longer than max_coast_s -> reinitialise from GPS,
  keep the ALARM latched, diagnostics["reinit"] = True.
  WHY: velocity integration drifts; latching stops the attacker being
  "forgiven" after a reset.
CUSUM per drone: S_t = max(0, S_{t-1} + NIS_t - (2 + k)), k = 1;
  alarm when S_t > cusum_h; change point = last step with S = 0 before alarm.
  WHY: E[NIS] = 2 when clean, so NIS - 3 averages negative and S sits near
  0. A small persistent anomaly adds a little each step until it crosses h.
  Docstring example: NIS averaging 4.5 adds ~1.5/step, crossing h = 25 in
  ~17 steps ≈ 1.7 s.
raw_flag = NIS > nis_threshold (persistence-filtered) OR S > cusum_h (passed
directly; already persistent). score = S. Diagnostics: nis, cusum,
innovation vectors, coasting mask, change points.
Note: B checks each drone against its own physics, so it sees swarm-wide
jumps at onset and swarm-wide drift via CUSUM.

=== 2.4 detection/detector_c_anchor.py ===
Per step: for drones with anchor_mask: r_i = ||gps_i - anchor_i||;
raw_flag_i = r_i > tau_c; others NO_EVIDENCE with score NaN.
swarm_score = median(r_i) (or mean); swarm_flag = swarm_score > tau_c.
Zero readings -> swarm status NO_EVIDENCE, swarm_score NaN (F8).
Persistence on per-drone and swarm flags.
WHY median: residuals [9.8, 10.1, 10.2, 100] (one faulty anchor) have mean
32.5 and median 10.15. Include this worked example. Limitation: if most
anchors are faulty the median fails too; Phase 3 RANSAC addresses this.

=== 2.5 analysis/metrics.py ===
Uses SimResult (truth allowed here) + DetectionRun.
RunMetrics: attacked; onset; detected (ALARM on an attacked drone at
t >= onset); detection_delay; false_alarm_drones (ALARM on never-attacked
drones); early_alarms (ALARM before onset); clean_false_alarm (any ALARM in
a no-attack run); id_precision and id_recall (alarmed-after-onset set vs
attacked set); time_to_full_identification; swarm_level_detected.
aggregate(list) -> DataFrame with rates, Wilson CIs, mean/median delay,
mean id precision/recall.
WHY separate detection from identification: saying "something is wrong"
and naming the right drone are different capabilities.

=== 2.6 analysis/calibration.py ===
calibrate(base_sim, n_seeds=200, include_manoeuvres=True, target_fa=0.01,
          progress=None) -> CalibrationResult
  1. Run clean seeds (half with manoeuvre_only events).
  2. A: grid tau_a 2.5..8.0 step 0.1 through the full pipeline (vote +
     persistence); pick the smallest with per-run FA rate <= target.
  3. B: run with h = inf, record max CUSUM per run; cusum_h = the
     (1 - target) quantile × 1.05; verify the full detector's FA rate.
  4. C: grid tau_c 2.0..8.0.
  5. Report achieved FA rates with Wilson CIs, thresholds, config hash,
     seed range, UTC timestamp.
Save/load JSON at data/calibration/<config_hash>.json; load_or_default()
returns DetectorConfig.
WHY calibrate: persistence, voting, gating and closed-loop control break
the independence assumptions behind the theoretical thresholds.
Calibration measures the real pipeline's false alarm rate, which is what
the "clean false alarms below 1%" claim requires.
Seeds: calibration 1000–1199; evaluation 5000+. Never overlap (tuning and
testing on the same data inflates results).
progress: optional callback(fraction) for the UI.

=== 2.7 cli.py additions ===
  detect --scenario KEY [--seed N]: per detector, detected / delay / false
    alarms / identified drones.
  calibrate [--seeds 200]: runs, prints the table, saves JSON.

ACCEPTANCE CHECKS
  single_hijack: A and B detect, A identifies drone index 3; C detects if
    that drone is anchored.
  rigid_swarm_shift: A does NOT alarm; B and C do.
  slow_drift: B detects via CUSUM before the offset reaches ~10 m.
  manoeuvre_only: no alarms with calibrated thresholds on most seeds.
############################################################################