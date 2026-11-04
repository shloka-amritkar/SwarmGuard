############################################################################
# PHASE 3 — FUSION, RECOVERY, EXPERIMENTS, EVASION SEARCH, PROPAGATION
# Files: detection/fusion.py, recovery/mds_alignment.py,
#        recovery/multilateration.py, analysis/experiments.py,
#        analysis/evasion_search.py, analysis/propagation.py, cli.py additions
############################################################################

=== 3.1 detection/fusion.py ===
Mass triple per detector per drone (m_attack, m_normal, m_uncertain), sum 1.
Raw belief b in [0,1]:
  A: clip(score_a, 0, 1)
  B: max(1 - exp(-S / cusum_h), min(NIS / (3 * nis_threshold), 1))
  C: clip((r - tau_c/2) / tau_c, 0, 1)
  m_attack = α b, m_normal = α (1 - b), m_uncertain = 1 - α
  reliability α: A 0.7, B 0.8, C 0.85 (C 0.6 when RANSAC reports outlier
  anchors); NO_EVIDENCE -> (0, 0, 1)
WHY discounting: Dempster's rule assumes independent sources; A and B share
the same GPS readings. Discounting reserves mass for "unknown" and limits
double counting. State this limitation.
Dempster's rule: K = m1a m2n + m1n m2a
  a = (m1a m2a + m1a m2u + m1u m2a)/(1-K)
  n = (m1n m2n + m1n m2u + m1u m2n)/(1-K)
  u = m1u m2u/(1-K)
  K > 0.95 -> (0, 0, 1), flag "high_conflict"
Belief = a; Plausibility = a + u.
Verdict: plausibility < 0.3 NORMAL; belief >= 0.7 COMPROMISED;
u > 0.8 INSUFFICIENT_EVIDENCE; else SUSPICIOUS.
agreement_matrix(): per drone, which detectors alarmed; swarm pattern key,
e.g. "A_silent_B_C_alarm".
WHY: A silent while B and C alarm is the fingerprint of a coordinated
shift. Naming the pattern is more useful than averaging scores.

=== 3.2 recovery/mds_alignment.py ===
classical_mds(D): fill NaNs with Floyd–Warshall shortest paths (record
  filled pairs; disconnected graph -> RecoveryError);
  B = -0.5 J D^2 J, J = I - (1/n) 1 1^T; np.linalg.eigh; top 2
  eigenvalues (negatives clipped to 0); X = V sqrt(Λ).
  WHY: double centering turns squared distances into the Gram matrix of
  centred points; the top eigenvectors rebuild the shape, up to rotation,
  reflection and translation.
kabsch(A, B) -> (R, t): centre both; H = A_c^T B_c; U S V^T = svd(H);
  d = sign(det(V U^T)); R = V diag(1, d) U^T; t = mean(B) - R mean(A).
  WHY d: prevents returning a mirror image. MDS output may be mirrored, so
  fit both X and its reflection and keep the lower residual.
ransac_align(X, anchor_idx, anchor_pos, inlier_thresh=3.0, max_iter=200,
  rng): enumerate triplets if C(k,3) <= max_iter, else sample; fit; count
  inliers; best by (inliers, -residual_sum); refit on inliers. Fewer than 3
  anchors -> INSUFFICIENT_ANCHORS.
recover_positions(obs, robust=True) -> RecoveryResult (positions, method,
  inlier_mask, outlier_anchors, status, residual_to_gps). robust=True
  by default (F4).
  WHY: under a rigid shift, the anchor-aligned formation sits where the
  swarm really is, and the mean of (gps - recovered) estimates the attack
  vector.

=== 3.3 recovery/multilateration.py ===
repair_drone(i, obs, flagged, init=None) -> RepairResult
  neighbours = unflagged drones with a valid range to i (F1)
  fewer than 3 -> status INSUFFICIENT_NEIGHBOURS, position None
  residual e_j(p) = ||p - gps_j|| - range_ij
  scipy.optimize.least_squares(loss="soft_l1", f_scale=1.0),
  x0 = init or the neighbours' mean GPS
  covariance ≈ inv(J^T J) * sum(res^2)/(m - 2)
  WHY soft_l1: one bad range cannot dominate.
  WHY exclude flagged drones: a spoofed neighbour would import the
  attacker's offset into the repair.
repair_all(obs, flagged); recovery_error(frame, repaired) for analysis only.

=== 3.4 analysis/experiments.py ===
run_trial(sim, atk, det, seed) -> dict (pure function, process-pool safe):
  metrics for A, B, C, fusion + recovery error.
sweep(param, values, base_scenario, n_seeds, workers, progress=None) ->
  tidy DataFrame: param, value, detector, seed, detected, delay,
  false_alarm, id_precision, id_recall.
summarise(df) with Wilson CIs.
Built-in sweeps: magnitude 1..20 m (constant_offset, swarm_wide);
  drift_rate 0.05..1.0 (gradual_drift, swarm drift); n_targets
  1..floor(N/2)+1 (where voting collapses); anchor_ratio 0..0.6
  (swarm_wide); n_drones 5..30; gps_sigma 0.5..3.0.
ProcessPoolExecutor when workers > 1. Evaluation seeds from 5000.

=== 3.5 analysis/evasion_search.py ===
min_detectable(param, detector, scenario, target_rate, lo, hi, n_seeds=40,
  tol): bisection on detection rate (assumes monotonic; state it).
stealth_envelope(scenario, detectors) -> thresholds per detector for
  magnitude and drift rate at 50% and 90% detection.
WHY: "Detector B detects 90% of drifts ≥ 0.12 m/s within 10 s" is a
measurable claim; "detects attacks" is not.

=== 3.6 analysis/propagation.py ===
Closed loop only (else ValueError).
sensitivity_matrix(base, magnitude=6.0, seed): baseline clean run; for each
  i, spoof only i (constant offset); M[i,j] = mean_t ||true_j(attack) -
  true_j(baseline)|| for j != i.
pagerank(W, damping=0.85, tol=1e-9): column-normalise (zero columns
  uniform), power iteration. Influence scores on the graph i -> j with
  weight M[i,j]; document the direction used.
min_separation(sim): minimum pairwise true distance over time;
  near-collision events < 1.0 m.
WHY: SwarmFuzz showed one compromised drone can push others into danger
through coordination. Here the consensus centre spreads offset/N to every
drone; the matrix measures the spread.

=== 3.7 cli.py additions ===
  sweep --param magnitude --scenario single_hijack --seeds 50 --workers 4
  evasion --scenario slow_drift
  propagation --scenario clean_baseline

ACCEPTANCE CHECKS
  rigid_swarm_shift fusion pattern = "A_silent_B_C_alarm"
  recovery on rigid_swarm_shift estimates an attack vector within ~2 m of
    the true 10 m east offset (truth used only in the check)
  faulty_anchor_shift: RANSAC marks the faulty anchors as outliers
  a 20-seed magnitude sweep runs and returns a sensible rising curve
############################################################################