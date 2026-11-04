############################################################################
# PHASE 6 — EXPERIMENTAL ML DETECTOR, TESTS, README, DEPLOYMENT
# Files: detection/detector_d_ml.py, tests/*.py, README.md,
#        views updates for Detector D, cli.py additions
############################################################################

=== 6.1 detection/detector_d_ml.py ===
Features per drone per step (rolling window w = 10 steps): nis, cusum,
nis_mean_w, a_vote_fraction, a_max_pair_residual, c_residual (-1 if none),
has_anchor, gps_speed, |gps_velocity - vel_meas|, gps_accel_norm,
heading_change (wrapped to [-π, π]). Built from ObservedFrame + outputs of
A, B, C only, never truth.
build_dataset(n_runs, scenarios, seed_start) -> X, y, groups
  y = attack_labels (persistent window, F5); groups = run id; mix of all
  attack types, clean and manoeuvre_only runs.
train(X, y, groups): GroupShuffleSplit(test_size=0.25) by run (F6);
  RandomForestClassifier(n_estimators=200, max_depth=12,
  min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1);
  metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
  feature importances, per-attack-type recall; save to
  data/models/rf.joblib with its SHA-256 and training config JSON.
MLDetector(Detector): predict_proba > 0.5 + persistence.
WHY "experimental": it learns from our simulator and can only be as
realistic as the simulation. Label it so everywhere. Its main value is
showing which features matter.
WHY group split: consecutive samples from one flight are near-copies;
row splits leak them into both train and test.
UI: add Detector D to the Detection and Experiments views behind an
"Experimental ML detector" toggle, with its metrics and feature importance
chart.

=== 6.2 tests (pytest, fast seeds, total under ~90 s) ===
test_schema: ObservedFrame has no truth fields; update(Frame) raises
  TypeError.
test_reproducibility: same seed identical; different seeds differ.
test_sensors: stored range noise std ≈ range_sigma (F2); symmetric,
  diagonal 0, NaN beyond comm_range.
test_attacks: swarm_wide and rigid_rotation preserve pairwise GPS
  distances (sigma = 0); labels persist; ramps go 0 -> 1.
test_detector_a: flags index 3 in single_hijack; never alarms in
  rigid_swarm_shift; two attackers flagged, no honest drone flagged.
test_detector_b: 10 m step detected within 1.0 s; 0.5 m/s drift detected
  within 15 s; gating keeps innovation persistent.
test_detector_c: rigid shift detected with anchor_ratio 0.3; zero anchors
  -> NO_EVIDENCE (F8); median robust to one faulty anchor of four.
test_calibration: 50 clean seeds with calibrated thresholds: FA rate within
  its Wilson bound of the target.
test_fusion: masses sum to 1; total conflict -> (0, 0, 1); NO_EVIDENCE
  neutral.
test_recovery: noiseless MDS reconstructs geometry (error < 1e-6 after
  Kabsch); Kabsch recovers a known transform and handles reflection; RANSAC
  rejects one faulty anchor among five; multilateration excludes flagged
  drones and returns INSUFFICIENT_NEIGHBOURS with < 3 (F1).
test_forensics: hash changes after one-byte flip; signature fails after
  manifest edit; tamper_test leaves the original intact; custody chain
  links packages.
test_ml: no group id in both train and test (F6).

=== 6.3 README.md ===
Project summary; the central idea with the cancellation equation;
architecture diagram in text; quick start for Windows (venv, pip install,
streamlit run, CLI examples); scenario catalogue with expected outcomes;
how each detector works (short, with equations); calibration and evaluation
methodology (seed ranges, FA target); results tables as templates the team
fills from their own runs (no invented numbers); fixes F1–F8; credits R6;
limitations (2D, simplified kinematics, Gaussian noise, independent channels
assumed unspoofed, simulation only, ML experimental); a module ownership
table for team members (left blank); deployment to Streamlit Community
Cloud (entry point app.py; keys are generated on first run and never
committed; offline map toggle for demos without internet).

=== 6.4 cli.py additions ===
  train-ml --runs 300

ACCEPTANCE CHECKS
  .venv\Scripts\python -m pytest -q passes
  .venv\Scripts\streamlit run app.py works from a fresh clone
  .venv\Scripts\python cli.py investigate --scenario light_show_pune
    --seed 5001, then cli.py verify <package>, both succeed
############################################################################