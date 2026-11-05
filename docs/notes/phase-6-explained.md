# Phase 6 Explained: Experimental ML Detector, Tests, README, Deployment

Phases 1 to 5 built the whole platform: simulator, detectors A, B and C,
fusion, recovery, statistics, signed evidence and the web app. Phase 6
finishes it in four ways:

1. **Detector D**, an *experimental* Random Forest that learns to spot
   attacks from the same measurements the other detectors use
   (`detection/detector_d_ml.py`).
2. **One test file per module**, so every team member has a short, clean
   file to show for their own part (`tests/test_*.py`).
3. **The README**: what the project is, how to run it, how each part works,
   and its honest limits.
4. **Deployment**: the app works from a fresh copy with no data, no keys and
   no trained model, locally or on Streamlit Community Cloud.

```
build_dataset(300 runs)          train()                       MLDetector
 every scenario, seeds 100000+   GroupShuffleSplit BY RUN (F6)  A, B, C run inside it
 ObservedFrame -> A, B, C        RandomForest 200 trees          11 features per drone
 -> 11 features per drone        test metrics + importances      P(attacked) > 0.5
 labels = attack window (F5)     save rf.joblib + SHA-256        + 3-of-5 persistence
```

---

## 1. `detection/detector_d_ml.py`: the experimental ML detector

### What it does
It turns every drone at every time step into a row of 11 numbers (the
*features*). A Random Forest learns which rows come from drones under
attack. At detection time it asks the forest "how likely is this drone to be
under attack right now?"

### The 11 features (`FEATURE_NAMES`)
| Feature | Where it comes from | Plain meaning |
|---|---|---|
| `nis` | Detector B | how surprising this GPS fix is |
| `cusum` | Detector B | accumulated surprise (slow drifts build up) |
| `nis_mean_w` | B, last 10 steps | recent average surprise |
| `a_vote_fraction` | Detector A | share of neighbours whose radio range disagrees |
| `a_max_pair_residual` | Detector A | the worst single GPS-vs-radio disagreement (m) |
| `c_residual` | Detector C | GPS-to-anchor distance, or **-1 if no anchor** |
| `has_anchor` | ObservedFrame | 1 if this drone has an absolute fix |
| `gps_speed` | GPS difference | \|gps_k - gps_{k-1}\| / dt |
| `gps_vel_mismatch` | GPS vs velocity sensor | \|GPS velocity - measured velocity\| |
| `gps_accel_norm` | GPS difference | change of GPS velocity / dt |
| `heading_change` | GPS velocity | turn angle, wrapped to [-pi, pi] |

**Wrapping an angle:** `wrapped = (a + pi) mod 2pi - pi`. A turn from +179
to -179 degrees is really 2 degrees, not 358. Below 0.5 m/s the heading is
set to 0, because the direction of a near-zero vector is pure noise.

**Why -1 for "no anchor":** a residual of 0 would say "the anchor agrees
perfectly", which is false: there is no anchor. -1 can never be a real
distance, so the forest learns it means "no data" (the same honesty as
fix F8).

**`FeatureBuilder` is used for training AND for detection.** The same code
computes the features both times, so they cannot drift apart.

### Building the dataset (`build_dataset`)
- Run `i` flies scenario `i mod 11` with seed `100000 + i`. It covers all 11
  scenarios, including `clean_baseline` and `manoeuvre_only`, which are
  normal flights. Legitimate turns teach the forest that "moving is not
  attacking".
- **Seeds 100000+** never overlap calibration (1000+) or evaluation (5000+)
  seeds, so the model never trains on a flight used to test it.
- **Labels `y`** are the simulation's `attack_labels`. They stay 1 for the
  *whole* attack window, even when an intermittent attack is in its "off"
  half (fix F5). These labels are truth, and they are used **only as the
  answer to learn**, never as an input feature (rule R1).
- `step_stride = 2` keeps every second step. Steps 0.1 s apart are almost
  identical, so this halves the work. The rolling features are still
  updated at every step.

### Training (`train`)
1. **Split by run** with `GroupShuffleSplit(test_size=0.25)`: 75 % of
   *flights* train the model and 25 % of *flights* test it (fix F6).
2. Fit `RandomForestClassifier(n_estimators=200, max_depth=12,
   min_samples_leaf=5, class_weight="balanced", random_state=42)`.
3. Score the test flights:
   - accuracy = (TP + TN) / all
   - precision = TP / (TP + FP): of the alarms, how many were real
   - recall = TP / (TP + FN): of the attacked rows, how many were caught
   - F1 = 2PR / (P + R)
   - ROC-AUC = the chance that a random attacked row scores above a random
     clean row
   - It also reports the confusion matrix, recall per attack type, the
     clean-row false-positive rate and the feature importances.

**Why the split must be by run:** two rows 0.1 s apart from one flight are
near-copies. With a row split, the test set would contain "twins" of
training rows, and the score would measure memory, not generalisation.

**Why `class_weight="balanced"`:** most rows are clean. Without weighting,
a forest that always says "clean" would already score about 70 % accuracy.
Balanced weighting makes each class count equally.

### Saving and loading (`save_model`, `load_model`)
`data/models/` holds three files:
- `rf.joblib`: the forest
- `rf.sha256`: its SHA-256
- `rf_config.json`: training settings, metrics, feature names, scikit-learn
  version and the hash

`load_model` **re-hashes the file and refuses it if it changed**. A joblib
file is a Python pickle, and loading a pickle can run code, so we check its
integrity first. This is the same idea as our signed evidence packages.

### The detector (`MLDetector`)
```
P_i      = average over the 200 trees of p_t(attacked | features of drone i)
raw flag = P_i > 0.5
ALARM    = at least 3 raw flags in the last 5 steps   (same persistence as A, B, C)
```
- It accepts only `ObservedFrame`s (rule R1). Passing a `Frame` raises
  `TypeError`, enforced by the shared `Detector.update`.
- It runs its own private copies of A, B and C to build its features.
- `run()` is overridden for speed. It builds all features first, then asks
  the forest once for all T x N rows, instead of 600 small questions. A test
  checks that this gives exactly the same scores as stepping frame by frame.

### Why it is "experimental", and kept out of fusion
- It learns **only our simulator's** attacks and noise. A real drone's
  sensors would differ.
- Its inputs **are** A, B and C. It is not an independent witness, and
  adding it to Dempster-Shafer fusion would count the same evidence twice.
- It was trained on 300 runs, and these are the held-out test-flight
  results (simulation result):
  - F1 0.88, ROC-AUC 0.95
  - recall 100 % on constant offsets, but only 56 % on rigid rotation
  - on `single_hijack` it caught Drone 04 in 0.2 s, but also briefly
    alarmed on 4 honest drones (precision 0.20)

  Our checks showed those false alarms are noise spikes: honest drones
  average P = 0.15, and 600 steps give the 3-of-5 rule enough chances.
  This is exactly why it is labelled experimental.
- **Its real value is the feature-importance chart.** CUSUM (0.66) and the
  10-step mean NIS (0.18) dominate, then C's residual. The GPS-motion
  features score near zero. With 1 m GPS noise and dt = 0.1 s, a GPS-derived
  speed carries about 14 m/s of noise per axis, which buries the signal.
  Detector B's Kalman filter exists precisely to handle that noise.

---

## 2. `cli.py`: new commands

```
python cli.py train-ml --runs 300          simulate, train, save, print metrics and SHA-256
python cli.py detect --scenario X --ml     also score Detector D on that run
```
scikit-learn is imported **inside** these commands only, because it takes
about a second to import and the other commands do not need it.

---

## 3. App changes (Detection and Experiments views)

- **Toggle "Experimental ML detector"**, off by default. With it off,
  nothing changes.
- **Detection view:** with the toggle on, a "Detector D" section shows:
  - P(attacked) over time for the selected drone, with the 0.5 line and
    red dots where an ALARM passed persistence;
  - an alarm timeline of A, B, C and D together;
  - the model's SHA-256 and the "simulation result" label.
- **Experiments view:** with the toggle on, a fifth tab "Experimental ML"
  shows:
  - KPI cards (accuracy, precision, recall, F1, ROC-AUC);
  - a violet feature-importance bar chart (D's fixed colour);
  - the confusion matrix;
  - recall per attack type;
  - which seeds were used and the model's SHA-256.
- **No model yet** (a fresh copy, Streamlit Cloud): both places show an
  empty state with a **Train quick model** button (40 runs, about a minute)
  and the CLI command for the full model.
- New helpers:
  - `ui/state.py`: `get_ml_model()`, `ml_run()`, `train_quick_model()`,
    `models_dir()` (`SWARMGUARD_MODELS` can redirect it)
  - `ui/plots.py`: `feature_importance_bar`, `confusion_chart`,
    `ml_probability_chart`
  - `ui/components.py`: `require_ml_model`
- `get_ml_model()` caches the loaded model keyed on the file's modification
  time, so a newly trained model is picked up without restarting the app.
- **Methodology page:** one new equation (the forest's averaged vote) and
  one new limitation.

---

## 4. The test files (`tests/`)

There is one file per module, each containing exactly the checks the Phase 6
specification lists, plus the older `test_phase1.py` ... `test_phase5.py`
suites. `tests/conftest.py` shares one cached scenario runner, so several
files reuse the same simulated flight. It uses the analytical default
thresholds, so results never depend on a calibration file that a fresh
clone does not have.

| File | What it proves |
|---|---|
| `test_schema.py` | `ObservedFrame` has no truth fields; A, B, C raise `TypeError` on a `Frame` (R1) |
| `test_reproducibility.py` | same seed gives identical arrays; a different seed differs (R2) |
| `test_sensors.py` | stored range noise std = 0.3 m (F2); symmetric, diagonal 0, NaN beyond radio range |
| `test_attacks.py` | rigid attacks keep every GPS distance (noise off); labels persist through pulse gaps (F5); ramps go 0 -> 1 |
| `test_detector_a.py` | flags only Drone 04 in a hijack; silent on a rigid shift; finds both liars, no honest drone |
| `test_detector_b.py` | 10 m jump caught within 1 s; 0.5 m/s drift within 15 s; gating keeps the ~10 m innovation visible |
| `test_detector_c.py` | rigid shift caught with 30 % anchors; zero anchors gives NO_EVIDENCE (F8); median ignores 1 bad anchor of 4 |
| `test_calibration.py` | calibrate on 200 seeds, then 50 NEW clean flights stay within the Wilson bound of the 5 % target |
| `test_fusion.py` | masses sum to 1; total conflict gives (0, 0, 1); NO_EVIDENCE changes nothing |
| `test_recovery.py` | noiseless MDS exact to 1e-6; Kabsch finds a known rotation; mirror handled; RANSAC drops 1 bad anchor of 5; multilateration skips flagged drones (F1) and reports INSUFFICIENT_NEIGHBOURS |
| `test_forensics.py` | one flipped byte changes the hash; editing the manifest breaks the signature; tamper test leaves the original intact; packages chain |
| `test_ml.py` | no run in both train and test (F6); 11 NaN-free features; labels hold for the whole window; batch = stepwise; hash-checked save/load; the toggle works in the real app with and without a model |

**A lesson the calibration test taught us.** Our first version calibrated
on only 60 flights, and Detector C then alarmed on 8 of 50 new clean flights
against a 5 % target. That was not a bug. Choosing "the smallest threshold
that meets the target" on a small sample is *optimistic*, because it picks
the threshold that happened to look good on those particular flights. With
the production setting of 200 calibration flights, the chosen threshold
gave 3 % on 300 new flights. The test now uses 200 flights (10 s long, to
stay fast) and states why.

**Timing:**
- The 12 new files take about 35 s.
- The whole suite (239 tests, including the Phase 1–5 files) takes about
  2 minutes.
- It also passes in a fresh copy of the repository with a brand-new virtual
  environment, no calibration file and no model.

---

## 5. `README.md`

It covers:
- what the project is and the "simulation only" statement;
- the central idea with the cancellation equation;
- a text architecture diagram;
- the Windows quick start and every CLI command;
- the scenario catalogue, taken from the presets themselves;
- each detector with its equations;
- calibration and evaluation seed ranges and the false-alarm target;
- **empty results tables** for the team to fill from their own runs (no
  invented numbers);
- the F1–F8 table with the file that implements each fix;
- the credits;
- the limitations;
- a blank module-ownership table;
- Streamlit Community Cloud deployment.

---

## 6. Changes to earlier-phase files (and why)

| File | Change | Why |
|---|---|---|
| `cli.py` | `train-ml` command, `detect --ml` | spec 6.4 |
| `ui/state.py` | ML helpers | load, cache and run D for the views |
| `ui/plots.py` | 3 new charts | feature importance, confusion matrix, probability |
| `ui/components.py` | `require_ml_model` | one shared "no model yet" empty state |
| `views/detection.py`, `views/experiments.py` | toggle and Detector D section / tab | spec 6.1 UI |
| `views/methodology.py` | D's equation + limitation | the reference page stays complete |

Nothing else changed:
- Fusion, `run_pipeline`, `DETECTOR_ORDER` and the Phase 1–5 tests are
  untouched. D is opt-in, so every earlier result is identical.
- `detection/__init__.py` does **not** import Detector D. The ML module
  imports `analysis.calibration` and `detection` itself, so exporting it
  from the package would create a circular import. It would also make every
  page load scikit-learn. Import it as `from detection.detector_d_ml
  import MLDetector`.

---

## VIVA NOTES

**1. Why is Detector D called "experimental"?**
It learns only from our own simulator, so it can be no more realistic than
the simulation's noise and attack models. A high score means "it learned the
simulator's patterns", not "it would work on real drones". We show it to
learn which measurements matter, not as a production detector.

**2. Why is the train/test split done by run and not by row?**
Rows from one flight are 0.1 s apart and almost identical. A row split puts
near-copies in both sets, so the test score measures memory, not
generalisation. `GroupShuffleSplit` keeps every flight wholly in train or
wholly in test, and a test checks that no run id appears in both (fix F6).

**3. Why do the labels stay 1 for the whole attack window?**
During an intermittent attack's "off" half, the drone is still under attack
by the same adversary. Labelling those moments 0 would teach the model that
a paused attack is normal. The label is 1 from start to end of the attack
window (fix F5).

**4. Doesn't using attack labels break rule R1 (no truth for detectors)?**
The labels are only the *answer* the model learns to predict while
training. They are never an input feature. At detection time the model
sees only features computed from `ObservedFrame`s and from A, B and C, and
`MLDetector.update` refuses a `Frame` with `TypeError`.

**5. Which features matter most, and why do the GPS-motion features score near zero?**
CUSUM (about 0.66) and the 10-step mean NIS (about 0.18) dominate, followed
by Detector C's residual. Raw GPS speed is very noisy: 1 m GPS noise over
0.1 s gives roughly 14 m/s of noise per axis, which buries the attack
signal. Detector B's Kalman filter exists precisely to handle that noise,
so the forest leans on B's outputs instead.

**6. Why is D not included in the Dempster-Shafer fusion?**
D's inputs are A, B and C, so fusing it with them would count the same
evidence twice and inflate belief. Dempster's rule assumes independent
sources. D is also experimental, and an experiment must not change the
official verdicts.

**7. What does `class_weight="balanced"` do?**
It weights each class by 1 / its frequency. Clean rows outnumber attacked
rows, so without it the forest could score well just by saying "clean"
often. With it, missing an attacked row costs as much as a false alarm on
a clean one.

**8. What is ROC-AUC, in plain words?**
Pick one attacked row and one clean row at random. ROC-AUC is the
probability that the model gives the attacked row the higher score. 0.5 is
guessing and 1.0 is perfect. It does not depend on the 0.5 threshold.

**9. Why store and check a SHA-256 of the model file?**
A `.joblib` file is a Python pickle, and loading a pickle can execute code.
Checking the hash before loading guarantees the file is the one we trained.
It is the same integrity idea as our evidence packages, and a test flips one
byte to prove the check works.

**10. Your calibration test first failed. What did you learn?**
Calibrating on only 60 flights picked a threshold for Detector C that gave
14 % false alarms on new flights against a 5 % target. Choosing "the
smallest threshold that passes" on a small sample over-fits to that sample.
With 200 flights it gave 3 %, which is why 200 is the production default
and what the test uses.

**11. Why compare against the Wilson interval instead of requiring k/50 <= 5 %?**
With only 50 flights, 3 or 4 false alarms can happen by chance even when
the true rate is exactly 5 %. The Wilson interval shows the range of true
rates consistent with what we saw. The test fails only if even its lower
end is above the target, meaning the target is clearly exceeded.

**12. Why is there a test file per module when phase test files already exist?**
Each member defends their own module in the viva. A short file that
contains only that module's checks (for example `test_detector_b.py`) is
easy to open and explain. The phase files remain as the full regression
suite.

**13. How does the app work on Streamlit Cloud with no data folder?**
`data/` and `keys/` are never committed. Detectors fall back to the
analytical default thresholds, and the choice is logged (fix F3). The
signing keys are generated on the first evidence package. The ML views
offer a "Train quick model" button. We verified this by running the full
test suite and the app from a fresh copy with a new virtual environment.

**14. Did Detector D ever give a wrong answer you can show?**
Yes. On `single_hijack` (seed 5000) it caught Drone 04 within 0.2 s, but it
also briefly alarmed on four honest drones, so its precision was 0.20.
Their average attack probability was only 0.15. The alarms came from rare
noise spikes that passed the 3-of-5 persistence rule somewhere in 600
steps. This is a concrete reason the detector stays experimental.
