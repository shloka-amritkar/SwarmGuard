# Phase 3 Explained: Fusion, Recovery, Experiments, Evasion Search, Propagation

Phase 2 gave us three detectors that each say "this drone's GPS looks wrong"
or not. Phase 3 answers the next five questions:

1. **What do the detectors say together?** Fusion (`detection/fusion.py`).
2. **Where are the drones really?** Recovery (`recovery/`).
3. **How well does this work in general, not just in one run?** Sweeps
   (`analysis/experiments.py`).
4. **What is the smallest attack each detector reliably catches?** Evasion
   search (`analysis/evasion_search.py`).
5. **How far does one spoofed drone disturb the rest of the swarm?**
   Propagation (`analysis/propagation.py`).

```
simulate ─► detectors A, B, C ─┐
        └─► recovery (MDS + RANSAC on ranges/anchors) ─► outlier flags ─┐
                               └──────────────► fusion ◄────────────────┘
                                                   │
               run_pipeline() = all of the above ─► PipelineResult
               run_trial() = one seed, scored ─► sweep() / stealth_envelope()
```

---

## 1. `detection/fusion.py`: combining evidence without hiding uncertainty

**Masses.** For each drone and step, every detector gives three numbers that
add up to 1:
- `m_attack`: evidence that the drone is attacked.
- `m_normal`: evidence that it is not.
- `m_uncertain`: "I don't know."

A single probability cannot tell "50/50" apart from "no idea". Masses can.

Each detector first produces a raw belief `b` between 0 and 1. Then:

    m_attack = α·b,  m_normal = α·(1 − b),  m_uncertain = 1 − α

| Detector | Raw belief b | Reliability α |
|---|---|---|
| A | its vote score (share of bad pairs) | 0.70 |
| B | max(1 − e^(−S/h), min(NIS / (3·nis_threshold), 1)) | 0.80 |
| C | clip((r − tau_c/2) / tau_c, 0, 1) | 0.85, or 0.60 while RANSAC rejects anchors |

`NO_EVIDENCE` → (0, 0, 1): complete ignorance, which changes nothing when
combined.

**Why α < 1 (discounting).** Dempster's rule assumes the sources are
independent, but A and B both read the same GPS. Keeping some mass as
"unknown" limits double counting. We state this as a limitation, not a solved
problem.

**Dempster's rule** for two sources:

    K = m1a·m2n + m1n·m2a            (how much they contradict each other)
    a = (m1a·m2a + m1a·m2u + m1u·m2a) / (1 − K)
    n = (m1n·m2n + m1n·m2u + m1u·m2n) / (1 − K)
    u = m1u·m2u / (1 − K)

If K > 0.95 the sources flatly contradict each other. We return (0, 0, 1) and
flag `high_conflict` instead of trusting the tiny remainder. The rule is
commutative and associative, so combining A, B, C in any order gives the
same answer.

**Worked example (also a test).** A says "normal" (0, 0.7, 0.3) and B says
"attack" (0.8, 0, 0.2). Then K = 0.7 × 0.8 = 0.56, and the result is
a = 0.24/0.44 = 0.55, n = 0.32, u = 0.14.

**Verdict.** Belief = a and plausibility = a + u. The rules are checked in
this order:
1. Plausibility < 0.3 → NORMAL.
2. Belief ≥ 0.7 → COMPROMISED.
3. u > 0.8 → INSUFFICIENT_EVIDENCE.
4. Otherwise → SUSPICIOUS.

**Fused alarm needs persistence (our addition).** On one evaluation run, a
single large GPS noise spike on an honest drone made A, B and C all
suspicious *at the same step*, because they all read that same GPS value.
Fusion called that one step COMPROMISED. The fused alarm now uses the same
3-of-5 rule as the detectors, while the per-step verdicts stay visible. In the
20-seed magnitude sweep, fusion's false-alarm rate fell from 10 % to 0 %.
The price is that fusion now catches only 5 % of 2 m offsets (it caught
80 % before); from 3 m upward it still detects 100 %.

**Agreement pattern.** `pattern_key(runs)` names which detectors alarmed in
the run, for example `A_silent_B_C_alarm`. The groups are always listed in the
order silent, alarm, no_evidence. Knowing *which* detectors disagree is more
useful than any averaged score. A silent while B and C alarm is the fingerprint
of a coordinated swarm shift, because A is blind to it by mathematics.
`agreement_matrix` shows the same thing per drone.

---

## 2. `recovery/mds_alignment.py`: where is the swarm really?

The spoofer controls GPS, but not the radio ranges and not the anchors. So
we rebuild the swarm without using GPS at all.

**Step 1: shape from distances (classical MDS).** Radio ranges give every
pairwise distance, collected in a matrix D. Missing pairs are filled with
Floyd–Warshall shortest paths (`D[i,j] = min(D[i,j], D[i,k] + D[k,j])`); if
some pair still cannot be reached, the graph is disconnected and we raise
`RecoveryError`. Then:

    J = I − (1/n)·11ᵀ,   B = −½·J·D²·J,   B = V Λ Vᵀ,   X = V₂·√Λ₂

Double centering turns squared distances into dot products of centred
points. The top two eigenvectors rebuild the shape, but only up to a
rotation, a mirror image and a shift.

**Step 2: place the shape (Kabsch).** We find the rotation R and shift t that
best map the MDS points onto the anchored drones' fixes. This uses an SVD of
`H = AᵀB`, and `d = sign(det(V Uᵀ))` prevents a mirror being returned as a
"rotation". MDS may give a mirrored shape, so we fit both the shape and its
mirror and keep the better one.

**RANSAC (robust, fix F4).** A faulty anchor would pull the whole fit. RANSAC
tries every 3-anchor subset (or a random sample of subsets if there are too
many), counts how many anchors agree within 3 m, keeps the best, and refits
on those inliers. The rest are reported as outlier anchors. With fewer than 3
anchors the result is `INSUFFICIENT_ANCHORS`. This robust version is the
default.

**Step 3: estimate the attack.** `gps − recovered` is each drone's apparent
GPS error. Under a rigid shift D every drone shows about D, so the mean
estimates the attack vector. Over a run, `summary()` takes the median over the
last 5 s. An anchor counts as an outlier only if RANSAC rejected it in at
least half of those steps, which stops one noisy frame from accusing an
honest anchor.

**Result (20 evaluation seeds, simulation result).**
- rigid_swarm_shift: the attack vector was estimated within 0.22 m of the
  true (10, 0) m on every seed.
- faulty_anchor_shift: RANSAC named exactly the faulty anchors on 20/20 seeds.

---

## 3. `recovery/multilateration.py`: repairing single drones

If drone i is flagged but its neighbours are trusted, we find the point p that
best fits the distances to them:

    e_j(p) = |p − gps_j| − range_ij,    minimise Σ ρ(e_j²)

This uses `scipy.optimize.least_squares` with `soft_l1` loss. That loss
behaves like ordinary least squares for small errors but only grows
linearly for big ones, so one bad range cannot dominate. Uncertainty is
estimated as `cov ≈ (JᵀJ)⁻¹ · Σres² / (m − 2)`.

**F1: flagged neighbours are never used.** A spoofed neighbour's GPS is wrong,
and ranging to it would copy the attacker's offset into the repair. With
fewer than 3 trusted neighbours the status is `INSUFFICIENT_NEIGHBOURS`.

**Limitation.** In a swarm-wide shift every drone is spoofed and flagged, so
nobody is left to trust. That case is handled by MDS + anchors.
`recovery_error()` compares the before and after errors against the truth, for
evaluation only.

---

## 4. `analysis/experiments.py`: from one run to statistics

- `run_pipeline` runs simulate → detect → recover → fuse. Recovery comes
  before fusion because fusion needs to know when RANSAC rejected anchors.
  Phases 4 and 5 reuse this function.
- `run_trial` runs one seed and scores A, B, C and F with the Phase 2
  metrics, plus the recovery error. It is a pure function, so it can run in a
  separate process.
- `sweep` runs every (value, seed) pair. With `workers > 1` it uses a
  `ProcessPoolExecutor`, and results are put back in task order so the output
  never depends on timing.
- Thresholds come from the base scenario's calibration and are held fixed.
  A sweep asks how fixed thresholds cope when conditions change, as they would
  in a real deployment.
- Seeds start at 5000 (evaluation). Seeds below that are refused.

**Example (single_hijack, 20 seeds, simulation result):**

| Offset | A | B | C | F |
|---|---|---|---|---|
| 1 m | 0 % | 0 % | 0 % | 0 % |
| 2 m | 0 % | 100 % | 20 % | 5 % |
| 3 m | 0 % | 100 % | 20 % | 100 % |
| 6 m | 95 % | 100 % | 20 % | 100 % |
| 10 m+ | 100 % | 100 % | 20 % | 100 % |

B is the most sensitive. A needs about 6 m because of its 4.3 m pair
threshold plus the majority vote. C is stuck at 20 %, because Drone 04
carries an anchor in only 4 of the 20 seeds.

---

## 5. `analysis/evasion_search.py`: the stealth envelope

**Bisection** finds the smallest attack strength x whose detection rate
reaches the target (50 % or 90 %). If `rate(mid) ≥ target`, search lower;
otherwise search higher. Stop when the search range is narrower than `tol`.

**Assumption, stated openly:** the detection rate rises steadily with attack
strength. That is true on average, but 40 seeds give some sampling wobble.

A detection only counts if it happens within `max_delay_s` (10 s by default).
A drift caught after 40 s has already done its damage. If even the largest
value in the range is not detected often enough, the answer is "not reliably
detected in range", never a made-up number. One trial scores all four
detectors, so a shared cache lets the different bisections reuse each other's
runs.

**Result (slow_drift, 40 seeds, within 10 s, simulation result):**

| Detector | 50 % at drift ≥ | 90 % at drift ≥ |
|---|---|---|
| A | 0.539 m/s | 0.625 m/s |
| B | 0.258 m/s | 0.297 m/s |
| C | not reliable (only 30 % of drones anchored) | — |
| F | 0.398 m/s | 0.430 m/s |

So a measurable claim reads: "Detector B detects 90 % of single-drone GPS
drifts of at least 0.30 m/s within 10 s (simulation result)."

---

## 6. `analysis/propagation.py`: how one spoof spreads

**Sensitivity matrix.** First run a clean flight. Then spoof drone i alone
(6 m), using the same seed so the sensor noise is identical. Then:

    M[i, j] = mean over the attack window of |true_j(attacked) − true_j(clean)|

The diagonal (how far the spoofed drone itself was dragged) is reported
separately as `self_displacement`.

**PageRank (our own implementation, no networkx).** `W[j, i]` is the weight of
the edge i → j. We column-normalise W (empty columns become uniform) and
iterate `r ← d·W r + (1 − d)/n` until it stops changing.
- influence = PageRank(M): high for drones whose spoofing moves many others.
- susceptibility = PageRank(Mᵀ): high for drones that get pushed around.

**Both centre gains (design decision D1).** Result on clean_baseline, seed
5000, 6 m spoof (simulation result):

| centre_gain | Other drones displaced (mean) | Spoofed drone dragged |
|---|---|---|
| 0.0 | 0.583 m (≈ D/N = 0.6 m) | 5.29 m |
| 0.5 | 0.017 m | 5.86 m |

The circle formation is symmetric, so PageRank is almost uniform (about
0.10 each). That is expected; less symmetric shapes show more differences.

**Near collisions.** `min_separation` tracks the closest true distance
between any two drones over time. Dragging Drone 03 or Drone 09 by 6 m brought
it within about 0.92 m of a neighbour. Events use hysteresis: a pair is
counted again only after separating beyond 2 m. Without this, noise made one
close pass count 40 times.

---

## 7. `cli.py` additions

```
.venv\Scripts\python cli.py detect --scenario rigid_swarm_shift --seed 5000
.venv\Scripts\python cli.py sweep --param magnitude --scenario single_hijack --seeds 20 --workers 4
.venv\Scripts\python cli.py evasion --scenario slow_drift --workers 4
.venv\Scripts\python cli.py propagation --scenario clean_baseline
```

`detect` now also prints fusion (F), the pattern with its explanation, and the
recovery estimate. Rough runtimes with 4 workers: sweep (11 values × 20 seeds)
about 30 s; evasion about 2 min; propagation about 1–2 s.

---

## 8. Tests (`tests/test_phase3.py`, 30 checks)

The tests cover:
- **Fusion:** masses always sum to 1; total conflict gives (0, 0, 1);
  NO_EVIDENCE is neutral; the Dempster worked example; the verdict rules; C's
  reliability drop; the rigid-shift pattern; a quiet run giving `A_B_C_silent`.
- **Recovery:** MDS rebuilds noiseless geometry to 1e-6; Kabsch recovers a
  known transform and never returns a mirror; Floyd–Warshall fills gaps and
  detects a split graph; RANSAC rejects 1 faulty anchor of 5 and needs 3
  anchors; the rigid-shift attack vector is within 2 m; RANSAC names the
  faulty anchors; recovery refuses truth frames.
- **Multilateration:** flagged neighbours are excluded (F1);
  INSUFFICIENT_NEIGHBOURS with fewer than 3.
- **Experiments and evasion:** sweep parameters are validated; trials are
  deterministic; recovery beats raw GPS error; a small sweep rises with
  magnitude; calibration seeds are refused; bisection works on a known
  function; the search parameter follows the attack type.
- **Propagation:** PageRank on a hand-drawn graph and a dangling graph; open
  loop is refused; centre_gain 0 spreads about D/N; one event per close pass.

---

## VIVA NOTES

**1. Why Dempster-Shafer instead of averaging the detector scores?**
It keeps "I don't know" separate from "50/50". A detector with no evidence
adds (0, 0, 1) and changes nothing, while an average would treat it as a vote
for normal. It also measures conflict (K) between detectors, which an average
hides.

**2. What is the reliability α, and why is it below 1?**
It discounts each detector: part of its mass goes to "unknown". Dempster's
rule assumes independent sources, but A and B both use the same GPS reading,
so full trust would double count. Discounting limits that; we state it as a
limitation.

**3. What happens when detectors completely disagree?**
If the conflict K exceeds 0.95, normalising by 1 − K would blow a tiny
remainder up into near certainty. We return total ignorance (0, 0, 1) and flag
`high_conflict` so a human looks at it.

**4. Why does the fused alarm need persistence when the detectors already have it?**
The per-step fused verdict combines *raw* evidence. A, B and C all read the
same GPS value, so one big GPS noise spike can make all three suspicious at
once. We saw this cause a one-step false COMPROMISED. Requiring 3 of 5 steps
removed it (10 % → 0 % false alarms in the sweep).

**5. What does the pattern A_silent_B_C_alarm mean?**
Relative geometry sees nothing while motion and anchors disagree with GPS.
Because A is mathematically blind to a common shift, this combination is the
fingerprint of a coordinated swarm-wide shift. It identifies an attack
pattern, not an attacker.

**6. How does MDS rebuild positions, and what can it not know?**
Double centering turns squared distances into the Gram matrix B = XXᵀ, and its
top two eigenvectors give the coordinates. Distances do not change under
rotation, reflection or translation, so MDS cannot know those. The anchors fix
them.

**7. Why do you try both the MDS shape and its mirror image?**
Distances are identical for a shape and its mirror, so MDS may return either.
Kabsch only finds rotations (its d term forbids reflections), so we fit both
versions and keep the one with the smaller residual.

**8. How does RANSAC find faulty anchors?**
It fits the shape on every 3-anchor subset and counts how many anchors agree
within 3 m. The subset most anchors agree with wins, and the rest are
outliers. It works while honest anchors are the majority; if most anchors are
faulty, it fails.

**9. How is the attack vector estimated?**
Recovered positions come only from ranges and anchors, which the spoofer does
not control. So gps − recovered is each drone's GPS error, and the mean over
the swarm estimates a common offset. On rigid_swarm_shift it was within
0.22 m of the true 10 m east on 20/20 seeds.

**10. Why must multilateration exclude flagged neighbours (F1)?**
A flagged neighbour's GPS is probably spoofed. Using it as a reference copies
the attacker's offset into the "repaired" position. With fewer than 3 trusted
neighbours we honestly return INSUFFICIENT_NEIGHBOURS.

**11. Why is the sweep's train/test separation important?**
Thresholds are calibrated on seeds 1000+ and sweeps use seeds 5000+. Testing
on the same flights used for tuning would make the results look better than
they really are, like grading a student on the exam questions they practised.

**12. What does bisection assume, and when could it go wrong?**
It assumes the detection rate rises steadily with attack strength. With 40
seeds the measured rate wobbles, so the threshold is accurate to roughly the
tolerance plus sampling noise. If the true rate were not monotonic, bisection
could miss a region.

**13. Why measure detection "within 10 s"?**
Late detection is little use: a drift detected after 40 s has already moved
the swarm. The time limit makes the claim operational, e.g. "B detects 90 % of
drifts of at least 0.30 m/s within 10 s".

**14. What does the sensitivity matrix show, and why two centre gains?**
M[i, j] is how far drone j's true path moves when only drone i is spoofed. With
centre_gain 0, the consensus spreads the spoof as about D/N to every drone
(0.58 m). With 0.5, the path term pulls them back (0.017 m). The control law
changes the answer, so we always report both (D1).

**15. Which direction does your PageRank graph point, and why?**
`W[j, i]` is the edge i → j. Influence = PageRank(M), so a drone ranks high
when the drones it displaces "vote" for it, meaning it is a good target to
attack (the SwarmFuzz seeding idea). PageRank(Mᵀ) instead ranks the most
susceptible drones.
