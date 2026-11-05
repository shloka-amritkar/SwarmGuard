# Phase 4 Explained: Forensics, Storage, PDF Reports

Phase 3 could tell us **what happened** in a run. Phase 4 makes that answer
**usable as evidence**. It answers four questions:

1. **What exactly did we find, and how can someone repeat it?** The incident
   record (`forensics/incident.py`) and its explanation (`forensics/explainer.py`).
2. **Where is it kept?** A SQLite database (`database/storage.py`).
3. **How can a reader prove nobody changed it later?** Hashes, an Ed25519
   signature and a custody chain (`forensics/hashing.py`, `signatures.py`,
   `evidence_package.py`).
4. **How does a human read it?** A PDF report (`reports/pdf_report.py`).

```
run_pipeline() ─► should_open_incident? ─no─► "no anomaly detected" (nothing stored)
                         │yes
                         ▼
     Storage.save_run ─► incident id SG-YYYYMMDD-NNN ─► from_run() ─► explain_incident()
                         │
                         ▼
build_package():  CSVs + JSON + figures ─► hash them ─► report.pdf (lists those hashes)
                  ─► manifest.json (hash of EVERY file) ─► manifest.sig (Ed25519)
                  ─► .zip ─► custody_log "created" (previous_manifest_hash = chain link)

verify_package():  hashes? signature? signer key? chain?  ─► PASS / FAIL / WARN / UNKNOWN
tamper_test():     copy ─► flip one bit ─► verify the copy ─► FAIL (original untouched)
```

---

## 0. `core/palette.py`: one place for colours

The PDF must look like the app, and the app's colours may still change
before Phase 5. So every colour lives in one frozen dataclass, `PALETTE`:
cream `#F4F1EA`, ink `#111111`, deep red `#C8221B`, blush `#F4B6B2`, muted
`#8A8577`, and the status colours. The PDF and its matplotlib figures read
`PALETTE.red` and the other names. `ui/theme.py` will import the same module
in Phase 5. Change a hex value here and the app, the PDF and the figures all
follow.

**Why in `core/` and not `ui/`?** The PDF is built from the command line.
Importing `ui/` would drag Streamlit in. `core/` has no UI dependencies.

`frozen=True` means `PALETTE.red = "..."` raises an error, so no code can
quietly change a colour while the program runs.

---

## 1. `forensics/hashing.py`: fingerprints of files

**SHA-256** turns any input into 64 hex characters. Change one bit of the
input and about half the output bits change ("avalanche effect"). Nobody
knows how to find two inputs with the same hash. So "the hash still matches"
means "the file is byte-for-byte unchanged".

- `sha256_file(path, chunk=65536)` reads the file 64 KiB at a time. A
  telemetry CSV is about 1.7 MB, and chunking keeps memory use small. The
  result is identical to hashing the whole file at once (a test checks this).
- `canonical_json(obj)` is `json.dumps(sort_keys=True, separators=(",", ":"),
  ensure_ascii=False, default=str)` encoded as UTF-8.

**Why canonical JSON?** `{"a":1,"b":2}` and `{"b": 2, "a": 1}` are the same
dictionary but different bytes, so they would get different hashes and
different signatures. Sorting the keys and removing optional spaces gives
every equal dictionary exactly one byte form.

**Limitation:** a hash proves nothing about who made the file. Anyone who
edits a file can recompute its hash. That is what the signature is for.

---

## 2. `forensics/signatures.py`: Ed25519 signatures

A key pair has a **private key** (secret, used to sign) and a **public key**
(shareable, used to verify).

    sig = Sign(private_key, manifest_bytes)            64 bytes
    Verify(public_key, manifest_bytes, sig) -> True / False

- `ensure_keypair("keys")` creates the pair on the first run and reuses it
  afterwards.
  - If the environment variable `SWARMGUARD_KEY_PASS` is set, the private key
    is stored encrypted (PKCS8, `BestAvailableEncryption`).
  - If it is not set, the key is stored unencrypted and a **warning is
    logged**. We do not hide that weakness.
  - It also calls `chmod 600` (owner-only). On Windows that only toggles
    "read-only", and we log that too.
- `verify()` catches `InvalidSignature` and returns `False`. For a verifier,
  "this was changed" is a normal answer, not a crash.
- `public_key_fingerprint()` is the first 16 hex characters of
  sha256(raw 32-byte public key). That is short enough to compare by eye
  between the PDF, the CLI and the app.

**Why Ed25519?**
- Small keys (32 B) and small signatures (64 B).
- Fast.
- Supported by the standard `cryptography` library.
- **Deterministic**: the same message always gives the same signature. Older
  schemes such as ECDSA need a fresh random number for every signature, and
  reusing one leaks the private key (this happened to the PlayStation 3).

`keys/` is in `.gitignore`. The private key must never reach GitHub.

**Limitation (stated in every PDF report):** the signing key is stored
unencrypted on disk; production use would require a protected key
(passphrase or hardware module). For this project we deliberately keep it
unencrypted so that demos and tests run without typing a passphrase.

---

## 3. `database/storage.py`: SQLite

Six tables:

| Table | Holds |
|---|---|
| `runs` | each simulation: scenario, seed, config hash and JSON, summary |
| `incidents` | each incident: id, pattern, affected drones, onset, full JSON |
| `detections_summary` | per-detector metrics of a run |
| `evidence_packages` | where each package folder and zip were written, and its manifest hash |
| `calibrations` | calibrations by config hash |
| `custody_log` | append-only events: created / verified / verify_failed |

Key design points:

- **`CREATE TABLE IF NOT EXISTS`** makes the schema idempotent: opening the
  database a hundred times is safe.
- **Parameterised queries** (`WHERE id = ?` with a value tuple). The driver
  sends values separately from the SQL text. A test stores the pattern name
  `x'; DROP TABLE incidents; --` and checks that the table survives: that
  is SQL injection prevented.
- **`_connect()` context manager**: it commits on success, rolls back on
  error, and always closes the connection. Plain `with sqlite3.connect()`
  commits but does *not* close, which keeps the file locked on Windows.
- **`next_incident_number(date)`** uses the largest existing NNN for that
  date, plus 1. Using the max rather than the count means a deleted row can
  never cause an id to be reused.
- **`last_manifest_hash()`** returns the manifest hash of the latest
  `"created"` custody row, or `"GENESIS"`. Verification events do not move
  the chain.

---

## 4. `forensics/incident.py`: the incident record

An `Incident` dataclass holds three groups of fields.

**Provenance**, so the run can be repeated exactly:
- The scenario, the configs and the seed.
- `calibration_hash`: the SHA-256 of the calibration file used, or "none
  (analytical defaults)".
- `software_version` (`VERSION = "0.4.0"`).
- `git_commit`: from `git rev-parse HEAD`, run **without a shell** and with
  a 5 s timeout. If git is missing it falls back to "unknown".

**Findings:**
- `detected[A/B/C]`: state, first alarm time, drones, peak score, threshold.
  For B it also stores the peak NIS and its threshold.
- `affected_drones_detected`: the union of drones any detector flagged.
- `onset_estimate`: the earliest **CUSUM change point** among flagged
  drones. The change point is the last step where CUSUM was 0 before it
  alarmed, so it says when the evidence *started* building up. An alarm
  time is always later than that.
- `fusion`: for each drone, its verdict, belief and plausibility at the step
  of **peak belief**. An intermittent attack may already be over at the last
  step, so the peak shows the strongest evidence fusion ever had.
- `pattern` (e.g. `A_silent_B_C_alarm`), the recovery summary, and the
  explanation sentences.

**Ground truth.** `simulation_ground_truth` holds `{type, targets, onset}`,
labelled `"SIMULATION GROUND TRUTH: not available to detectors"`. Detectors
never see it (rule R1). It is there so an examiner can compare the findings
with what really happened. On the rigid shift, the estimated onset is 24.6 s
and the true onset is 25.0 s.

`should_open_incident(runs)` is True if any detector raised any ALARM. A
clean run therefore produces no incident: the CLI prints "no anomaly
detected" (R4). `--force` overrides this.

---

## 5. `forensics/explainer.py`: sentences only from facts

`explain_incident()` fills **templates** with fields of the incident, in a
fixed order:

1. overview
2. one sentence each for A, B and C
3. onset
4. pattern
5. fusion
6. recovery (one or two sentences)
7. three fixed limitation sentences

That gives 10 to 12 sentences. If a field is missing, its sentence is
skipped. Nothing is invented.

Example, from the real rigid-shift run:

> Detector B (motion consistency) first alarmed at 25.0 s on 10 drones (...);
> its NIS peaked at 179.4, far above the 10.71 threshold ...

("far above" is used when the value is at least 3 times the threshold.)

**A bug this phase caught.** A silent detector's peak can be *above* its
threshold for a single noisy step, while the 3-of-5 persistence rule never
confirms it. The first draft said "peaked at 4.5, below the 3.4 threshold",
which is false. The sentence now says "briefly reached 4.5 m, above the
3.4 m threshold, but never on enough consecutive steps to pass the
persistence rule". A test guards against this.

**Honest language (R4):**
- A silent detector "detected no anomaly". It never says "GPS genuine" or
  "safe".
- Every explanation ends with "No anomaly detected does not prove GPS was
  genuine." and "The system does not identify who caused this."

`explain_drone(drone, step_results)` gives one short reason per detector for
the Phase 5 decision inspector, e.g. "B: ALARM - NIS 48.2 vs 10.71, CUSUM
312.0 vs 37.0".

---

## 6. `reports/pdf_report.py`: the PDF

The report is built with ReportLab **platypus**. We list flowables
(paragraphs, tables, images) and ReportLab breaks them across A4 pages
itself.

**Sections:**
1. Cover (SWARMGUARD, incident id, time, "SIMULATION REPORT", disclaimer).
2. Executive summary (the explainer's sentences).
3. Configuration table.
4. Timeline figure.
5. Detector table.
6. Fusion verdicts.
7. Recovery.
8. Evidence integrity: the file hashes in Courier, the key fingerprint and
   the previous manifest hash.
9. Limitations.

Every page has a header and footer with the incident id and the page number.

- **Style:** white paper for printing, deep-red headings and rules, blush
  table header rows, black grid lines. All of these come from `PALETTE`.
- **Fonts:** the built-in Helvetica and Courier. They are always available,
  there are no font files to ship, and the spec allows Helvetica.
- **`timeline_figure()`** (matplotlib, `Agg` backend, no window) draws one
  row per (detector, alarmed drone), with bars where that detector was in
  ALARM, and a dashed line at the estimated onset. The heading, caption and
  figure are wrapped in `KeepTogether` so the heading is never stranded at
  the bottom of a page.
- **`trajectory_figure()`** draws the reported GPS (dotted) and the true
  paths (solid). Its legend labels the true paths "SIMULATION GROUND TRUTH".

**Why the report cannot contain its own hash:** writing the hash into the
file changes the file, and therefore the hash. So the report lists the other
files' hashes, and its own hash goes into `manifest.json`, which is signed.

---

## 7. `forensics/evidence_package.py`: build, verify, tamper-test

### Build (the order matters)

1. Write `telemetry_observed.csv`, `ground_truth_SIMULATION.csv`,
   `detections.csv` (A, B, C and F stacked, with a `detector` column),
   `fusion.csv`, `config.json`, `incident.json`, `figures/*.png` and
   `public_key.pem`.
2. Hash them, then build `report.pdf`, which prints those hashes.
3. Write `manifest.json`, listing `{path, sha256, size_bytes}` for every
   file, sorted by path. Paths use forward slashes, so a package made on
   Windows verifies on Linux.
4. `manifest.sig` = the Ed25519 signature over `canonical_json(manifest)`.
5. Zip the folder to `<incident_id>.zip`. Log custody `"created"` and save
   the package row.

An existing package folder is **never overwritten**: building it again
raises a `ValueError`.

### The chain

    manifest_hash             = sha256(canonical_json(manifest))
    previous_manifest_hash(k) = manifest_hash(k - 1)        ("GENESIS" for the first)

Package 2 commits to package 1, package 3 to package 2, and so on. Suppose
someone deletes package 2, or replaces it with a re-signed fake. Package 3
still points at the *original* hash of package 2, and the custody log
disagrees. This is the same idea as a blockchain or an append-only audit
log.

### Verify (folder or zip)

| Check | Result |
|---|---|
| every listed file exists and its hash matches | PASS / FAIL |
| files present but not listed | WARN |
| signature over the canonical manifest | PASS / FAIL |
| signer key equals the trusted key (`keys/public_key.pem`) | PASS / FAIL / UNKNOWN |
| chain: custody row matches, and previous = the package created just before | PASS / FAIL / UNKNOWN |

The overall result is FAIL if any check fails. WARN and UNKNOWN are reported
but are not fatal.

**Why the signer-key check?** An attacker could replace the *whole* package,
including its `public_key.pem`, and sign it with their own key. The
signature check alone would pass. Comparing the fingerprint with the key this
installation trusts (also printed on the PDF) catches that.

**Zip slip.** A malicious zip can contain names like `../../evil.py`, which
would be written outside the folder we extract into. `_safe_extract` refuses
any such path.

### Tamper test

1. Copy the package to a temporary folder.
2. Flip one bit: `byte = byte XOR 0x01`.
3. Verify the copy.

The byte is chosen with `np.random.default_rng(0)` unless one is given (R2).
The original is only read, never written, and a test checks that its bytes
are unchanged.

---

## 8. `cli.py` additions

    .venv\Scripts\python cli.py investigate --scenario rigid_swarm_shift --seed 5000
    .venv\Scripts\python cli.py verify data\packages\SG-20261105-001.zip
    .venv\Scripts\python cli.py tamper-test data\packages\SG-20261105-001 --file detections.csv

- `investigate` runs the full pipeline. If something alarmed, it stores the
  run and its metrics, opens the incident, builds the package, and prints
  the explanation and all paths.
- `verify` prints every check, logs a "verified" or "verify_failed" custody
  event, and exits with code 0 only if nothing failed.
- `tamper-test` shows the changed byte (e.g. `0x34 -> 0x35`) and the failing
  check, then re-verifies the original to show it still PASSes.

`drone_label` moved from `cli.py` to `core/utils.py` (together with
`format_drones`), so the CLI, the explainer and the PDF all name drones the
same way. The CLI now imports it from there; its behaviour is unchanged.

**Acceptance results (simulation results):**
- `investigate` on `rigid_swarm_shift`, seed 5000, opened SG-20261105-001
  with pattern `A_silent_B_C_alarm`. Recovery estimated an offset of
  (10.10, 0.05) m.
- `verify` passed every check.
- `tamper-test` gave FAIL on the copy (the `detections.csv` hash differs),
  while the original still passed.
- A second investigation (`single_hijack`) had
  `previous_manifest_hash = 5d98c35b...`, which is exactly the first
  package's manifest hash, and its chain check passed.

---

## 9. Tests (`tests/test_phase4.py`, 20 checks)

All tests write to pytest's `tmp_path`, so the real `data/` and `keys/`
folders are never touched. They cover:

- **Palette:** its values, and that it cannot be changed (frozen).
- **Hashing:** key order does not matter, chunked hashing matches whole-file
  hashing, and the known SHA-256 of the empty string.
- **Signatures:** sign and verify round trip; a changed message or a flipped
  signature bit fails; the fingerprint format; keys are reused; an
  encrypted key works with the right passphrase and fails with a wrong one.
- **Storage:** the schema is idempotent, the incident counter works, the
  injection string is stored safely, and the custody helpers behave.
- **Incident:** fields for the rigid shift (pattern, onset close to the
  truth, recovery about 10 m) and a JSON round trip.
- **Explainer:** sentence count, required limitations, no "safe" or "GPS
  genuine", numbers quoted from fields, no false "below threshold", and the
  C_silent_A_B_alarm description names both possible causes of C's silence.
- **Packages:** all PASS for both folder and zip; tamper gives FAIL on the
  copy while the original stays unchanged and still passes; the second
  package chains to the first; the chain is UNKNOWN outside the database; an
  extra file gives WARN; a missing file gives FAIL; a foreign signing key
  gives FAIL; tampering with the manifest gives FAIL; packages are never
  overwritten.

---

## VIVA NOTES

**Q1. Why sign the manifest if every file already has a hash?**
A hash shows a file is unchanged only if you trust the list of hashes. An
editor can change a file, recompute its hash and edit the manifest. They
cannot produce a valid Ed25519 signature over the edited manifest without
the private key, so the signature protects the list itself.

**Q2. What does a PASS from `verify` prove, and what does it not?**
It proves the files are byte-for-byte what was signed, by the holder of the
trusted key, and that the package sits correctly in the chain. It does not
prove the telemetry was genuine. If the input data was false when we signed
it, the signature faithfully preserves false data (rule R4).

**Q3. Why canonical JSON?**
The same dictionary can be written with different key orders or spacing,
which gives different bytes and therefore different hashes and signatures.
Sorting the keys and using fixed separators gives one byte form per
dictionary, so we sign and hash the *content*, not the formatting.

**Q4. Why Ed25519 instead of RSA or ECDSA?**
Its keys and signatures are small (32 and 64 bytes), it is fast, and it is
well supported. It is also deterministic: ECDSA needs a fresh random number
for every signature, and reusing one leaks the private key. Ed25519 derives
that value from the message, so this mistake cannot happen.

**Q5. How does the custody chain work, and what attack does it stop?**
Each manifest stores the hash of the previous package's manifest ("GENESIS"
for the first), and the custody log records every link. If an earlier
package is deleted or swapped for a re-signed fake, the next package's
`previous_manifest_hash` no longer matches. It is an append-only audit log,
like a blockchain without the network.

**Q6. Why is the chain sometimes UNKNOWN instead of PASS?**
The links are checked against our SQLite custody log. A package verified on
another machine has no access to that log, so we honestly report UNKNOWN
rather than claiming PASS or FAIL. The hashes and the signature can still be
checked anywhere.

**Q7. Why does the report not contain its own hash?**
Writing a hash into a file changes the file, and therefore its hash, so a
file cannot contain its own correct SHA-256. The report lists the other
files' hashes; its own hash is in `manifest.json`, which the signature
covers.

**Q8. How do you prevent SQL injection?**
Every query uses `?` placeholders, and the values travel separately from the
SQL text, so they are never executed. A test stores
`x'; DROP TABLE incidents; --` as a pattern name and checks that the table
survives and the text comes back unchanged.

**Q9. How is the onset estimated, and why not just use the first alarm?**
An alarm comes after evidence has piled up, so it is always late. Detector
B's CUSUM records its change point, the last step where CUSUM was 0 before
it alarmed, which marks when the anomaly started. On the rigid shift the
estimate is 24.6 s against a true onset of 25.0 s (the true onset is
evaluation-only ground truth); the first alarm came later, at 25.0-25.2 s.

**Q10. How do you make sure the explanation never makes things up?**
Every sentence is a template filled from one field of the incident, and if
the field is missing the sentence is skipped. While testing we found one
template that said "below threshold" when a single noisy step was actually
above it. We fixed it so the sentence states the persistence rule, and
added a test for it.

**Q11. Why does the incident contain the ground truth, if detectors must not see it?**
Rule R1 is about detector inputs. The detectors finish before the incident
is built, and they never receive it. The truth is stored in a clearly named
file and field ("SIMULATION GROUND TRUTH: not available to detectors"), so
that an examiner can check the findings against what really happened.

**Q12. What does the tamper test show, and why work on a copy?**
It flips a single bit (XOR 0x01) in one file of a copy and re-verifies:
that file's hash check fails and the overall result becomes FAIL. Working on
a copy means the demonstration can never damage real evidence. A test checks
that the original's bytes are unchanged and that it still passes.

**Q13. Could an attacker replace the public key inside the package?**
Yes, and they could sign with their own key, so the signature check alone
would pass. That is why verification also compares the package's key
fingerprint with the trusted key in `keys/` (and the fingerprint printed on
the PDF). A foreign key gives FAIL.

**Q14. Where are the weaknesses?**
- The signing key is stored unencrypted on disk; production use would
  require a protected key (passphrase or hardware module). This is a
  deliberate choice for the project (no `SWARMGUARD_KEY_PASS` is set, and a
  warning is logged), and on Windows chmod gives only weak protection. The
  PDF's limitations section says the same.
- Whoever holds the private key could rebuild the entire chain.
- Dempster's rule assumes independent sources, while A and B both read GPS.
The report states these limits instead of hiding them.
