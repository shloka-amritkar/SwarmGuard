############################################################################
# PHASE 4 — FORENSICS, STORAGE, PDF REPORTS
# Files: forensics/incident.py, forensics/hashing.py,
#        forensics/signatures.py, forensics/evidence_package.py,
#        forensics/explainer.py, database/storage.py,
#        reports/pdf_report.py, cli.py additions
############################################################################

=== 4.1 forensics/incident.py ===
@dataclass Incident
  incident_id "SG-YYYYMMDD-NNN" (NNN from the database counter per date)
  created_utc (ISO 8601); scenario_key; config; attack; seed;
  calibration_hash; software_version (VERSION constant); git_commit
  (`git rev-parse HEAD` via subprocess, no shell, timeout; else "unknown")
  detected: detector -> {first_alarm_t, drones, peak_score}
  affected_drones_detected; onset_estimate (earliest CUSUM change point
  among alarmed drones, else earliest alarm)
  fusion (per-drone verdict, belief, plausibility); pattern
  recovery summary (method, status, estimated attack vector)
  simulation_ground_truth {type, targets, onset}, labelled
    "SIMULATION GROUND TRUTH: not available to detectors"
  explanation (from the explainer)
from_run(...) builder; should_open_incident(runs) -> any ALARM.

=== 4.2 forensics/hashing.py ===
sha256_file(path, chunk=65536); sha256_bytes(b);
canonical_json(obj) = json.dumps(sort_keys=True, separators=(",", ":"),
  ensure_ascii=False, default=str).encode("utf-8")
WHY canonical JSON: the same dictionary can serialise in different key
orders, giving different bytes and hashes.

=== 4.3 forensics/signatures.py ===
cryptography ed25519.
ensure_keypair("keys"): generate on first run; private_key.pem (PKCS8;
  BestAvailableEncryption if SWARMGUARD_KEY_PASS is set, else NoEncryption
  with a logged warning); public_key.pem; restrictive permissions where
  supported.
sign(priv, data) -> 64 bytes; verify(pub, data, sig) -> bool (catch
  InvalidSignature); public_key_fingerprint(pub) = first 16 hex of
  sha256(raw public bytes).
WHY a signature on top of hashes: an editor can recompute hashes and edit
the manifest, but cannot sign the edited manifest without the private key.
WHY Ed25519: small keys and signatures, deterministic signing, fast, well
supported.

=== 4.4 forensics/evidence_package.py ===
build_package(incident, sim, runs, fusion_df, figures, out_root=
  "data/packages") -> Path. Folder <incident_id>/:
  telemetry_observed.csv, ground_truth_SIMULATION.csv, detections.csv,
  fusion.csv, config.json, incident.json, report.pdf, figures/*.png,
  manifest.json {package_version, incident_id, created_utc, tool_version,
    public_key_fingerprint, previous_manifest_hash,
    files: [{path, sha256, size_bytes}] sorted by path},
  manifest.sig (signature over canonical manifest bytes), public_key.pem;
  then zip to <incident_id>.zip.
previous_manifest_hash = last entry in custody_log, or "GENESIS".
WHY a chain: each package commits to the previous one; silently deleting
or replacing an earlier package breaks the chain (an append-only audit log).
verify_package(path) -> VerificationReport: each file exists + hash matches
  (PASS/FAIL); unlisted extra files (WARN); signature (PASS/FAIL); chain
  (PASS/FAIL/UNKNOWN outside the original database).
tamper_test(path, target_file, byte_index=None, rng): copy the package to a
  temp directory, XOR one byte with 0x01, verify the copy, return
  before/after reports and the changed byte. The original is never touched.

=== 4.5 forensics/explainer.py ===
explain_incident(incident) -> 5–12 sentences from templates keyed by facts:
  which detectors alarmed, when, on which drones, with numbers ("Detector
  B's NIS peaked at 48.2, far above the 9.21 threshold"); the pattern
  sentence (for A_silent_B_C_alarm: the formation stayed geometrically
  consistent, so relative ranging saw nothing, while motion and anchor
  checks disagreed with GPS, which matches a coordinated whole-swarm
  shift); recovery and estimated attack vector; limitations ("No anomaly
  detected does not prove GPS was genuine"; "The system does not identify
  who caused this").
explain_drone(drone, step_results) -> short reasons for the UI inspector.
Every sentence must come from a field; never invent facts.

=== 4.6 database/storage.py ===
SQLite at data/swarmguard.db; idempotent schema creation:
  runs(id INTEGER PK, created_utc, scenario, seed, config_hash,
       config_json, summary_json)
  incidents(id TEXT PK, run_id, created_utc, pattern, affected, onset,
       incident_json)
  detections_summary(id INTEGER PK, run_id, detector, detected, delay,
       false_alarms, id_precision, id_recall)
  evidence_packages(incident_id TEXT PK, path, zip_path, manifest_sha256,
       created_utc)
  calibrations(config_hash TEXT PK, created_utc, json)
  custody_log(seq INTEGER PK AUTOINCREMENT, incident_id, manifest_sha256,
       previous_sha256, action, created_utc)
Storage class: context-managed connections, parameterised queries only
(WHY: prevents SQL injection), helpers save_run, save_incident,
next_incident_number(date), list_incidents(filters), get_incident,
save_package, last_manifest_hash, log_custody, list_custody.

=== 4.7 reports/pdf_report.py ===
ReportLab platypus, A4, styled to match the app (dark-teal accent on white
paper for printability; Helvetica fallback if fonts are unavailable).
Sections: cover (SWARMGUARD, incident id, time, "SIMULATION REPORT");
executive summary (explainer); configuration table; timeline figure
(matplotlib: per-detector alarm bars per alarmed drone, onset line);
detector results table; fusion verdicts; recovery; evidence file hashes
(monospace, wrapped), public key fingerprint, previous manifest hash, and
the note that the report's own hash is in manifest.json; limitations and
disclaimer. Header/footer with incident id and page numbers.

=== 4.8 cli.py additions ===
  investigate --scenario KEY --seed N   (run, detect, fuse, recover, open
    incident, build package, print paths)
  verify PATH
  tamper-test PATH --file detections.csv

ACCEPTANCE CHECKS
  investigate on rigid_swarm_shift produces a package and PDF
  verify -> all PASS
  tamper-test -> FAIL on the copy, original still PASS
  a second investigation chains to the first (previous_manifest_hash)
############################################################################