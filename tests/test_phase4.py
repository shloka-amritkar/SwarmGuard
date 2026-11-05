"""Phase 4 tests: hashing, signatures, storage, incidents, explainer, evidence packages.

Every test writes to pytest's tmp_path (database, keys, packages), so the
real data/ and keys/ folders are never touched.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from analysis.calibration import load_or_default
from analysis.experiments import PipelineResult, run_pipeline
from attacks.scenarios import get_scenario
from core.palette import PALETTE, hex_to_rgb01
from database.storage import GENESIS, Storage
from forensics.evidence_package import (
    FAIL,
    PASS,
    UNKNOWN,
    WARN,
    build_package,
    detections_table,
    tamper_test,
    verify_package,
)
from forensics.explainer import (
    LIMITATION_NO_ATTRIBUTION,
    LIMITATION_NO_PROOF,
    explain_drone,
    explain_incident,
)
from forensics.hashing import canonical_json, sha256_bytes, sha256_file
from forensics.incident import (
    GROUND_TRUTH_LABEL,
    Incident,
    from_run,
    make_incident_id,
    should_open_incident,
)
from forensics.signatures import (
    ensure_keypair,
    load_private_key,
    load_public_key,
    public_key_fingerprint,
    sign,
    verify,
)

EVAL_SEED = 5000


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shift_pipe() -> PipelineResult:
    """Pipeline on rigid_swarm_shift; recovery every 10th step keeps it fast."""
    sc = get_scenario("rigid_swarm_shift", seed=EVAL_SEED)
    return run_pipeline(sc.sim, sc.attack, load_or_default(sc.sim), recovery_every=10)


@pytest.fixture(scope="module")
def clean_pipe() -> PipelineResult:
    """Pipeline on the clean baseline (no attack)."""
    sc = get_scenario("clean_baseline", seed=EVAL_SEED)
    return run_pipeline(sc.sim, sc.attack, load_or_default(sc.sim), recovery_every=10)


@pytest.fixture(scope="module")
def shift_incident(shift_pipe: PipelineResult) -> Incident:
    """An incident built from the rigid shift run."""
    return from_run(shift_pipe, "rigid_swarm_shift", "SG-20261105-001", "test-calibration")


def _package(tmp_path: Path, pipe: PipelineResult, incident: Incident, storage: Storage | None):
    """Build a package into tmp_path with keys in tmp_path/keys."""
    return build_package(incident, pipe.sim_result, pipe.all_runs(), pipe.fusion.to_dataframe(),
                         out_root=tmp_path / "packages", storage=storage, keys_dir=tmp_path / "keys")


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
def test_palette_values_and_conversion() -> None:
    assert PALETTE.red == "#C8221B"
    assert hex_to_rgb01("#FFFFFF") == (1.0, 1.0, 1.0)
    assert PALETTE.status_colour("compromised") == PALETTE.red
    with pytest.raises(ValueError):
        hex_to_rgb01("red")
    with pytest.raises(Exception):
        PALETTE.red = "#000000"  # type: ignore[misc]  (frozen dataclass)


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------
def test_canonical_json_ignores_key_order() -> None:
    a = {"b": 1, "a": {"y": 2, "x": [1, 2]}}
    b = {"a": {"x": [1, 2], "y": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(a) == b'{"a":{"x":[1,2],"y":2},"b":1}'


def test_sha256_file_matches_bytes(tmp_path: Path) -> None:
    data = bytes(range(256)) * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert sha256_file(path, chunk=1000) == sha256_bytes(data)
    # Known vector: SHA-256 of the empty string.
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------
def test_sign_verify_round_trip_and_tamper(tmp_path: Path) -> None:
    paths = ensure_keypair(tmp_path / "keys")
    priv, pub = load_private_key(paths.private), load_public_key(paths.public)
    msg = b"manifest bytes"
    sig = sign(priv, msg)
    assert len(sig) == 64
    assert verify(pub, msg, sig)
    assert not verify(pub, b"manifest bytez", sig)
    flipped = bytes([sig[0] ^ 0x01]) + sig[1:]
    assert not verify(pub, msg, flipped)
    fp = public_key_fingerprint(pub)
    assert len(fp) == 16 and all(c in "0123456789abcdef" for c in fp)


def test_ensure_keypair_reuses_existing_keys(tmp_path: Path) -> None:
    first = ensure_keypair(tmp_path / "keys")
    pem = first.public.read_bytes()
    second = ensure_keypair(tmp_path / "keys")
    assert second.public.read_bytes() == pem


def test_encrypted_private_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWARMGUARD_KEY_PASS", "correct horse")
    paths = ensure_keypair(tmp_path / "keys")
    assert b"ENCRYPTED" in paths.private.read_bytes()
    load_private_key(paths.private)
    monkeypatch.setenv("SWARMGUARD_KEY_PASS", "wrong")
    with pytest.raises(ValueError):
        load_private_key(paths.private)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def test_storage_schema_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "sg.db"
    Storage(db)
    store = Storage(db)          # second creation must not fail
    store.create_schema()
    assert store.table_names() == sorted(["calibrations", "custody_log", "detections_summary",
                                          "evidence_packages", "incidents", "runs"])


def test_incident_counter_and_injection_safety(tmp_path: Path) -> None:
    store = Storage(tmp_path / "sg.db")
    day = date(2026, 11, 5)
    assert store.next_incident_number(day) == 1
    nasty = "x'; DROP TABLE incidents; --"
    for n in (1, 2):
        store.save_incident({"incident_id": make_incident_id(day, n), "created_utc": f"2026-11-05T0{n}",
                             "pattern": nasty, "affected_drones_detected": [3], "onset_estimate": 20.0,
                             "scenario_key": "single_hijack"})
    assert store.next_incident_number(day) == 3
    assert store.next_incident_number(date(2026, 11, 6)) == 1
    assert "incidents" in store.table_names()          # the table survived
    assert len(store.list_incidents(pattern=nasty)) == 2
    assert len(store.list_incidents(scenario="single_hijack")) == 2
    assert store.list_incidents(scenario="other") == []
    assert store.get_incident("SG-20261105-001")["pattern"] == nasty


def test_custody_chain_helpers(tmp_path: Path) -> None:
    store = Storage(tmp_path / "sg.db")
    assert store.last_manifest_hash() == GENESIS
    store.log_custody("SG-1", "h1", GENESIS, "created")
    store.log_custody("SG-1", "h1", GENESIS, "verified")
    assert store.last_manifest_hash() == "h1"          # "verified" rows do not move the chain
    store.log_custody("SG-2", "h2", "h1", "created")
    assert store.last_manifest_hash() == "h2"
    assert [r["action"] for r in store.list_custody()] == ["created", "verified", "created"]
    assert store.custody_entry("SG-2")["previous_sha256"] == "h1"


def test_make_incident_id_format() -> None:
    assert make_incident_id(date(2026, 1, 2), 7) == "SG-20260102-007"
    with pytest.raises(ValueError):
        make_incident_id(date(2026, 1, 2), 1000)


# --------------------------------------------------------------------------
# Incident and explainer
# --------------------------------------------------------------------------
def test_should_open_incident(shift_pipe: PipelineResult) -> None:
    assert should_open_incident(shift_pipe.runs)


def test_incident_fields_for_rigid_shift(shift_incident: Incident) -> None:
    inc = shift_incident
    assert inc.pattern == "A_silent_B_C_alarm"
    assert inc.detected["A"]["state"] == "silent"
    assert inc.detected["B"]["state"] == "alarm" and inc.detected["B"]["peak_nis"] is not None
    assert inc.onset_estimate is not None
    truth_onset = inc.simulation_ground_truth["onset"]
    assert abs(inc.onset_estimate - truth_onset) < 5.0
    assert inc.simulation_ground_truth["label"] == GROUND_TRUTH_LABEL
    assert inc.recovery["status"] == "ok"
    assert inc.recovery["attack_magnitude"] == pytest.approx(10.0, abs=1.5)
    # JSON round trip
    again = Incident.from_dict(json.loads(json.dumps(inc.to_dict())))
    assert again == inc


def test_explanation_is_honest_and_sized(shift_incident: Incident) -> None:
    sentences = explain_incident(shift_incident)
    assert 5 <= len(sentences) <= 12
    text = " ".join(sentences).lower()
    assert LIMITATION_NO_PROOF in sentences and LIMITATION_NO_ATTRIBUTION in sentences
    assert "gps genuine" not in text.replace("gps was genuine", "")
    assert "safe" not in text
    assert "B and C" in sentences[0]
    # The NIS number quoted comes from the incident field.
    assert f"{shift_incident.detected['B']['peak_nis']:.1f}" in text


def test_explanation_for_clean_run(clean_pipe: PipelineResult) -> None:
    inc = from_run(clean_pipe, "clean_baseline", "SG-20261105-009", "none")
    sentences = explain_incident(inc)
    assert 5 <= len(sentences) <= 12
    if not inc.alarmed_detectors():
        assert "no detector raised an alarm" in sentences[0]
        assert inc.onset_estimate is None


def test_quiet_sentence_never_contradicts_numbers(shift_incident: Incident) -> None:
    """A silent detector whose peak crossed the threshold must not be called "below" it."""
    data = shift_incident.to_dict()
    data["detected"]["A"] = {**data["detected"]["A"], "state": "silent", "peak_score": 0.8,
                             "threshold": 0.5}
    sentence = [s for s in explain_incident(Incident.from_dict(data)) if s.startswith("Detector A")][0]
    assert "below" not in sentence and "persistence" in sentence


def test_c_silent_pattern_states_only_what_is_known() -> None:
    """C's silence has two possible causes; the description must not pick one."""
    from detection.fusion import describe_pattern
    text = describe_pattern("C_silent_A_B_alarm")
    assert "may carry no anchor" in text and "never persisted" in text


def test_explain_drone_reasons(shift_pipe: PipelineResult) -> None:
    k = len(shift_pipe.runs[0].steps) - 1
    steps = {r.short: r.steps[k] for r in shift_pipe.runs}
    reasons = explain_drone(0, steps, shift_pipe.det_cfg)
    assert len(reasons) == 3
    assert reasons[1].startswith("B:") and "NIS" in reasons[1]
    with pytest.raises(ValueError):
        explain_drone(99, steps)


# --------------------------------------------------------------------------
# Evidence packages
# --------------------------------------------------------------------------
def test_detections_table_has_all_detectors(shift_pipe: PipelineResult) -> None:
    table = detections_table(shift_pipe.all_runs())
    assert set(table["detector"]) == {"A", "B", "C", "F"}
    assert table.columns[0] == "detector"


def test_package_verifies_tamper_fails_and_chains(tmp_path: Path, shift_pipe: PipelineResult,
                                                 shift_incident: Incident) -> None:
    store = Storage(tmp_path / "sg.db")
    trusted = tmp_path / "keys" / "public_key.pem"
    first = _package(tmp_path, shift_pipe, shift_incident, store)
    assert first.previous_manifest_hash == GENESIS
    expected = {"telemetry_observed.csv", "ground_truth_SIMULATION.csv", "detections.csv", "fusion.csv",
                "config.json", "incident.json", "report.pdf", "manifest.json", "manifest.sig",
                "public_key.pem", "figures"}
    assert expected <= {p.name for p in first.folder.iterdir()}
    assert first.report.read_bytes().startswith(b"%PDF")

    for target in (first.folder, first.zip_path):
        report = verify_package(target, storage=store, trusted_key=trusted)
        assert report.overall == PASS
        assert report.count(FAIL) == 0 and report.count(WARN) == 0 and report.count(UNKNOWN) == 0

    original_bytes = (first.folder / "detections.csv").read_bytes()
    result = tamper_test(first.folder, "detections.csv", byte_index=10, storage=store, trusted_key=trusted)
    assert result.before.overall == PASS and result.after.overall == FAIL
    assert result.new_byte == result.old_byte ^ 0x01
    assert (first.folder / "detections.csv").read_bytes() == original_bytes     # original untouched
    assert verify_package(first.folder, storage=store, trusted_key=trusted).overall == PASS

    second_inc = Incident.from_dict({**shift_incident.to_dict(), "incident_id": "SG-20261105-002"})
    second = _package(tmp_path, shift_pipe, second_inc, store)
    manifest = json.loads((second.folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["previous_manifest_hash"] == first.manifest_sha256
    chain = [c for c in verify_package(second.zip_path, storage=store, trusted_key=trusted).checks
             if c.name == "chain"][0]
    assert chain.status == PASS


def test_verification_edge_cases(tmp_path: Path, shift_pipe: PipelineResult,
                                 shift_incident: Incident) -> None:
    paths = _package(tmp_path, shift_pipe, shift_incident, storage=None)
    # Outside the original database the chain is UNKNOWN, not FAIL.
    report = verify_package(paths.folder, storage=None, trusted_key=None)
    statuses = {c.name: c.status for c in report.checks}
    assert statuses["chain"] == UNKNOWN and statuses["signer key"] == UNKNOWN
    assert report.overall == PASS
    # An extra unlisted file is a WARN; a deleted file is a FAIL.
    copy = tmp_path / "copy"
    shutil.copytree(paths.folder, copy)
    (copy / "notes.txt").write_text("added later", encoding="utf-8")
    assert verify_package(copy, trusted_key=None).count(WARN) == 1
    (copy / "fusion.csv").unlink()
    assert verify_package(copy, trusted_key=None).overall == FAIL
    # A package signed by a different key fails the trusted-key check.
    other = ensure_keypair(tmp_path / "other_keys")
    statuses = {c.name: c.status for c in verify_package(paths.folder, trusted_key=other.public).checks}
    assert statuses["signer key"] == FAIL
    # Tampering with the manifest itself breaks the signature.
    result = tamper_test(paths.folder, "manifest.json", rng=np.random.default_rng(1), trusted_key=None)
    assert result.after.overall == FAIL
    # Evidence is never overwritten.
    with pytest.raises(ValueError):
        _package(tmp_path, shift_pipe, shift_incident, storage=None)
