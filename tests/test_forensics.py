"""Evidence integrity: hashes, Ed25519 signatures, the tamper test and the custody chain.

Every file is written under pytest's tmp_path, never into data/ or keys/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.experiments import PipelineResult, run_pipeline
from attacks.scenarios import get_scenario
from config import DetectorConfig
from database.storage import GENESIS, Storage
from forensics.evidence_package import FAIL, PASS, build_package, tamper_test, verify_package
from forensics.hashing import sha256_bytes, sha256_file
from forensics.incident import Incident, from_run


@pytest.fixture(scope="module")
def pipe() -> PipelineResult:
    sc = get_scenario("single_hijack", seed=5000)
    sc.sim.duration = 30.0
    return run_pipeline(sc.sim, sc.attack, DetectorConfig(), recovery_every=10)


@pytest.fixture(scope="module")
def incident(pipe: PipelineResult) -> Incident:
    return from_run(pipe, "single_hijack", "SG-20261105-001", "test")


def _build(tmp_path: Path, pipe: PipelineResult, incident: Incident, store: Storage):
    return build_package(incident, pipe.sim_result, pipe.all_runs(), pipe.fusion.to_dataframe(),
                         out_root=tmp_path / "packages", storage=store, keys_dir=tmp_path / "keys")


def test_hash_changes_after_one_byte_flip(tmp_path: Path) -> None:
    data = bytearray(b"drone telemetry " * 100)
    path = tmp_path / "file.bin"
    path.write_bytes(bytes(data))
    before = sha256_file(path)
    assert before == sha256_bytes(bytes(data))
    data[37] ^= 0x01
    path.write_bytes(bytes(data))
    assert sha256_file(path) != before


def test_signature_fails_after_manifest_edit(tmp_path: Path, pipe, incident) -> None:
    store = Storage(tmp_path / "sg.db")
    paths = _build(tmp_path, pipe, incident, store)
    trusted = tmp_path / "keys" / "public_key.pem"
    assert verify_package(paths.folder, storage=store, trusted_key=trusted).overall == PASS
    manifest = paths.folder / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["incident_id"] = "SG-20261105-999"            # an "innocent" edit
    manifest.write_text(json.dumps(data), encoding="utf-8")
    report = verify_package(paths.folder, storage=store, trusted_key=trusted)
    assert report.overall == FAIL
    assert any(c.status == FAIL and "signature" in c.name for c in report.checks)


def test_tamper_test_leaves_original_intact(tmp_path: Path, pipe, incident) -> None:
    store = Storage(tmp_path / "sg.db")
    paths = _build(tmp_path, pipe, incident, store)
    trusted = tmp_path / "keys" / "public_key.pem"
    original = {p.name: p.read_bytes() for p in paths.folder.iterdir() if p.is_file()}
    result = tamper_test(paths.folder, "detections.csv", byte_index=10, storage=store, trusted_key=trusted)
    assert result.before.overall == PASS and result.after.overall == FAIL
    assert {p.name: p.read_bytes() for p in paths.folder.iterdir() if p.is_file()} == original
    assert verify_package(paths.folder, storage=store, trusted_key=trusted).overall == PASS


def test_custody_chain_links_packages(tmp_path: Path, pipe, incident) -> None:
    store = Storage(tmp_path / "sg.db")
    first = _build(tmp_path, pipe, incident, store)
    assert first.previous_manifest_hash == GENESIS
    second_incident = Incident.from_dict({**incident.to_dict(), "incident_id": "SG-20261105-002"})
    second = _build(tmp_path, pipe, second_incident, store)
    manifest = json.loads((second.folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["previous_manifest_hash"] == first.manifest_sha256
    assert store.last_manifest_hash() == second.manifest_sha256
    chain = [c for c in verify_package(second.folder, storage=store).checks if c.name == "chain"]
    assert chain and chain[0].status == PASS
