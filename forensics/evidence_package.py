"""Tamper-evident evidence packages: build, verify, and tamper-test.

PACKAGE LAYOUT  data/packages/<incident_id>/ (also zipped to <incident_id>.zip)
    telemetry_observed.csv        what the drones measured (detector input)
    ground_truth_SIMULATION.csv   hidden simulation truth, for evaluation only
    detections.csv                every detector's per-step output (A, B, C, F)
    fusion.csv                    fused masses and verdicts
    config.json, incident.json    settings and the incident record
    figures/*.png                 timeline and any extra figures
    report.pdf                    the human-readable report
    public_key.pem                key to check the signature with
    manifest.json                 SHA-256 + size of every file above
    manifest.sig                  Ed25519 signature over the canonical manifest

THREE LAYERS OF PROTECTION
1. Hashes: change one byte of a file and its SHA-256 no longer matches.
2. Signature: whoever edits a file and fixes its hash in the manifest still
   cannot re-sign the manifest without the private key.
3. Chain: manifest.previous_manifest_hash = manifest hash of the package
   created before it ("GENESIS" for the first). The custody log records
   every link, so silently deleting or replacing an earlier package breaks
   the chain (an append-only audit log).
   manifest hash = sha256(canonical_json(manifest)), the same bytes that are
   signed. The file on disk is indented for humans; verification parses it
   and re-canonicalises before hashing.

WHAT VERIFICATION CANNOT TELL YOU (R4): a PASS means "unchanged since
signing", not "the telemetry was genuine".
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from analysis.experiments import PipelineResult
from core.schema import SimResult
from database.storage import GENESIS, Storage, utc_now
from detection.base import DetectionRun
from forensics.hashing import canonical_json, sha256_bytes, sha256_file
from forensics.incident import VERSION, Incident
from forensics.signatures import (
    ensure_keypair,
    load_private_key,
    public_key_fingerprint,
    public_key_from_pem,
    public_key_pem,
    sign,
    verify,
)
from reports.pdf_report import build_pdf_report, timeline_figure, trajectory_figure

logger = logging.getLogger(__name__)

PACKAGE_VERSION = "1.0"
MANIFEST = "manifest.json"
SIGNATURE = "manifest.sig"
PUBLIC_KEY = "public_key.pem"
REPORT = "report.pdf"
NOT_IN_MANIFEST = {MANIFEST, SIGNATURE}
PASS, FAIL, WARN, UNKNOWN = "PASS", "FAIL", "WARN", "UNKNOWN"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class Check:
    """One line of a verification report."""

    name: str
    status: str
    detail: str = ""


@dataclass
class VerificationReport:
    """All checks on one package. overall = FAIL if any check failed, else PASS."""

    package: str
    incident_id: str | None
    manifest_sha256: str | None
    checks: list[Check] = field(default_factory=list)

    @property
    def overall(self) -> str:
        """FAIL if any check is FAIL, otherwise PASS (WARN/UNKNOWN are reported, not fatal)."""
        return FAIL if any(c.status == FAIL for c in self.checks) else PASS

    def count(self, status: str) -> int:
        """How many checks have this status."""
        return sum(c.status == status for c in self.checks)

    def to_dataframe(self) -> pd.DataFrame:
        """Table of checks (name, status, detail) for display."""
        return pd.DataFrame([vars(c) for c in self.checks], columns=["name", "status", "detail"])


@dataclass
class TamperResult:
    """Outcome of tamper_test: the copy before and after flipping one bit."""

    target_file: str
    byte_index: int
    old_byte: int
    new_byte: int
    before: VerificationReport
    after: VerificationReport


@dataclass
class PackagePaths:
    """Where build_package wrote things."""

    folder: Path
    zip_path: Path
    report: Path
    manifest_sha256: str
    previous_manifest_hash: str


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------
def detections_table(runs: Sequence[DetectionRun]) -> pd.DataFrame:
    """All detector runs stacked into one long table with a `detector` column."""
    frames = [r.to_dataframe().assign(detector=r.short) for r in runs]
    table = pd.concat(frames, ignore_index=True)
    return table[["detector", *[c for c in table.columns if c != "detector"]]]


def _write_data_files(folder: Path, incident: Incident, sim: SimResult, runs: Sequence[DetectionRun],
                      fusion_df: pd.DataFrame, figures: dict[str, Figure]) -> None:
    """Write CSV, JSON and figure files (everything except report and manifest).

    Figures are closed after saving, so callers should not reuse them.
    """
    sim.to_observed_dataframe().to_csv(folder / "telemetry_observed.csv", index=False)
    sim.to_truth_dataframe().to_csv(folder / "ground_truth_SIMULATION.csv", index=False)
    detections_table(runs).to_csv(folder / "detections.csv", index=False)
    fusion_df.to_csv(folder / "fusion.csv", index=False)
    config = {"scenario_key": incident.scenario_key, "seed": incident.seed,
              "calibration_hash": incident.calibration_hash, **incident.config,
              "attack_SIMULATION": incident.attack}
    (folder / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str),
                                        encoding="utf-8")
    (folder / "incident.json").write_text(json.dumps(incident.to_dict(), indent=2, sort_keys=True,
                                                     default=str), encoding="utf-8")
    fig_dir = folder / "figures"
    fig_dir.mkdir(exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)  # WHY: free the figure's memory once it is on disk


def file_entries(folder: Path, exclude: set[str] = NOT_IN_MANIFEST) -> list[dict[str, Any]]:
    """[{path, sha256, size_bytes}] for every file under folder, sorted by path.

    Paths use forward slashes (e.g. "figures/timeline.png") so a manifest
    written on Windows verifies identically on Linux.
    """
    entries = []
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder).as_posix()
        if p.is_file() and rel not in exclude:
            entries.append({"path": rel, "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    return sorted(entries, key=lambda e: e["path"])


def _zip_folder(folder: Path) -> Path:
    """Zip folder to <folder>.zip, entries stored as <incident_id>/<file>."""
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=f"{folder.name}/{p.relative_to(folder).as_posix()}")
    return zip_path


def build_package(
    incident: Incident,
    sim: SimResult,
    runs: Sequence[DetectionRun],
    fusion_df: pd.DataFrame,
    figures: dict[str, Figure] | None = None,
    out_root: str | Path = "data/packages",
    storage: Storage | None = None,
    keys_dir: str | Path = "keys",
) -> PackagePaths:
    """Write, sign, zip and log the evidence package of one incident.

    Order matters:
      1. data files + figures (a timeline figure is always added)
      2. public_key.pem
      3. report.pdf, which lists the hashes of steps 1-2
      4. manifest.json over every file, then manifest.sig over its canonical bytes
      5. zip, then custody log entry "created" and evidence_packages row
    runs should include the fused run "F" (PipelineResult.all_runs()).
    """
    folder = Path(out_root) / incident.incident_id
    if folder.exists():
        raise ValueError(f"package folder {folder} already exists; evidence is never overwritten")
    folder.mkdir(parents=True)
    keys = ensure_keypair(keys_dir)
    private_key = load_private_key(keys.private)
    fingerprint = public_key_fingerprint(private_key.public_key())
    previous = storage.last_manifest_hash() if storage is not None else GENESIS

    all_figures = {"timeline": timeline_figure(runs, incident.onset_estimate), **(figures or {})}
    _write_data_files(folder, incident, sim, runs, fusion_df, all_figures)
    (folder / PUBLIC_KEY).write_bytes(public_key_pem(private_key.public_key()))
    report = build_pdf_report(folder / REPORT, incident.to_dict(), file_entries(folder), fingerprint,
                              previous, timeline_png=folder / "figures" / "timeline.png")

    manifest = {"package_version": PACKAGE_VERSION, "incident_id": incident.incident_id,
                "created_utc": utc_now(), "tool_version": VERSION,
                "public_key_fingerprint": fingerprint, "previous_manifest_hash": previous,
                "files": file_entries(folder)}
    canonical = canonical_json(manifest)
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (folder / SIGNATURE).write_bytes(sign(private_key, canonical))
    manifest_hash = sha256_bytes(canonical)

    zip_path = _zip_folder(folder)
    if storage is not None:
        storage.log_custody(incident.incident_id, manifest_hash, previous, "created")
        storage.save_package(incident.incident_id, folder, zip_path, manifest_hash)
    logger.info("built evidence package %s (manifest %s)", folder, manifest_hash[:16])
    return PackagePaths(folder, zip_path, report, manifest_hash, previous)


def package_incident(
    incident: Incident,
    pipe: PipelineResult,
    storage: Storage,
    out_root: str | Path = "data/packages",
    keys_dir: str | Path = "keys",
) -> PackagePaths:
    """Build the evidence package of an incident from its pipeline run.

    Adds the flight-path figure, and passes every detector run plus the
    fused run "F". Used by both the CLI (investigate) and the app (Evidence
    Centre), so both produce identical packages.
    """
    return build_package(incident, pipe.sim_result, pipe.all_runs(), pipe.fusion.to_dataframe(),
                         {"trajectories": trajectory_figure(pipe.sim_result)},
                         out_root=out_root, storage=storage, keys_dir=keys_dir)


# --------------------------------------------------------------------------
# Verifying
# --------------------------------------------------------------------------
def _safe_extract(zip_path: Path, dest: Path) -> Path:
    """Extract a zip into dest and return the folder holding manifest.json.

    WHY check every member path: a malicious zip can contain names like
    "../../evil.py" ("zip slip") that would write outside dest. We refuse them.
    """
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe path in zip: {member!r}")
        zf.extractall(dest)
    manifests = sorted(dest.rglob(MANIFEST))
    if not manifests:
        raise ValueError(f"{zip_path} contains no {MANIFEST}")
    return manifests[0].parent


def _check_files(folder: Path, manifest: dict[str, Any]) -> list[Check]:
    """Existence + hash of every listed file, and a WARN for unlisted extras."""
    checks, listed = [], set()
    for entry in manifest.get("files", []):
        rel = entry["path"]
        listed.add(rel)
        path = folder / rel
        if not path.is_file():
            checks.append(Check(f"file {rel}", FAIL, "missing"))
            continue
        actual = sha256_file(path)
        ok = actual == entry["sha256"]
        checks.append(Check(f"file {rel}", PASS if ok else FAIL,
                            "hash matches" if ok else f"hash {actual[:16]}... != listed {entry['sha256'][:16]}..."))
    for extra in file_entries(folder):
        if extra["path"] not in listed:
            checks.append(Check(f"file {extra['path']}", WARN, "not listed in the manifest"))
    return checks


def _check_signature(folder: Path, manifest: dict[str, Any], trusted_key: Path | None) -> list[Check]:
    """Signature over the canonical manifest, and whether the key is the trusted one."""
    sig_path, key_path = folder / SIGNATURE, folder / PUBLIC_KEY
    if not sig_path.is_file() or not key_path.is_file():
        return [Check("signature", FAIL, "manifest.sig or public_key.pem missing")]
    try:
        key = public_key_from_pem(key_path.read_bytes())
    except ValueError as exc:
        return [Check("signature", FAIL, str(exc))]
    ok = verify(key, canonical_json(manifest), sig_path.read_bytes())
    fingerprint = public_key_fingerprint(key)
    checks = [Check("signature", PASS if ok else FAIL,
                    "valid Ed25519 signature over the manifest" if ok else "signature does not match")]
    if manifest.get("public_key_fingerprint") != fingerprint:
        checks.append(Check("key fingerprint", FAIL, "manifest fingerprint differs from public_key.pem"))
    # WHY a trusted key: someone could replace the whole package, including
    # its public key, and re-sign with their own key. Comparing with the key
    # this installation trusts (or the fingerprint printed on the report)
    # catches that.
    if trusted_key is None or not trusted_key.is_file():
        checks.append(Check("signer key", UNKNOWN, f"fingerprint {fingerprint}; no trusted key to compare"))
    else:
        same = public_key_fingerprint(public_key_from_pem(trusted_key.read_bytes())) == fingerprint
        checks.append(Check("signer key", PASS if same else FAIL,
                            f"fingerprint {fingerprint} " + ("matches" if same else "is NOT") + " the trusted key"))
    return checks


def _check_chain(incident_id: str, manifest_hash: str, previous: str, storage: Storage | None) -> Check:
    """Chain link: recorded at creation, and previous = the package created just before."""
    entry = storage.custody_entry(incident_id) if storage is not None else None
    if entry is None:
        return Check("chain", UNKNOWN, "package not in this database; chain cannot be checked")
    if entry["manifest_sha256"] != manifest_hash:
        return Check("chain", FAIL, "manifest differs from the one recorded at creation")
    if entry["previous_sha256"] != previous:
        return Check("chain", FAIL, "previous_manifest_hash differs from the custody log")
    earlier = storage.created_before(entry["seq"])
    expected = earlier[-1]["manifest_sha256"] if earlier else GENESIS
    if previous != expected:
        return Check("chain", FAIL, "link does not point to the package created before it")
    return Check("chain", PASS, f"links to {previous[:16]}{'...' if previous != GENESIS else ''}")


def verify_folder(folder: Path, storage: Storage | None = None,
                  trusted_key: Path | None = None, label: str | None = None) -> VerificationReport:
    """Verify an unpacked package folder (see verify_package)."""
    try:
        manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not a JSON object")
    except (OSError, ValueError) as exc:  # json.JSONDecodeError and UnicodeDecodeError are ValueErrors
        return VerificationReport(label or str(folder), None, None, [Check("manifest", FAIL, f"unreadable: {exc}")])
    manifest_hash = sha256_bytes(canonical_json(manifest))
    report = VerificationReport(label or str(folder), manifest.get("incident_id"), manifest_hash)
    report.checks.append(Check("manifest", PASS, f"sha256 {manifest_hash[:16]}..."))
    report.checks += _check_files(folder, manifest)
    report.checks += _check_signature(folder, manifest, trusted_key)
    report.checks.append(_check_chain(str(manifest.get("incident_id")), manifest_hash,
                                      str(manifest.get("previous_manifest_hash")), storage))
    return report


def verify_package(path: str | Path, storage: Storage | None = None,
                   trusted_key: str | Path | None = Path("keys") / PUBLIC_KEY) -> VerificationReport:
    """Verify a package folder or .zip.

    Checks: every listed file exists and its hash matches (PASS/FAIL);
    unlisted extra files (WARN); signature (PASS/FAIL); signer key vs the
    trusted key (PASS/FAIL/UNKNOWN); chain against the custody log
    (PASS/FAIL, UNKNOWN outside the original database).
    """
    src = Path(path)
    trusted = Path(trusted_key) if trusted_key is not None else None
    if src.is_dir():
        return verify_folder(src, storage, trusted, label=str(src))
    if src.is_file() and zipfile.is_zipfile(src):
        with tempfile.TemporaryDirectory() as tmp:
            return verify_folder(_safe_extract(src, Path(tmp)), storage, trusted, label=str(src))
    raise ValueError(f"{src} is neither a package folder nor a zip file")


# --------------------------------------------------------------------------
# Tamper test (always on a copy)
# --------------------------------------------------------------------------
def tamper_test(
    path: str | Path,
    target_file: str,
    byte_index: int | None = None,
    rng: np.random.Generator | None = None,
    storage: Storage | None = None,
    trusted_key: str | Path | None = Path("keys") / PUBLIC_KEY,
) -> TamperResult:
    """Copy the package to a temp folder, flip one bit of one file, verify the copy.

    new_byte = old_byte XOR 0x01 (the smallest possible change: one bit).
    byte_index None -> chosen with rng (default np.random.default_rng(0), R2).
    The original package is only read, never written.
    """
    src = Path(path)
    trusted = Path(trusted_key) if trusted_key is not None else None
    rng = rng if rng is not None else np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        if src.is_dir():
            copy = Path(tmp) / src.name
            shutil.copytree(src, copy)
        elif src.is_file() and zipfile.is_zipfile(src):
            copy = _safe_extract(src, Path(tmp))
        else:
            raise ValueError(f"{src} is neither a package folder nor a zip file")
        before = verify_folder(copy, storage, trusted, label=f"copy of {src}")
        target = copy / target_file
        if not target.is_file():
            raise ValueError(f"{target_file!r} is not a file in the package")
        data = bytearray(target.read_bytes())
        if not data:
            raise ValueError(f"{target_file!r} is empty; nothing to tamper with")
        index = int(rng.integers(len(data))) if byte_index is None else int(byte_index)
        if not (0 <= index < len(data)):
            raise ValueError(f"byte_index must be in [0, {len(data)}), got {index}")
        old = data[index]
        data[index] = old ^ 0x01
        target.write_bytes(bytes(data))
        after = verify_folder(copy, storage, trusted, label=f"tampered copy of {src}")
    return TamperResult(target_file, index, old, data[index], before, after)
