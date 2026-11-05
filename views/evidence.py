"""Evidence Centre: build, download, verify and tamper-test signed evidence packages.

The checks are the Phase 4 ones (forensics/evidence_package.py):
  every file's SHA-256 matches the manifest, the Ed25519 signature over the
  manifest is valid, the signer is the trusted key, and the manifest links
  to the previous package in the custody log (hash chain).
The tamper test flips ONE bit of one file in a temporary COPY of the
package; the original is only read, never written.

What a PASS means (rule R4): the evidence is unchanged since it was signed.
It does NOT prove the telemetry was genuine, and it does not identify anyone.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from forensics.evidence_package import (PUBLIC_KEY, TamperResult, VerificationReport, package_incident,
                                        tamper_test, verify_package)
from forensics.incident import Incident
from forensics.signatures import load_public_key, public_key_fingerprint
from ui import components as c
from ui.state import QUICK_START, get_storage, keys_dir, packages_dir, pipeline_for_incident, run_scenario

SELECT_KEY = "evidence-select"
BYTES_AROUND = 8   # bytes shown either side of the flipped byte


def pick_incident(ids: list[str]) -> str:
    """Incident selector (accepts a one-time hand-over from the Incidents page)."""
    handoff = st.session_state.pop(SELECT_KEY, None)
    if handoff in ids:
        st.session_state["ev-pick"] = handoff
    return st.selectbox("Incident", ids, key="ev-pick")


def ensure_package(incident_id: str) -> dict[str, Any] | None:
    """The package row, or a Build button when the package does not exist yet."""
    storage = get_storage()
    row = storage.get_package(incident_id)
    if row is not None and Path(row["zip_path"]).exists():
        return row
    st.info("No evidence package has been built for this incident yet.")
    if st.button("Build evidence package", type="primary", key="ev-build"):
        incident = storage.get_incident(incident_id)
        with st.spinner("Re-running the stored configs, hashing, signing and writing the PDF..."):
            package_incident(Incident.from_dict(incident), pipeline_for_incident(incident), storage,
                             out_root=packages_dir(), keys_dir=keys_dir())
        st.rerun()
    return None


def downloads(row: dict[str, Any]) -> None:
    """Download the zip and the PDF report."""
    zip_path, report = Path(row["zip_path"]), Path(row["path"]) / "report.pdf"
    a, b, _ = st.columns([1, 1, 1.5])
    a.download_button("Download package (.zip)", zip_path.read_bytes(), file_name=zip_path.name,
                      mime="application/zip", key="ev-dl-zip", width="stretch")
    if report.exists():
        b.download_button("Download PDF report", report.read_bytes(), file_name=f"{row['incident_id']}.pdf",
                          mime="application/pdf", key="ev-dl-pdf", width="stretch")
    c.html_block(f'<p class="sg-hash">manifest sha256 {html.escape(row["manifest_sha256"])}</p>')


def verify_section(incident_id: str, row: dict[str, Any]) -> None:
    """VERIFY EVIDENCE: run every check, log it in the custody chain, show the table."""
    c.section("Verify evidence", "Re-hashes every file and checks the signature and the hash chain.",
              eyebrow="Integrity")
    store = st.session_state.setdefault("ev-verify", {})
    if st.button("Verify evidence", type="primary", key="ev-verify-btn"):
        storage = get_storage()
        report = verify_package(row["zip_path"], storage=storage, trusted_key=keys_dir() / PUBLIC_KEY)
        entry = storage.custody_entry(incident_id)
        if entry is not None and report.manifest_sha256:
            storage.log_custody(incident_id, report.manifest_sha256, entry["previous_sha256"],
                                "verified" if report.overall == "PASS" else "verify_failed")
        store[incident_id] = report
    report: VerificationReport | None = store.get(incident_id)
    if report is None:
        return
    c.verification_table(report.checks, key="ev-verify")
    c.status_pill(report.overall.lower(), f"Overall {report.overall}")


def byte_context(folder: Path, result: TamperResult) -> str:
    """Hex bytes around the flipped byte, with the changed byte highlighted (old -> new)."""
    data = (folder / result.target_file).read_bytes()
    lo, hi = max(0, result.byte_index - BYTES_AROUND), min(len(data), result.byte_index + BYTES_AROUND + 1)
    parts = []
    for i in range(lo, hi):
        if i == result.byte_index:
            parts.append(f'<span class="sg-byte">{result.old_byte:02x} &rarr; {result.new_byte:02x}</span>')
        else:
            parts.append(f"{data[i]:02x}")
    return (f'<p class="sg-hash">{html.escape(result.target_file)} · byte {result.byte_index} (one bit '
            f'flipped)</p><p class="sg-mono">{" ".join(parts)}</p>')


def tamper_section(incident_id: str, row: dict[str, Any]) -> None:
    """Flip one bit in a COPY of the package, then show both verification results."""
    c.section("Tamper test", "Changes one bit of one file in a temporary copy. The original package is "
              "never modified.", eyebrow="Destructive test on a copy")
    folder = Path(row["path"])
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    target = st.selectbox("File to tamper with", files, key="ev-tamper-file",
                          index=files.index("detections.csv") if "detections.csv" in files else 0)
    store = st.session_state.setdefault("ev-tamper", {})
    if c.danger_button("Run tamper test", key="ev-tamper-btn"):
        store[incident_id] = tamper_test(folder, target, storage=get_storage(),
                                         trusted_key=keys_dir() / PUBLIC_KEY)
    result: TamperResult | None = store.get(incident_id)
    if result is None:
        return
    c.html_block(byte_context(folder, result))
    before, after = st.columns(2)
    with before:
        c.verification_table(result.before.checks, key="ev-before")
        c.status_pill(result.before.overall.lower(), f"Copy before: {result.before.overall}")
    with after:
        c.verification_table(result.after.checks, key="ev-after")
        c.status_pill(result.after.overall.lower(), f"Copy after: {result.after.overall}")
    if result.before.overall == "PASS" and result.after.overall == "FAIL":
        st.success("Tamper detected: one flipped bit is enough to break the hash.")


def custody_and_key(incident_id: str) -> None:
    """Custody log of this incident, and the signing key's fingerprint and PEM."""
    c.section("Chain of custody", "Append-only log; each package links to the one before it.")
    rows = get_storage().list_custody(incident_id)
    df = pd.DataFrame(rows, columns=["seq", "action", "created_utc", "manifest_sha256", "previous_sha256"])
    st.dataframe(df, hide_index=True, width="stretch")
    public = keys_dir() / PUBLIC_KEY
    if public.exists():
        key = load_public_key(public)
        c.html_block(f'<p class="sg-hash">Signing key fingerprint {public_key_fingerprint(key)}</p>')
        with st.expander("Public key (PEM)"):
            st.code(public.read_text(encoding="utf-8"), language=None)
    st.info("A valid signature proves the evidence is unchanged since it was signed. It does not prove "
            "the telemetry was genuine, and it does not identify an attacker. The signing key is stored "
            "unencrypted on disk; production use would require a protected key.")


def main() -> None:
    """Draw the page."""
    c.page_header("Evidence", "centre", eyebrow="Forensics",
                  caption="Signed, hash-chained evidence packages you can verify. Simulation results.")
    ids = [r["id"] for r in get_storage().list_incidents()]
    if not ids:
        if c.empty_state("No incidents yet, so there is no evidence to package. The quick start "
                         "scenario opens one.", "Run quick start", sticker_name="fingerprint", key="evidence"):
            with st.spinner("Simulating, detecting, fusing and recovering..."):
                run_scenario(QUICK_START)
            st.rerun()
        return
    incident_id = pick_incident(ids)
    row = ensure_package(incident_id)
    if row is None:
        return
    downloads(row)
    verify_section(incident_id, row)
    tamper_section(incident_id, row)
    custody_and_key(incident_id)


main()
