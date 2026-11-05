"""Digital forensics: incidents, hashing, signatures, evidence packages, explanations."""

from __future__ import annotations

from forensics.evidence_package import (
    Check,
    PackagePaths,
    TamperResult,
    VerificationReport,
    build_package,
    tamper_test,
    verify_package,
)
from forensics.explainer import explain_drone, explain_incident
from forensics.hashing import canonical_json, sha256_bytes, sha256_file
from forensics.incident import VERSION, Incident, from_run, should_open_incident
from forensics.signatures import ensure_keypair, public_key_fingerprint, sign, verify

__all__ = [
    "VERSION",
    "Check",
    "Incident",
    "PackagePaths",
    "TamperResult",
    "VerificationReport",
    "build_package",
    "canonical_json",
    "ensure_keypair",
    "explain_drone",
    "explain_incident",
    "from_run",
    "public_key_fingerprint",
    "sha256_bytes",
    "sha256_file",
    "should_open_incident",
    "sign",
    "tamper_test",
    "verify",
    "verify_package",
]
