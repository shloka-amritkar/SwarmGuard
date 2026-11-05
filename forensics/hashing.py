"""SHA-256 hashing helpers for evidence files and JSON documents.

A cryptographic hash turns any input into a fixed 64-hex-character
fingerprint. Changing even one bit of the input changes, on average, half of
the output bits, and nobody knows how to find two different inputs with the
same SHA-256. So "the hash still matches" is strong evidence that a file is
byte-for-byte unchanged.

A hash alone does NOT prove who made a file: anyone can recompute it after
editing. That is why the manifest of hashes is also signed (see
forensics/signatures.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CHUNK = 65536  # 64 KiB


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest (64 lowercase hex characters) of data."""
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(f"sha256_bytes expects bytes, got {type(data).__name__}")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = DEFAULT_CHUNK) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks.

    WHY chunks: a telemetry CSV can be many megabytes. Reading 64 KiB at a
    time and feeding each piece to the hash keeps memory use small; the
    result is identical to hashing the whole file at once.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Serialise obj to ONE fixed byte form, so its hash is reproducible.

        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, default=str).encode("utf-8")

    WHY canonical JSON: the same dictionary can be written with keys in a
    different order or with different spacing, giving different bytes and
    therefore different hashes and signatures. Sorting keys and removing
    optional spaces gives every equal dictionary exactly the same bytes.
    default=str turns values JSON cannot store (e.g. a datetime) into text
    instead of crashing.
    """
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return text.encode("utf-8")
