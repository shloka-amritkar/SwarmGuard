"""Ed25519 digital signatures for evidence manifests.

WHAT A SIGNATURE ADDS ON TOP OF HASHES
The manifest lists the SHA-256 hash of every evidence file. Someone who
edits a file could simply recompute its hash and edit the manifest too.
They cannot, however, produce a valid signature over the edited manifest
without the PRIVATE key. Anyone holding the PUBLIC key can check the
signature, so verification needs no secret.

    sign:    sig = Sign(private_key, manifest_bytes)      (64 bytes)
    verify:  Verify(public_key, manifest_bytes, sig) -> True / False

WHY Ed25519: 32-byte keys and 64-byte signatures, deterministic signing (the
same message always gives the same signature, so no random number can be
reused by mistake), fast, and supported by the standard `cryptography`
library.

WHAT A VALID SIGNATURE DOES NOT PROVE (rule R4): it shows the evidence is
unchanged since it was signed. It does not show the telemetry was genuine.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

logger = logging.getLogger(__name__)

PRIVATE_KEY_FILE = "private_key.pem"
PUBLIC_KEY_FILE = "public_key.pem"
PASSPHRASE_ENV = "SWARMGUARD_KEY_PASS"
SIGNATURE_BYTES = 64


@dataclass
class KeyPaths:
    """Where a key pair lives on disk."""

    private: Path
    public: Path


def _passphrase() -> bytes | None:
    """The private-key passphrase from SWARMGUARD_KEY_PASS, or None if unset/empty."""
    value = os.environ.get(PASSPHRASE_ENV, "")
    return value.encode("utf-8") if value else None


def _restrict_permissions(path: Path) -> None:
    """Make a key file readable by its owner only (chmod 600) where supported.

    WHY: other users of the computer should not be able to read the
    private key. On Windows, os.chmod can only toggle "read-only", so the
    call is harmless but weak; real protection there comes from the user
    profile's folder permissions. We log that limitation instead of hiding it.
    """
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.warning("could not restrict permissions on %s: %s", path, exc)
        return
    if os.name == "nt":
        logger.info("Windows: chmod 600 is only partly honoured for %s", path)


def ensure_keypair(keys_dir: str | Path = "keys") -> KeyPaths:
    """Create an Ed25519 key pair in keys_dir on the first run; reuse it afterwards.

    private_key.pem  PKCS8 PEM. Encrypted with BestAvailableEncryption when
                     SWARMGUARD_KEY_PASS is set, otherwise unencrypted (a
                     warning is logged, because anyone who copies the file
                     can then sign as this installation).
    public_key.pem   SubjectPublicKeyInfo PEM, safe to share.
    """
    folder = Path(keys_dir)
    paths = KeyPaths(folder / PRIVATE_KEY_FILE, folder / PUBLIC_KEY_FILE)
    if paths.private.exists() and paths.public.exists():
        return paths
    if paths.private.exists() != paths.public.exists():
        raise ValueError(f"{folder} holds only half a key pair; remove it or restore the missing file")
    folder.mkdir(parents=True, exist_ok=True)
    private_key = Ed25519PrivateKey.generate()
    password = _passphrase()
    if password is None:
        encryption: serialization.KeySerializationEncryption = serialization.NoEncryption()
        logger.warning("%s is not set: the private key is stored UNENCRYPTED in %s",
                       PASSPHRASE_ENV, paths.private)
    else:
        encryption = serialization.BestAvailableEncryption(password)
    paths.private.write_bytes(private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption))
    paths.public.write_bytes(public_key_pem(private_key.public_key()))
    _restrict_permissions(paths.private)
    logger.info("generated a new Ed25519 key pair in %s", folder)
    return paths


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    """Load a PKCS8 PEM private key (using SWARMGUARD_KEY_PASS if it is encrypted)."""
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=_passphrase())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot load private key {path}: {exc} "
                         f"(is {PASSPHRASE_ENV} set correctly?)") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{path} is not an Ed25519 private key")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    """Load a PEM public key and check it is Ed25519."""
    return public_key_from_pem(Path(path).read_bytes())


def public_key_from_pem(pem: bytes) -> Ed25519PublicKey:
    """Parse PEM bytes into an Ed25519 public key (ValueError if not one)."""
    try:
        key = serialization.load_pem_public_key(pem)
    except ValueError as exc:
        raise ValueError(f"invalid public key PEM: {exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("public key is not Ed25519")
    return key


def public_key_pem(public_key: Ed25519PublicKey) -> bytes:
    """Serialise a public key as SubjectPublicKeyInfo PEM bytes."""
    return public_key.public_bytes(serialization.Encoding.PEM,
                                   serialization.PublicFormat.SubjectPublicKeyInfo)


def sign(private_key: Ed25519PrivateKey, data: bytes) -> bytes:
    """Sign data; returns a 64-byte Ed25519 signature."""
    return private_key.sign(bytes(data))


def verify(public_key: Ed25519PublicKey, data: bytes, signature: bytes) -> bool:
    """True if signature is a valid signature of data under public_key.

    WHY catch InvalidSignature and return False: for a verifier, a bad
    signature is an expected answer ("this evidence was changed"), not a
    crash.
    """
    try:
        public_key.verify(bytes(signature), bytes(data))
    except InvalidSignature:
        return False
    return True


def public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    """Short ID of a public key: first 16 hex characters of sha256(raw 32 key bytes).

    WHY a fingerprint: comparing 16 characters by eye (e.g. on the PDF and
    in the app) is practical; comparing a whole PEM block is not.
    """
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]
