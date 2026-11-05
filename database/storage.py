"""SQLite storage for runs, incidents, evidence packages and the custody log.

TABLES
    runs                one simulation + detection run
    incidents           an opened incident (full JSON kept in incident_json)
    detections_summary  per-detector metrics of a run
    evidence_packages   where each incident's package was written
    calibrations        saved threshold calibrations, keyed by config hash
    custody_log         append-only chain of custody events

WHY SQLite: it is a single file, ships with Python (the sqlite3 module), and
needs no server. That is enough for one investigator's workstation.

WHY parameterised queries ("... WHERE id = ?" with a separate value tuple):
the database driver passes values separately from the SQL text, so a value
such as "x'; DROP TABLE incidents; --" is stored as plain text and never
executed. Building SQL with f-strings would allow SQL injection.

WHY a new connection per operation (the _connect context manager): every
write is committed (or rolled back on error) and the file is closed straight
away, so a crash never leaves half-written data or a locked database.

CUSTODY CHAIN
Every package records previous_manifest_hash = the manifest hash of the
package created just before it ("GENESIS" for the first one). The custody
log stores both hashes, so silently deleting or replacing an earlier
package breaks the chain.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data") / "swarmguard.db"
GENESIS = "GENESIS"
CREATED = "created"

SCHEMA: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY,
        created_utc TEXT NOT NULL,
        scenario TEXT NOT NULL,
        seed INTEGER NOT NULL,
        config_hash TEXT NOT NULL,
        config_json TEXT NOT NULL,
        summary_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        run_id INTEGER,
        created_utc TEXT NOT NULL,
        pattern TEXT,
        affected TEXT,
        onset REAL,
        incident_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS detections_summary (
        id INTEGER PRIMARY KEY,
        run_id INTEGER NOT NULL,
        detector TEXT NOT NULL,
        detected INTEGER,
        delay REAL,
        false_alarms INTEGER,
        id_precision REAL,
        id_recall REAL)""",
    """CREATE TABLE IF NOT EXISTS evidence_packages (
        incident_id TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        zip_path TEXT NOT NULL,
        manifest_sha256 TEXT NOT NULL,
        created_utc TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS calibrations (
        config_hash TEXT PRIMARY KEY,
        created_utc TEXT NOT NULL,
        json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS custody_log (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id TEXT NOT NULL,
        manifest_sha256 TEXT NOT NULL,
        previous_sha256 TEXT NOT NULL,
        action TEXT NOT NULL,
        created_utc TEXT NOT NULL)""",
)


def utc_now() -> str:
    """Current UTC time as ISO 8601 text, e.g. "2026-11-05T10:30:00+00:00"."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _nan_to_none(value: Any) -> Any:
    """NaN -> None so SQLite stores NULL (NaN is not a portable SQL value)."""
    if isinstance(value, float) and value != value:  # WHY: NaN is the only value != itself
        return None
    return value


class Storage:
    """All database access for SwarmGuard. Create one with Storage(path)."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.create_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection; commit on success, roll back on error, always close.

        WHY not `with sqlite3.connect(...)` alone: that form commits but
        does NOT close the connection, which keeps the file locked on Windows.
        """
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row      # rows behave like dictionaries
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_schema(self) -> None:
        """Create every table if it does not exist yet (safe to call any number of times)."""
        with self._connect() as conn:
            for statement in SCHEMA:
                conn.execute(statement)

    def table_names(self) -> list[str]:
        """Names of the tables in the database, sorted."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        return sorted(r["name"] for r in rows)

    # ------------------------------------------------------------------ runs
    def save_run(self, scenario: str, seed: int, config_hash: str,
                 config: dict[str, Any], summary: dict[str, Any]) -> int:
        """Insert one run; returns its new integer id."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (created_utc, scenario, seed, config_hash, config_json, summary_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (utc_now(), scenario, int(seed), config_hash,
                 json.dumps(config, sort_keys=True, default=str),
                 json.dumps(summary, sort_keys=True, default=str)),
            )
            return int(cur.lastrowid)

    def save_detection_summary(self, run_id: int, rows: list[dict[str, Any]]) -> None:
        """Insert per-detector metrics rows: detector, detected, delay, false_alarms, id_precision, id_recall."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO detections_summary (run_id, detector, detected, delay, false_alarms, "
                "id_precision, id_recall) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(int(run_id), r["detector"], int(bool(r["detected"])), _nan_to_none(r.get("delay")),
                  int(r.get("false_alarms", 0)), _nan_to_none(r.get("id_precision")),
                  _nan_to_none(r.get("id_recall"))) for r in rows],
            )

    def detection_summary(self, run_id: int) -> list[dict[str, Any]]:
        """Per-detector metric rows of one run."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM detections_summary WHERE run_id = ? ORDER BY detector",
                                (int(run_id),)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- incidents
    def next_incident_number(self, day: date) -> int:
        """Next free NNN for incident ids "SG-YYYYMMDD-NNN" on this date (starts at 1).

        Uses the largest existing NNN + 1 rather than a count, so a gap
        (e.g. a deleted row) can never cause an id to be reused.
        """
        prefix = f"SG-{day.strftime('%Y%m%d')}-"
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM incidents WHERE id LIKE ?", (prefix + "%",)).fetchall()
        numbers = [int(r["id"][len(prefix):]) for r in rows if r["id"][len(prefix):].isdigit()]
        return max(numbers, default=0) + 1

    def save_incident(self, incident: dict[str, Any], run_id: int | None = None) -> None:
        """Insert (or replace) an incident given as a dict (Incident.to_dict())."""
        affected = ",".join(str(i) for i in incident.get("affected_drones_detected", []))
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO incidents (id, run_id, created_utc, pattern, affected, onset, "
                "incident_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (incident["incident_id"], run_id, incident["created_utc"], incident.get("pattern"),
                 affected, _nan_to_none(incident.get("onset_estimate")),
                 json.dumps(incident, sort_keys=True, default=str)),
            )

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        """The stored incident dict, or None if the id is unknown."""
        with self._connect() as conn:
            row = conn.execute("SELECT incident_json FROM incidents WHERE id = ?",
                               (incident_id,)).fetchone()
        return json.loads(row["incident_json"]) if row else None

    def find_incident_by_run_key(self, run_key: str) -> dict[str, Any] | None:
        """The incident opened for exactly this run (same configs + seed), or None.

        WHY: the same configuration and seed always give identical output
        (rule R2), so a second incident for it would only duplicate the
        first. The app and the CLI reuse the existing one instead.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT incident_json FROM incidents "
                               "WHERE json_extract(incident_json, '$.run_key') = ? "
                               "ORDER BY created_utc LIMIT 1", (run_key,)).fetchone()
        return json.loads(row["incident_json"]) if row else None

    def list_incidents(self, pattern: str | None = None, scenario: str | None = None,
                       since: str | None = None) -> list[dict[str, Any]]:
        """Incident rows (newest first), optionally filtered.

        pattern   exact agreement-pattern key, e.g. "A_silent_B_C_alarm"
        scenario  scenario key stored inside the incident JSON
        since     ISO time; only incidents created at or after it

        WHY the WHERE clause is assembled from fixed fragments: only the
        constant text "pattern = ?" etc. is joined; user values still go
        through the ? placeholders, so this stays injection-safe.
        """
        clauses, values = [], []
        if pattern:
            clauses.append("i.pattern = ?")
            values.append(pattern)
        if scenario:
            clauses.append("json_extract(i.incident_json, '$.scenario_key') = ?")
            values.append(scenario)
        if since:
            clauses.append("i.created_utc >= ?")
            values.append(since)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = ("SELECT i.id, i.run_id, i.created_utc, i.pattern, i.affected, i.onset, "
               "json_extract(i.incident_json, '$.scenario_key') AS scenario, "
               "p.zip_path FROM incidents i LEFT JOIN evidence_packages p ON p.incident_id = i.id"
               f"{where} ORDER BY i.created_utc DESC, i.id DESC")
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(values)).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------- calibrations
    def save_calibration(self, config_hash: str, data: dict[str, Any]) -> None:
        """Store (or replace) a calibration as JSON under its config hash."""
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO calibrations (config_hash, created_utc, json) "
                         "VALUES (?, ?, ?)", (config_hash, utc_now(), json.dumps(data, sort_keys=True)))

    def get_calibration(self, config_hash: str) -> dict[str, Any] | None:
        """A stored calibration dict, or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT json FROM calibrations WHERE config_hash = ?",
                               (config_hash,)).fetchone()
        return json.loads(row["json"]) if row else None

    # ------------------------------------------------------------- packages
    def save_package(self, incident_id: str, path: str | Path, zip_path: str | Path,
                     manifest_sha256: str) -> None:
        """Record where an incident's evidence package was written."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO evidence_packages (incident_id, path, zip_path, manifest_sha256, "
                "created_utc) VALUES (?, ?, ?, ?, ?)",
                (incident_id, str(path), str(zip_path), manifest_sha256, utc_now()),
            )

    def list_packages(self, limit: int = 8) -> list[dict[str, Any]]:
        """The newest evidence packages: [{incident_id, manifest_sha256, created_utc}, ...].

        Newest first, at most `limit` rows. Used by the app's hash ribbon
        (Phase 7, motif M5) to show REAL manifest hashes, never made-up ones.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        with self._connect() as conn:
            rows = conn.execute("SELECT incident_id, manifest_sha256, created_utc FROM evidence_packages "
                                "ORDER BY created_utc DESC, incident_id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_package(self, incident_id: str) -> dict[str, Any] | None:
        """The evidence_packages row of an incident, or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM evidence_packages WHERE incident_id = ?",
                               (incident_id,)).fetchone()
        return dict(row) if row else None

    # --------------------------------------------------------------- custody
    def last_manifest_hash(self) -> str:
        """Manifest hash of the most recently CREATED package, or "GENESIS" if none."""
        with self._connect() as conn:
            row = conn.execute("SELECT manifest_sha256 FROM custody_log WHERE action = ? "
                               "ORDER BY seq DESC LIMIT 1", (CREATED,)).fetchone()
        return row["manifest_sha256"] if row else GENESIS

    def log_custody(self, incident_id: str, manifest_sha256: str, previous_sha256: str,
                    action: str) -> int:
        """Append one custody event (e.g. "created", "verified"); returns its seq number.

        WHY append-only: rows are only ever inserted, never updated, so the
        log is a history of what happened, in order.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO custody_log (incident_id, manifest_sha256, previous_sha256, action, "
                "created_utc) VALUES (?, ?, ?, ?, ?)",
                (incident_id, manifest_sha256, previous_sha256, action, utc_now()),
            )
            return int(cur.lastrowid)

    def list_custody(self, incident_id: str | None = None) -> list[dict[str, Any]]:
        """Custody events in order (all, or only one incident's)."""
        sql = "SELECT * FROM custody_log"
        values: tuple[Any, ...] = ()
        if incident_id is not None:
            sql += " WHERE incident_id = ?"
            values = (incident_id,)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY seq", values).fetchall()
        return [dict(r) for r in rows]

    def custody_entry(self, incident_id: str) -> dict[str, Any] | None:
        """The "created" custody row of an incident, or None if it is not in this database."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM custody_log WHERE incident_id = ? AND action = ? "
                               "ORDER BY seq LIMIT 1", (incident_id, CREATED)).fetchone()
        return dict(row) if row else None

    def created_before(self, seq: int) -> list[dict[str, Any]]:
        """All "created" custody rows with a smaller seq (earlier packages), in order."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM custody_log WHERE action = ? AND seq < ? ORDER BY seq",
                                (CREATED, int(seq))).fetchall()
        return [dict(r) for r in rows]
