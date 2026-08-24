#!/usr/bin/env python3
"""
db.py — SQLite storage and the atomic work-claim logic for the fleet hub.

The single most important guarantee in this file:

    A work item is handed to exactly one agent at a time.

That is enforced by `claim_next()`, which runs the select-and-mark as one
`BEGIN IMMEDIATE` transaction.  SQLite allows only one writer at a time, so
two agents polling simultaneously are serialised by the database itself --
the second one's conditional UPDATE matches zero rows and it retries.
Claims carry a lease (`claim_expires`); if an agent dies mid-file the lease
lapses and the item returns to the pending pool.

Nothing here is specific to any machine, path or OS: the hub never touches
the media filesystem.  Agents do all filesystem work and report back.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Lifecycle vocabulary
# ---------------------------------------------------------------------------

JOB_STATES  = ("draft", "scanning", "running", "paused", "done", "stopped", "failed")
ITEM_STATES = ("pending", "claimed", "running", "done", "failed", "skipped")

# How long a claim is valid without a heartbeat before the item is re-queued.
# Generous, because a single video can legitimately take a long time and the
# agent only heartbeats between progress updates.
CLAIM_LEASE_SECONDS = 900

# No check-in within this window means the agent is shown as offline.
AGENT_OFFLINE_SECONDS = 30

_DB_PATH: Optional[Path] = None


# ---------------------------------------------------------------------------
# Connection handling
# ---------------------------------------------------------------------------

def configure(db_path: Path) -> None:
    """Point the module at a database file and make sure the schema exists."""
    global _DB_PATH
    _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _create_schema(conn)
        _migrate_schema(conn)


def connect() -> sqlite3.Connection:
    """
    A fresh connection per call.  `timeout` makes concurrent writers wait for
    the write lock rather than immediately raising 'database is locked'.
    """
    if _DB_PATH is None:
        raise RuntimeError("db.configure() has not been called")
    conn = sqlite3.connect(str(_DB_PATH), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agents (
            agent_id         TEXT PRIMARY KEY,
            status_json      TEXT NOT NULL DEFAULT '{}',
            last_seen        REAL NOT NULL DEFAULT 0,
            paused           INTEGER NOT NULL DEFAULT 0,
            abort_requested  INTEGER NOT NULL DEFAULT 0,
            first_seen       REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS profiles (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL UNIQUE,
            args_template    TEXT NOT NULL,
            lang_code        TEXT NOT NULL DEFAULT 'qu',
            force            INTEGER NOT NULL DEFAULT 0,
            video_extensions TEXT NOT NULL DEFAULT '[]',
            notes            TEXT NOT NULL DEFAULT '',
            created_at       REAL NOT NULL,
            updated_at       REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL,
            target_dir       TEXT NOT NULL,
            profile_id       INTEGER,
            profile_name     TEXT NOT NULL DEFAULT '',
            args_template    TEXT NOT NULL,
            lang_code        TEXT NOT NULL,
            force            INTEGER NOT NULL DEFAULT 0,
            video_extensions TEXT NOT NULL DEFAULT '[]',
            state            TEXT NOT NULL DEFAULT 'draft',
            priority         INTEGER NOT NULL DEFAULT 100,
            dry_run          INTEGER NOT NULL DEFAULT 0,
            agent_allowlist  TEXT NOT NULL DEFAULT '[]',
            scan_agent       TEXT,
            scan_started_at  REAL,
            error            TEXT NOT NULL DEFAULT '',
            created_at       REAL NOT NULL,
            updated_at       REAL NOT NULL,
            FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS work_items (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id        INTEGER NOT NULL,
            rel_path      TEXT NOT NULL,
            state         TEXT NOT NULL DEFAULT 'pending',
            claimed_by    TEXT,
            claim_expires REAL NOT NULL DEFAULT 0,
            attempts      INTEGER NOT NULL DEFAULT 0,
            progress      TEXT NOT NULL DEFAULT '',
            last_error    TEXT NOT NULL DEFAULT '',
            duration      REAL NOT NULL DEFAULT 0,
            started_at    REAL,
            finished_at   REAL,
            updated_at    REAL NOT NULL,
            UNIQUE (job_id, rel_path),
            FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_items_claimable
            ON work_items (job_id, state, id);
        CREATE INDEX IF NOT EXISTS idx_items_lease
            ON work_items (state, claim_expires);

        CREATE TABLE IF NOT EXISTS events (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       REAL NOT NULL,
            level    TEXT NOT NULL DEFAULT 'info',
            agent_id TEXT NOT NULL DEFAULT '',
            job_id   INTEGER,
            message  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts DESC);

        -- Folder picker: the hub cannot see the media filesystem, so a browse
        -- request is parked here and answered by an agent on its next poll.
        CREATE TABLE IF NOT EXISTS browse_requests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id    TEXT NOT NULL,
            path        TEXT NOT NULL,
            state       TEXT NOT NULL DEFAULT 'pending',
            result_json TEXT NOT NULL DEFAULT '{}',
            created_at  REAL NOT NULL
        );
        """
    )


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database's first `CREATE TABLE` ran."""
    job_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "agent_allowlist" not in job_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN agent_allowlist TEXT NOT NULL DEFAULT '[]'")

    agent_cols = {row["name"] for row in conn.execute("PRAGMA table_info(agents)")}
    if "abort_requested" not in agent_cols:
        conn.execute("ALTER TABLE agents ADD COLUMN abort_requested INTEGER NOT NULL DEFAULT 0")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def now() -> float:
    return time.time()


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    return dict(row) if row is not None else None


def agent_allowed(allowlist_json: Optional[str], agent_id: str) -> bool:
    """Empty allowlist ([] or '') means unrestricted -- any agent may claim."""
    allowlist = json.loads(allowlist_json) if allowlist_json else []
    return not allowlist or agent_id in allowlist


def log_event(conn: sqlite3.Connection, message: str, *,
              level: str = "info", agent_id: str = "", job_id: Optional[int] = None) -> None:
    conn.execute(
        "INSERT INTO events (ts, level, agent_id, job_id, message) VALUES (?, ?, ?, ?, ?)",
        (now(), level, agent_id, job_id, message),
    )
    # Keep the ring buffer bounded so the DB never grows without limit.
    conn.execute(
        "DELETE FROM events WHERE id < (SELECT MAX(id) - 2000 FROM events)"
    )


# ---------------------------------------------------------------------------
# The claim -- the core of the double-processing guard
# ---------------------------------------------------------------------------

def release_expired_claims(conn: sqlite3.Connection) -> int:
    """
    Return items whose lease lapsed to the pending pool.  Called before every
    claim attempt and by the janitor, so a crashed agent never strands work.
    """
    cur = conn.execute(
        """
        UPDATE work_items
           SET state = 'pending',
               claimed_by = NULL,
               claim_expires = 0,
               progress = '',
               last_error = 'claim lease expired -- re-queued',
               updated_at = ?
         WHERE state IN ('claimed', 'running')
           AND claim_expires > 0
           AND claim_expires < ?
        """,
        (now(), now()),
    )
    return cur.rowcount or 0


def claim_next(agent_id: str, *, lease_seconds: int = CLAIM_LEASE_SECONDS) -> Optional[dict]:
    """
    Atomically hand this agent exactly one pending work item, or None.

    Everything happens inside one BEGIN IMMEDIATE transaction, which takes
    SQLite's write lock up front.  A concurrent caller either waits for the
    lock or finds the row already claimed -- it can never receive the same
    item.  The `AND state = 'pending'` on the UPDATE is the belt-and-braces
    check that makes the operation a genuine compare-and-swap.
    """
    conn = connect()
    try:
        for _ in range(5):
            conn.execute("BEGIN IMMEDIATE")
            try:
                release_expired_claims(conn)

                # `agent_allowlist` is a property of the *job*, not of the
                # item, so eligibility is decided per job.  That matters:
                # filtering a batch of candidate items in Python instead meant
                # an agent could be handed nothing at all when enough items
                # routed to other machines sat ahead of its next eligible one.
                # Jobs number in the tens, so walking them is cheap and needs
                # no LIMIT -- which is what makes starvation impossible rather
                # than merely unlikely.  (The filtering stays in Python because
                # SQLite's JSON1 extension is not guaranteed to be present.)
                #
                # Ordering is unchanged: priority, then job id, then item id
                # within the chosen job.
                eligible_jobs = conn.execute(
                    """
                    SELECT id, agent_allowlist
                      FROM jobs
                     WHERE state = 'running'
                     ORDER BY priority ASC, id ASC
                    """
                ).fetchall()

                row = None
                for job_row in eligible_jobs:
                    if not agent_allowed(job_row["agent_allowlist"], agent_id):
                        continue
                    # Indexed by idx_items_claimable (job_id, state, id).
                    row = conn.execute(
                        """
                        SELECT id, job_id, rel_path, attempts
                          FROM work_items
                         WHERE job_id = ? AND state = 'pending'
                         ORDER BY id ASC
                         LIMIT 1
                        """,
                        (job_row["id"],),
                    ).fetchone()
                    if row is not None:
                        break

                if row is None:
                    conn.execute("COMMIT")
                    return None

                expires = now() + lease_seconds
                cur = conn.execute(
                    """
                    UPDATE work_items
                       SET state = 'claimed',
                           claimed_by = ?,
                           claim_expires = ?,
                           attempts = attempts + 1,
                           started_at = COALESCE(started_at, ?),
                           progress = '',
                           updated_at = ?
                     WHERE id = ?
                       AND state = 'pending'
                    """,
                    (agent_id, expires, now(), now(), row["id"]),
                )

                if cur.rowcount != 1:
                    # Someone beat us to this row; roll back and look again.
                    conn.execute("ROLLBACK")
                    continue

                job = conn.execute(
                    """
                    SELECT id, name, target_dir, args_template, lang_code,
                           force, dry_run, video_extensions
                      FROM jobs WHERE id = ?
                    """,
                    (row["job_id"],),
                ).fetchone()

                conn.execute("COMMIT")

                return {
                    "item_id":       row["id"],
                    "job_id":        row["job_id"],
                    "rel_path":      row["rel_path"],
                    "attempt":       (row["attempts"] or 0) + 1,
                    "lease_seconds": lease_seconds,
                    "job": {
                        "id":               job["id"],
                        "name":             job["name"],
                        "target_dir":       job["target_dir"],
                        "args_template":    job["args_template"],
                        "lang_code":        job["lang_code"],
                        "force":            bool(job["force"]),
                        "dry_run":          bool(job["dry_run"]),
                        "video_extensions": json.loads(job["video_extensions"] or "[]"),
                    },
                }
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
        return None
    finally:
        conn.close()


def heartbeat_item(conn: sqlite3.Connection, item_id: int, agent_id: str,
                   progress: str = "", *, lease_seconds: int = CLAIM_LEASE_SECONDS) -> bool:
    """
    Extend the lease on an in-flight item.  Scoped to `claimed_by = agent_id`
    so an agent that lost its claim to lease expiry cannot silently resurrect
    it after another agent has taken over.
    """
    cur = conn.execute(
        """
        UPDATE work_items
           SET state = 'running',
               claim_expires = ?,
               progress = ?,
               updated_at = ?
         WHERE id = ?
           AND claimed_by = ?
           AND state IN ('claimed', 'running')
        """,
        (now() + lease_seconds, progress, now(), item_id, agent_id),
    )
    return (cur.rowcount or 0) == 1


def complete_item(conn: sqlite3.Connection, item_id: int, agent_id: str,
                  state: str, message: str = "", duration: float = 0.0) -> bool:
    """Record a terminal result for an item.  Also scoped to the claim owner."""
    if state not in ("done", "failed", "skipped"):
        raise ValueError(f"invalid terminal state: {state}")
    cur = conn.execute(
        """
        UPDATE work_items
           SET state = ?,
               last_error = ?,
               duration = ?,
               claim_expires = 0,
               finished_at = ?,
               progress = '',
               updated_at = ?
         WHERE id = ?
           AND claimed_by = ?
           AND state IN ('claimed', 'running')
        """,
        (state, message, duration, now(), now(), item_id, agent_id),
    )
    return (cur.rowcount or 0) == 1


def requeue_item(conn: sqlite3.Connection, item_id: int, agent_id: str, reason: str = "") -> bool:
    """
    Put a claimed item back without counting it as a failure -- used when an
    agent is told to pause/stop mid-file, or when it declines the work.
    """
    cur = conn.execute(
        """
        UPDATE work_items
           SET state = 'pending',
               claimed_by = NULL,
               claim_expires = 0,
               progress = '',
               last_error = ?,
               updated_at = ?
         WHERE id = ?
           AND claimed_by = ?
           AND state IN ('claimed', 'running')
        """,
        (reason, now(), item_id, agent_id),
    )
    return (cur.rowcount or 0) == 1


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

def job_counts(conn: sqlite3.Connection, job_id: int) -> dict:
    counts = {s: 0 for s in ITEM_STATES}
    for row in conn.execute(
        "SELECT state, COUNT(*) AS n FROM work_items WHERE job_id = ? GROUP BY state",
        (job_id,),
    ):
        counts[row["state"]] = row["n"]
    counts["total"] = sum(counts[s] for s in ITEM_STATES)
    counts["finished"] = counts["done"] + counts["failed"] + counts["skipped"]
    return counts


def maybe_finish_job(conn: sqlite3.Connection, job_id: int) -> None:
    """Flip a running job to 'done' once no item can still be worked on."""
    row = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None or row["state"] != "running":
        return
    counts = job_counts(conn, job_id)
    outstanding = counts["pending"] + counts["claimed"] + counts["running"]
    if counts["total"] > 0 and outstanding == 0:
        conn.execute(
            "UPDATE jobs SET state = 'done', updated_at = ? WHERE id = ?",
            (now(), job_id),
        )
        log_event(conn, f"Job finished: {counts['done']} done, "
                        f"{counts['failed']} failed, {counts['skipped']} skipped",
                  job_id=job_id)
