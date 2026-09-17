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

import contextlib
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


def _open() -> sqlite3.Connection:
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


@contextlib.contextmanager
def connect():
    """
    `with db.connect() as conn:` -- a fresh connection that is *closed* on
    exit, whatever happens inside.

    This must not be replaced by `with sqlite3.connect(...) as conn:`.  A
    Connection's own context manager only commits or rolls back (a no-op in
    our autocommit mode); it never closes.  And dropping the last reference
    does not close it either: every Connection sits in a reference cycle
    with its own statement cache (an lru_cache wrapper around the
    connection), so it is only reclaimed when the cyclic garbage collector
    happens to run a full pass.  Until then each abandoned connection keeps
    its file descriptors and up to ~2 MB of page cache alive.  On a hub
    answering a request every second that accumulated ~200 open connections
    between GC passes and eventually OOM-killed the process in its LXC.
    """
    conn = _open()
    try:
        yield conn
    finally:
        conn.close()


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agents (
            agent_id         TEXT PRIMARY KEY,
            status_json      TEXT NOT NULL DEFAULT '{}',
            last_seen        REAL NOT NULL DEFAULT 0,
            paused           INTEGER NOT NULL DEFAULT 0,
            abort_requested  INTEGER NOT NULL DEFAULT 0,
            ignore_jasna     INTEGER NOT NULL DEFAULT 0,
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
            webhook_url      TEXT NOT NULL DEFAULT '',
            webhook_headers  TEXT NOT NULL DEFAULT '{}',
            webhook_body     TEXT NOT NULL DEFAULT '{}',
            webhook_rel_path TEXT NOT NULL DEFAULT '',
            webhook_states   TEXT NOT NULL DEFAULT '["done", "skipped"]',
            auto_delete_on_finish INTEGER NOT NULL DEFAULT 0,
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

        -- Fleet-wide hold: a single row.  While a hold is active no agent is
        -- handed new work (and, with abort_in_flight, files in progress are
        -- aborted and requeued).  Two sources, merged by fleet_hold():
        --   * an ad-hoc hold with a deadline (`held_until`), set by an
        --     external event -- a SnapRAID sync starting -- or by hand.  The
        --     deadline is what makes it fail-open: a releaser that never
        --     calls back cannot park the fleet forever;
        --   * a daily window (`window_*`), hub local time, for a maintenance
        --     slot that must be covered whether or not any event fires.
        --     `window_skip_until` is the end of one occurrence to sit out
        --     (dashboard "Release" during a window = skip today only).
        CREATE TABLE IF NOT EXISTS fleet_hold (
            id                INTEGER PRIMARY KEY CHECK (id = 1),
            reason            TEXT NOT NULL DEFAULT '',
            held_until        REAL NOT NULL DEFAULT 0,
            set_at            REAL NOT NULL DEFAULT 0,
            set_by            TEXT NOT NULL DEFAULT '',
            abort_in_flight   INTEGER NOT NULL DEFAULT 0,
            window_enabled    INTEGER NOT NULL DEFAULT 0,
            window_start      TEXT NOT NULL DEFAULT '02:55',
            window_minutes    INTEGER NOT NULL DEFAULT 65,
            window_reason     TEXT NOT NULL DEFAULT 'maintenance window',
            window_abort      INTEGER NOT NULL DEFAULT 1,
            window_skip_until REAL NOT NULL DEFAULT 0
        );
        INSERT OR IGNORE INTO fleet_hold (id) VALUES (1);
        """
    )


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database's first `CREATE TABLE` ran."""
    job_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "agent_allowlist" not in job_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN agent_allowlist TEXT NOT NULL DEFAULT '[]'")
    if "webhook_url" not in job_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN webhook_url TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE jobs ADD COLUMN webhook_headers TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE jobs ADD COLUMN webhook_body TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE jobs ADD COLUMN webhook_rel_path TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE jobs ADD COLUMN webhook_states TEXT NOT NULL DEFAULT '[\"done\", \"skipped\"]'")
    if "auto_delete_on_finish" not in job_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN auto_delete_on_finish INTEGER NOT NULL DEFAULT 0")

    agent_cols = {row["name"] for row in conn.execute("PRAGMA table_info(agents)")}
    if "abort_requested" not in agent_cols:
        conn.execute("ALTER TABLE agents ADD COLUMN abort_requested INTEGER NOT NULL DEFAULT 0")
    if "ignore_jasna" not in agent_cols:
        conn.execute("ALTER TABLE agents ADD COLUMN ignore_jasna INTEGER NOT NULL DEFAULT 0")

    hold_cols = {row["name"] for row in conn.execute("PRAGMA table_info(fleet_hold)")}
    if "window_enabled" not in hold_cols:
        for ddl in (
            "abort_in_flight   INTEGER NOT NULL DEFAULT 0",
            "window_enabled    INTEGER NOT NULL DEFAULT 0",
            "window_start      TEXT NOT NULL DEFAULT '02:55'",
            "window_minutes    INTEGER NOT NULL DEFAULT 65",
            "window_reason     TEXT NOT NULL DEFAULT 'maintenance window'",
            "window_abort      INTEGER NOT NULL DEFAULT 1",
            "window_skip_until REAL NOT NULL DEFAULT 0",
        ):
            conn.execute(f"ALTER TABLE fleet_hold ADD COLUMN {ddl}")


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

# ---------------------------------------------------------------------------
# Fleet hold
# ---------------------------------------------------------------------------

def parse_hhmm(text: str) -> Optional[tuple[int, int]]:
    try:
        hh, mm = (int(x) for x in str(text).strip().split(":"))
    except ValueError:
        return None
    if 0 <= hh < 24 and 0 <= mm < 60:
        return hh, mm
    return None


def window_bounds(start_hhmm: str, minutes: int, t: float) -> Optional[tuple[float, float]]:
    """
    (start, end) epoch seconds of the daily window occurrence that contains
    local time `t`, or None.  Checks today's and yesterday's start so a window
    that crosses midnight (23:30 for 90 min) is still recognised at 00:15.
    `mktime` on a possibly-out-of-range day-of-month is well defined and
    handles month wrap and DST for us.
    """
    hm = parse_hhmm(start_hhmm)
    if hm is None or minutes <= 0:
        return None
    lt = time.localtime(t)
    for day_offset in (0, -1):
        st = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + day_offset, hm[0], hm[1], 0, 0, 0, -1))
        en = st + minutes * 60
        if st <= t < en:
            return st, en
    return None


def window_next_start(start_hhmm: str, t: float) -> Optional[float]:
    """Epoch seconds of the next occurrence's start strictly after `t`."""
    hm = parse_hhmm(start_hhmm)
    if hm is None:
        return None
    lt = time.localtime(t)
    for day_offset in (0, 1):
        st = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + day_offset, hm[0], hm[1], 0, 0, 0, -1))
        if st > t:
            return st
    return None


def _hold_row(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute("SELECT * FROM fleet_hold WHERE id = 1").fetchone()


def fleet_hold(conn: sqlite3.Connection) -> Optional[dict]:
    """
    The effective fleet-wide hold, or None.  Merges the ad-hoc hold and the
    scheduled window: the fleet is held while either is active, until the
    later of their deadlines, aborting in-flight files if either says so.

    An ad-hoc hold whose deadline has passed is cleared here, lazily, the
    first time anything asks after it lapsed -- the hub has no timers, and
    every path that cares (check-in, claim, heartbeat, the dashboard poll)
    goes through this function.
    """
    row = _hold_row(conn)
    if row is None:
        return None
    t = now()
    sources, reasons = [], []
    until, set_at, set_by, abort = 0.0, 0.0, "", False

    if row["held_until"] > t:
        sources.append("manual")
        reasons.append(row["reason"] or "no reason given")
        until, set_at, set_by = row["held_until"], row["set_at"], row["set_by"]
        abort = bool(row["abort_in_flight"])
    elif row["held_until"]:
        conn.execute("UPDATE fleet_hold SET held_until = 0 WHERE id = 1 AND held_until = ?",
                     (row["held_until"],))
        log_event(conn, f"Fleet hold expired without release: {row['reason'] or 'no reason given'}",
                  level="warn")

    if row["window_enabled"]:
        b = window_bounds(row["window_start"], row["window_minutes"], t)
        if b and b[1] > (row["window_skip_until"] or 0):
            sources.append("window")
            reasons.append(row["window_reason"] or "maintenance window")
            if b[1] > until:
                until = b[1]
            if not set_at:
                set_at, set_by = b[0], "schedule"
            abort = abort or bool(row["window_abort"])

    if not sources:
        return None
    return {
        "reason":          " · ".join(reasons),
        "until":           until,
        "remaining":       round(until - t, 1),
        "set_at":          set_at,
        "set_by":          set_by,
        "abort_in_flight": abort,
        "sources":         sources,
    }


def set_fleet_hold(conn: sqlite3.Connection, reason: str, ttl_seconds: float, set_by: str = "",
                   abort_in_flight: bool = False, extend_only: bool = False) -> Optional[dict]:
    """
    Start, extend or re-label the ad-hoc hold.  Idempotent: a second call
    moves the deadline and updates the reason.  With `extend_only` the
    deadline can only move later -- for callers that layer on top of a hold
    someone else set (the SnapRAID hook must never cut a manual hold short).
    """
    t = now()
    new_until = t + ttl_seconds
    if extend_only:
        row = _hold_row(conn)
        if row["held_until"] >= new_until:
            # Still honour a request to start aborting in-flight files.
            if abort_in_flight and not row["abort_in_flight"]:
                conn.execute("UPDATE fleet_hold SET abort_in_flight = 1 WHERE id = 1")
            return fleet_hold(conn)
    conn.execute(
        """
        UPDATE fleet_hold
           SET reason = ?, held_until = ?, set_at = ?, set_by = ?, abort_in_flight = ?
         WHERE id = 1
        """,
        (reason, new_until, t, set_by, 1 if abort_in_flight else 0),
    )
    return fleet_hold(conn)


def clear_fleet_hold(conn: sqlite3.Connection) -> list[str]:
    """
    Release whatever is holding the fleet right now: the ad-hoc hold is
    cleared, and if the scheduled window is open this occurrence is skipped
    (the schedule itself stays; it fires again tomorrow).  Returns the
    sources that were released.
    """
    t = now()
    released = []
    cur = conn.execute("UPDATE fleet_hold SET held_until = 0 WHERE id = 1 AND held_until > ?", (t,))
    if cur.rowcount:
        released.append("manual")
    row = _hold_row(conn)
    if row["window_enabled"]:
        b = window_bounds(row["window_start"], row["window_minutes"], t)
        if b and b[1] > (row["window_skip_until"] or 0):
            conn.execute("UPDATE fleet_hold SET window_skip_until = ? WHERE id = 1", (b[1],))
            released.append("window")
    return released


def hold_window(conn: sqlite3.Connection) -> dict:
    """The schedule as configured, plus when it next opens -- for the dashboard."""
    row = _hold_row(conn)
    t = now()
    b = window_bounds(row["window_start"], row["window_minutes"], t)
    skipped = bool(b) and b[1] <= (row["window_skip_until"] or 0)
    nxt = None
    if row["window_enabled"]:
        # If today's occurrence is open (or skipped), the next one is tomorrow's.
        nxt = window_next_start(row["window_start"], b[1] if b else t)
    return {
        "enabled":         bool(row["window_enabled"]),
        "start":           row["window_start"],
        "minutes":         row["window_minutes"],
        "reason":          row["window_reason"],
        "abort_in_flight": bool(row["window_abort"]),
        "open_now":        bool(b) and not skipped,
        "skipped_today":   skipped,
        "next_start":      nxt,
        "hub_tz":          time.strftime("%Z", time.localtime(t)),
        "hub_time":        time.strftime("%H:%M", time.localtime(t)),
    }


def set_hold_window(conn: sqlite3.Connection, *, enabled: bool, start: str, minutes: int,
                    reason: str, abort_in_flight: bool) -> dict:
    if parse_hhmm(start) is None:
        raise ValueError("start must be HH:MM (24h)")
    if not (1 <= int(minutes) <= 24 * 60):
        raise ValueError("minutes must be between 1 and 1440")
    conn.execute(
        """
        UPDATE fleet_hold
           SET window_enabled = ?, window_start = ?, window_minutes = ?, window_reason = ?,
               window_abort = ?, window_skip_until = 0
         WHERE id = 1
        """,
        (1 if enabled else 0, start.strip(), int(minutes), reason.strip() or "maintenance window",
         1 if abort_in_flight else 0),
    )
    return hold_window(conn)


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
               started_at = NULL,
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
    with connect() as conn:
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
                           started_at = ?,
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
               started_at = NULL,
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

def current_item_started(conn: sqlite3.Connection, agent_id: str) -> Optional[float]:
    """When the file this agent is working on right now was claimed, or None
    if it has nothing in flight.  `started_at` is per attempt (claim_next sets
    it, requeue_item / release_expired_claims clear it), so the time since is
    this run of whisperjav on that file, not time since it was first tried."""
    row = conn.execute(
        """
        SELECT started_at FROM work_items
         WHERE claimed_by = ? AND state IN ('claimed', 'running') AND started_at IS NOT NULL
         ORDER BY started_at LIMIT 1
        """,
        (agent_id,),
    ).fetchone()
    return row["started_at"] if row else None


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
    row = conn.execute("SELECT name, state, auto_delete_on_finish FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None or row["state"] != "running":
        return
    counts = job_counts(conn, job_id)
    outstanding = counts["pending"] + counts["claimed"] + counts["running"]
    if counts["total"] > 0 and outstanding == 0:
        # auto_delete only fires on a clean finish -- a job with failures is
        # left in place (with its items' error text) for manual review, even
        # if the caller asked for cleanup. It can still be deleted by hand.
        if row["auto_delete_on_finish"] and counts["failed"] == 0:
            log_event(conn, f"Job '{row['name']}' finished cleanly "
                            f"({counts['done']} done, {counts['skipped']} skipped) and was auto-deleted")
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return
        conn.execute(
            "UPDATE jobs SET state = 'done', updated_at = ? WHERE id = ?",
            (now(), job_id),
        )
        log_event(conn, f"Job finished: {counts['done']} done, "
                        f"{counts['failed']} failed, {counts['skipped']} skipped",
                  job_id=job_id)
