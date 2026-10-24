#!/usr/bin/env python3
"""
app.py -- WhisperJAV fleet hub.

One always-on server (designed to sit happily in a small LXC) that owns the
job queue and hands out work.  It deliberately never touches the media
filesystem: every path operation -- scanning, browsing, reading, writing --
is performed by an agent on a worker machine and reported back.  That is what
lets the hub run somewhere with no CIFS mount and no GPU.

Agents *pull* over plain HTTP, so worker machines need no inbound ports.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8787

Optional auth (recommended if reachable beyond a trusted LAN):
    export HUB_USER=someuser
    export HUB_PASS=somepassword
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

import db

BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH    = Path(os.environ.get("HUB_DB_PATH", str(BASE_DIR / "hub.sqlite3")))

HUB_USER = os.environ.get("HUB_USER", "")
HUB_PASS = os.environ.get("HUB_PASS", "")

# A scan that produces no result in this long is handed to another agent.
SCAN_TIMEOUT_SECONDS = 300

# A fleet hold set without an explicit ttl_seconds lapses after this long.
# Long enough for a nightly SnapRAID sync, short enough that a hook script
# that dies before releasing does not idle the fleet for days.
HOLD_DEFAULT_TTL_SECONDS = 6 * 3600
HOLD_MAX_TTL_SECONDS = 7 * 24 * 3600

app = FastAPI(title="WhisperJAV Fleet Hub")
security = HTTPBasic(auto_error=False)

db.configure(DB_PATH)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def check_auth(credentials: Optional[HTTPBasicCredentials] = Depends(security)):
    if not HUB_USER:
        return
    ok = credentials is not None and (
        secrets.compare_digest(credentials.username, HUB_USER)
        and secrets.compare_digest(credentials.password, HUB_PASS)
    )
    if not ok:
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Basic"})


Auth = Depends(check_auth)


# ---------------------------------------------------------------------------
# Default profile -- seeded from the original batch script's recipe
# ---------------------------------------------------------------------------

DEFAULT_ARGS_TEMPLATE = """
    {INPUT_FILE}
    --ensemble
    --pass1-pipeline        qwen
    --pass1-sensitivity     aggressive
    --pass1-scene-detector  semantic
    --pass1-speech-segmenter firered-vad
    --pass1-model           jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame
    --merge-strategy        pass1_primary
    --output-dir            source
    --subs-language         native
    --language              japanese
    --translate
    --translate-provider    ollama
    --translate-target      english
    --translate-tone        standard
    --translate-model       translategemma:12b
    --translate-title       {TRANSLATE_TITLE}
    --translate-actress     {TRANSLATE_ACTRESS}
    --translate-plot        {TRANSLATE_PLOT}
""".strip("\n")

DEFAULT_VIDEO_EXTENSIONS = [
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".m4v", ".ts", ".m2ts",
]


def seed_defaults() -> None:
    with db.connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM profiles").fetchone()
        if row["n"]:
            return
        now = db.now()
        conn.execute(
            """
            INSERT INTO profiles
                (name, args_template, lang_code, force, video_extensions,
                 notes, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                "qwen-galgame-translate",
                DEFAULT_ARGS_TEMPLATE,
                "qu",
                json.dumps(DEFAULT_VIDEO_EXTENSIONS),
                "Seeded from claude_whisperjav_batch.py: Qwen3ASR / aggressive / "
                "semantic / FireRedVAD, translated via ollama translategemma:12b.",
                now, now,
            ),
        )
        db.log_event(conn, "Hub started; seeded default profile")


seed_defaults()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ProfileIn(BaseModel):
    name: str
    args_template: str
    lang_code: str = "qu"
    force: bool = False
    video_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_VIDEO_EXTENSIONS))
    notes: str = ""


class JobIn(BaseModel):
    name: str = ""
    target_dir: str
    profile_id: Optional[int] = None
    args_template: Optional[str] = None      # per-job override / free-form box
    lang_code: Optional[str] = None
    force: Optional[bool] = None
    video_extensions: Optional[list[str]] = None
    priority: int = 100
    dry_run: bool = False
    start: bool = True
    agent_allowlist: list[str] = Field(default_factory=list)   # empty = any agent

    # Fired (best-effort, fire-and-forget) when a work item finishes. The hub
    # sends exactly webhook_headers/webhook_body to webhook_url as-is -- it
    # has no idea what's on the other end, which is what lets one generic
    # mechanism serve any caller (e.g. a Stash plugin's own metadataScan
    # request) without the hub knowing anything about it.
    webhook_url: str = ""
    webhook_headers: dict = Field(default_factory=dict)
    webhook_body: dict = Field(default_factory=dict)
    webhook_rel_path: str = ""                    # blank = every item in this job
    webhook_states: list[str] = Field(default_factory=lambda: ["done", "skipped"])

    # If true, a job that finishes with zero failed items is deleted rather
    # than left in the 'done' list -- for callers (e.g. a Stash plugin) that
    # submit many small one-off jobs and have no use for keeping them around.
    # A job with any failures is kept regardless, so the error is not lost.
    auto_delete_on_finish: bool = False


class JobAgents(BaseModel):
    agent_allowlist: list[str] = Field(default_factory=list)


class JobPriority(BaseModel):
    priority: int


class ItemDone(BaseModel):
    item_id: int
    state: str
    message: str = ""
    duration: float = 0.0


class ItemBeat(BaseModel):
    item_id: int
    progress: str = ""


class ItemRequeue(BaseModel):
    item_id: int
    reason: str = ""


class ScanResult(BaseModel):
    job_id: int
    files: list[str] = Field(default_factory=list)
    error: str = ""


class BrowseIn(BaseModel):
    agent_id: str
    path: str = ""


class BrowseResult(BaseModel):
    request_id: int
    entries: list[dict] = Field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Agent-facing endpoints
# ---------------------------------------------------------------------------

@app.post("/api/agent/{agent_id}/checkin")
def agent_checkin(agent_id: str, status: dict, _a=Auth):
    """
    Heartbeat + work assignment channel.  Returns any side work the agent
    should do before asking for a file: answer a folder-browse request, or
    scan a freshly created job's target directory.
    """
    now = db.now()
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO agents (agent_id, status_json, last_seen, first_seen)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                status_json = excluded.status_json,
                last_seen   = excluded.last_seen
            """,
            (agent_id, json.dumps(status), now, now),
        )

        arow = conn.execute(
            "SELECT paused, ignore_jasna FROM agents WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        paused = bool(arow["paused"]) if arow else False
        ignore_jasna = bool(arow["ignore_jasna"]) if arow else False
        hold = db.fleet_hold(conn)

        # --- Pending folder-browse request for this agent -------------------
        browse = conn.execute(
            """
            SELECT id, path FROM browse_requests
             WHERE agent_id = ? AND state = 'pending'
             ORDER BY id ASC LIMIT 1
            """,
            (agent_id,),
        ).fetchone()
        browse_task = None
        if browse:
            conn.execute(
                "UPDATE browse_requests SET state = 'sent' WHERE id = ?", (browse["id"],)
            )
            browse_task = {"request_id": browse["id"], "path": browse["path"]}

        # --- A job waiting to be scanned ------------------------------------
        # Never hand a scan to a machine that is mid-transcription: walking a
        # large tree would only slow the file it is already working on.
        scan_task = None
        if not paused and status.get("state") != "working":
            conn.execute(
                """
                UPDATE jobs SET scan_agent = NULL
                 WHERE state = 'scanning'
                   AND scan_agent IS NOT NULL
                   AND scan_started_at < ?
                """,
                (now - SCAN_TIMEOUT_SECONDS,),
            )
            waiting = conn.execute(
                """
                SELECT id, target_dir, lang_code, force, video_extensions, agent_allowlist
                  FROM jobs
                 WHERE state = 'scanning' AND scan_agent IS NULL
                 ORDER BY priority ASC, id ASC LIMIT 50
                """
            ).fetchall()
            job = next(
                (j for j in waiting if db.agent_allowed(j["agent_allowlist"], agent_id)),
                None,
            )
            if job:
                conn.execute(
                    "UPDATE jobs SET scan_agent = ?, scan_started_at = ?, updated_at = ? WHERE id = ?",
                    (agent_id, now, now, job["id"]),
                )
                scan_task = {
                    "job_id":           job["id"],
                    "target_dir":       job["target_dir"],
                    "lang_code":        job["lang_code"],
                    "force":            bool(job["force"]),
                    "video_extensions": json.loads(job["video_extensions"] or "[]"),
                }

    # `hold` is advisory here (the claim endpoint enforces it): it lets the
    # agent report itself as holding and skip claim round-trips.  Scans and
    # browses still go out -- they only read the share, and a job created
    # during a hold should be queued and ready the moment it lifts.
    return {"paused": paused, "ignore_jasna": ignore_jasna, "hold": hold,
            "browse": browse_task, "scan": scan_task}


@app.post("/api/agent/{agent_id}/claim")
def agent_claim(agent_id: str, _a=Auth):
    """Hand this agent exactly one work item, or nothing.  See db.claim_next."""
    with db.connect() as conn:
        row = conn.execute("SELECT paused FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
        if row and row["paused"]:
            return {"item": None, "reason": "agent paused"}
        hold = db.fleet_hold(conn)
        if hold:
            return {"item": None, "reason": f"fleet hold: {hold['reason'] or 'no reason given'}"}

    item = db.claim_next(agent_id)
    if item:
        with db.connect() as conn:
            db.log_event(
                conn, f"claimed {item['rel_path']}",
                agent_id=agent_id, job_id=item["job_id"],
            )
    return {"item": item}


@app.post("/api/agent/{agent_id}/heartbeat")
def agent_heartbeat(agent_id: str, beat: ItemBeat, _a=Auth):
    """
    Extend the claim lease.  Also the channel that tells a busy agent to give
    up: `abort` comes back true if the job was paused/stopped from the
    dashboard, or if this specific agent was Stopped -- either way the agent
    kills the in-flight whisperjav process.
    """
    with db.connect() as conn:
        # One-shot: consume the Stop request the instant it's seen, so it
        # can't also abort whatever this agent claims next after resuming.
        cur = conn.execute(
            "UPDATE agents SET abort_requested = 0 WHERE agent_id = ? AND abort_requested = 1",
            (agent_id,),
        )
        stopped = cur.rowcount > 0

        job = conn.execute(
            """
            SELECT j.state FROM jobs j
              JOIN work_items wi ON wi.job_id = j.id
             WHERE wi.id = ?
            """,
            (beat.item_id,),
        ).fetchone()
        if job is None:
            return {"ok": False, "abort": True, "reason": "unknown item"}
        if stopped:
            return {"ok": True, "abort": True, "reason": "stopped from dashboard"}
        if job["state"] != "running":
            return {"ok": True, "abort": True, "reason": f"job is {job['state']}"}
        hold = db.fleet_hold(conn)
        if hold and hold["abort_in_flight"]:
            # The fleet is held with "abort in flight": the file is put back
            # (agent requeues it) and cannot be reclaimed until the hold lifts.
            return {"ok": True, "abort": True, "reason": f"fleet hold: {hold['reason']}"}

        ok = db.heartbeat_item(conn, beat.item_id, agent_id, beat.progress)
        if not ok:
            # Our claim was taken from us -- stop working on this file.
            return {"ok": False, "abort": True, "reason": "claim lost"}
    return {"ok": True, "abort": False, "reason": ""}


def _fire_webhook(url: str, headers: dict, body: dict, job_id: Optional[int]) -> None:
    """
    Best-effort, off-thread. The hub has no idea what this request does --
    it's a generic notification mechanism, not a Stash (or anything-else)
    integration -- and a slow or dead target must never delay or fail the
    agent's /complete call, which is why this never runs inline and never
    raises. It does *record* the outcome on the job's events, though: a
    webhook that nobody answers (wrong address, expired key) used to vanish
    without a trace, and the job with it when auto_delete was on.
    """
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", **headers},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok, detail = True, f"{url} -> HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        snippet = exc.read(200).decode(errors="replace").strip()
        ok, detail = False, f"{url} -> HTTP {exc.code}" + (f" {snippet}" if snippet else "")
    except Exception as exc:
        ok, detail = False, f"{url} -> {exc}"
    if job_id is None:
        return
    try:
        with db.connect() as conn:
            db.record_webhook(conn, job_id, ok, detail)
    except Exception:
        pass


def _maybe_fire_webhook(row: Any, state: str, rel_path: str | None = "",
                        job_id: Optional[int] = None) -> None:
    """Fire the job's webhook for an item outcome.  `rel_path` is the item
    that finished (defaults to the row's own); None means the whole job
    finished with no items, which bypasses the per-file filter."""
    url = row["webhook_url"]
    if not url:
        return
    if rel_path is not None:
        filter_path = row["webhook_rel_path"]
        if filter_path and filter_path != (rel_path or row["rel_path"]):
            return
    states = json.loads(row["webhook_states"] or '["done", "skipped"]')
    if state not in states:
        return
    headers = json.loads(row["webhook_headers"] or "{}")
    body = json.loads(row["webhook_body"] or "{}")
    threading.Thread(target=_fire_webhook, args=(url, headers, body, job_id), daemon=True).start()


@app.post("/api/agent/{agent_id}/complete")
def agent_complete(agent_id: str, res: ItemDone, _a=Auth):
    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT wi.job_id, wi.rel_path,
                   j.webhook_url, j.webhook_headers, j.webhook_body,
                   j.webhook_rel_path, j.webhook_states
              FROM work_items wi
              JOIN jobs j ON j.id = wi.job_id
             WHERE wi.id = ?
            """,
            (res.item_id,),
        ).fetchone()
        ok = db.complete_item(conn, res.item_id, agent_id, res.state, res.message, res.duration)
        if ok and row:
            level = "error" if res.state == "failed" else "info"
            db.log_event(
                conn, f"{res.state}: {row['rel_path']}" + (f" -- {res.message}" if res.message else ""),
                level=level, agent_id=agent_id, job_id=row["job_id"],
            )
            db.maybe_finish_job(conn, row["job_id"])
            _maybe_fire_webhook(row, res.state, job_id=row["job_id"])
    return {"ok": ok}


@app.post("/api/agent/{agent_id}/requeue")
def agent_requeue(agent_id: str, req: ItemRequeue, _a=Auth):
    with db.connect() as conn:
        ok = db.requeue_item(conn, req.item_id, agent_id, req.reason)
    return {"ok": ok}


@app.post("/api/agent/{agent_id}/scan_result")
def agent_scan_result(agent_id: str, res: ScanResult, _a=Auth):
    """
    Take the file list an agent produced for a job and turn it into work
    items.  INSERT OR IGNORE against the (job_id, rel_path) unique index makes
    a re-scan idempotent, so a rescan after a timeout cannot duplicate work.
    """
    now = db.now()
    with db.connect() as conn:
        job = conn.execute("SELECT id, state FROM jobs WHERE id = ?", (res.job_id,)).fetchone()
        if job is None:
            raise HTTPException(404, "no such job")

        if res.error:
            conn.execute(
                "UPDATE jobs SET state = 'failed', error = ?, scan_agent = NULL, updated_at = ? WHERE id = ?",
                (res.error, now, res.job_id),
            )
            db.log_event(conn, f"Scan failed: {res.error}", level="error",
                         agent_id=agent_id, job_id=res.job_id)
            return {"ok": False, "added": 0}

        added = 0
        for rel in res.files:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO work_items (job_id, rel_path, state, updated_at)
                VALUES (?, ?, 'pending', ?)
                """,
                (res.job_id, rel, now),
            )
            added += cur.rowcount or 0

        conn.execute(
            "UPDATE jobs SET state = 'running', scan_agent = NULL, updated_at = ? WHERE id = ?",
            (now, res.job_id),
        )
        db.log_event(conn, f"Scan complete: {len(res.files)} file(s) found, {added} queued",
                     agent_id=agent_id, job_id=res.job_id)
        if not res.files:
            # Nothing to queue: every video under the target already has its
            # SRT (the agent filters those out at scan time).  Had they been
            # queued, each would have finished as 'skipped' and fired the
            # webhook, so give the caller that same notification once here
            # -- the Stash plugin relies on it to rescan for captions.
            db.log_event(conn, "Nothing to do: every video already has its subtitle",
                         agent_id=agent_id, job_id=res.job_id)
            wh = conn.execute(
                """
                SELECT webhook_url, webhook_headers, webhook_body, webhook_states
                  FROM jobs WHERE id = ?
                """,
                (res.job_id,),
            ).fetchone()
            _maybe_fire_webhook(wh, "skipped", rel_path=None, job_id=res.job_id)
        db.maybe_finish_job(conn, res.job_id)
    return {"ok": True, "added": added}


@app.post("/api/agent/{agent_id}/browse_result")
def agent_browse_result(agent_id: str, res: BrowseResult, _a=Auth):
    with db.connect() as conn:
        conn.execute(
            "UPDATE browse_requests SET state = ?, result_json = ? WHERE id = ?",
            (
                "error" if res.error else "done",
                json.dumps({"entries": res.entries, "error": res.error}),
                res.request_id,
            ),
        )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Dashboard-facing endpoints
# ---------------------------------------------------------------------------

@app.get("/api/state")
def get_state(_a=Auth):
    """Everything the dashboard needs, in one poll."""
    now = db.now()
    with db.connect() as conn:
        db.release_expired_claims(conn)
        hold = db.fleet_hold(conn)

        # Per-agent transcription time, start to finish, over every file it
        # completed that is still in the database (deleting a job drops its
        # items).  `duration` is measured by the agent itself around the
        # whisperjav run.  Both the all-time mean and the mean of the last
        # ten, since file lengths and settings drift over time.
        timing: dict[str, dict] = {}
        for r in conn.execute(
            """
            SELECT claimed_by, AVG(duration) AS avg, COUNT(*) AS n
              FROM work_items
             WHERE state = 'done' AND duration > 0 AND claimed_by IS NOT NULL
             GROUP BY claimed_by
            """
        ):
            timing[r["claimed_by"]] = {"avg_duration": round(r["avg"]), "done_total": r["n"]}
        for agent_id, t in timing.items():
            recent = [r["duration"] for r in conn.execute(
                """
                SELECT duration FROM work_items
                 WHERE state = 'done' AND duration > 0 AND claimed_by = ?
                 ORDER BY finished_at DESC LIMIT 10
                """,
                (agent_id,),
            )]
            t["avg_recent"] = round(sum(recent) / len(recent)) if recent else None
            t["recent_n"] = len(recent)

        agents = []
        for row in conn.execute("SELECT * FROM agents ORDER BY agent_id"):
            status = json.loads(row["status_json"] or "{}")
            # How long whisperjav has been on the agent's current file, from
            # the hub's claim clock (the dashboard's "This file" row).
            started = db.current_item_started(conn, row["agent_id"])
            agents.append({
                "agent_id": row["agent_id"],
                "online":   (now - row["last_seen"]) < db.AGENT_OFFLINE_SECONDS,
                "paused":   bool(row["paused"]),
                "ignore_jasna": bool(row["ignore_jasna"]),
                "last_seen_ago": round(now - row["last_seen"], 1),
                "status":   status,
                "timing":   timing.get(row["agent_id"]),
                "file_elapsed": round(now - started, 1) if started is not None else None,
            })

        jobs = []
        for row in conn.execute("SELECT * FROM jobs ORDER BY id DESC"):
            job = dict(row)
            job["force"]   = bool(job["force"])
            job["dry_run"] = bool(job["dry_run"])
            job["video_extensions"] = json.loads(job["video_extensions"] or "[]")
            job["agent_allowlist"] = json.loads(job["agent_allowlist"] or "[]")
            job["webhook_headers"] = json.loads(job["webhook_headers"] or "{}")
            job["webhook_body"] = json.loads(job["webhook_body"] or "{}")
            job["webhook_states"] = json.loads(job["webhook_states"] or "[]")
            job["auto_delete_on_finish"] = bool(job["auto_delete_on_finish"])
            job["counts"] = db.job_counts(conn, row["id"])
            job["active"] = [
                {
                    "rel_path":   r["rel_path"],
                    "claimed_by": r["claimed_by"],
                    "progress":   r["progress"],
                    "elapsed":    round(now - r["started_at"], 1) if r["started_at"] else 0,
                }
                for r in conn.execute(
                    """
                    SELECT rel_path, claimed_by, progress, started_at
                      FROM work_items
                     WHERE job_id = ? AND state IN ('claimed', 'running')
                     ORDER BY rel_path
                    """,
                    (row["id"],),
                )
            ]
            jobs.append(job)

        events = [dict(r) for r in conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT 100"
        )]

        profiles = []
        for r in conn.execute("SELECT * FROM profiles ORDER BY name"):
            p = dict(r)
            p["force"] = bool(p["force"])
            p["video_extensions"] = json.loads(p["video_extensions"] or "[]")
            profiles.append(p)

        hold_window = db.hold_window(conn)

    return {"now": now, "hold": hold, "hold_window": hold_window, "agents": agents, "jobs": jobs,
            "events": events, "profiles": profiles}


@app.get("/api/jobs/{job_id}/items")
def get_job_items(job_id: int, state: str = "", limit: int = 500, _a=Auth):
    with db.connect() as conn:
        if state:
            rows = conn.execute(
                "SELECT * FROM work_items WHERE job_id = ? AND state = ? ORDER BY rel_path LIMIT ?",
                (job_id, state, limit),
            )
        else:
            rows = conn.execute(
                "SELECT * FROM work_items WHERE job_id = ? ORDER BY rel_path LIMIT ?",
                (job_id, limit),
            )
        return {"items": [dict(r) for r in rows]}


@app.post("/api/jobs")
def create_job(job: JobIn, _a=Auth):
    """
    Create a job.  The args template is *snapshotted* onto the job, so editing
    a profile later never changes the behaviour of a job already queued.
    """
    now = db.now()
    with db.connect() as conn:
        prof = None
        if job.profile_id is not None:
            prof = conn.execute("SELECT * FROM profiles WHERE id = ?", (job.profile_id,)).fetchone()
            if prof is None:
                raise HTTPException(404, "no such profile")

        args_template = job.args_template if job.args_template is not None else (
            prof["args_template"] if prof else DEFAULT_ARGS_TEMPLATE
        )
        lang_code = job.lang_code if job.lang_code is not None else (
            prof["lang_code"] if prof else "qu"
        )
        force = job.force if job.force is not None else (
            bool(prof["force"]) if prof else False
        )
        exts = job.video_extensions if job.video_extensions is not None else (
            json.loads(prof["video_extensions"]) if prof else list(DEFAULT_VIDEO_EXTENSIONS)
        )

        if "{INPUT_FILE}" not in args_template:
            raise HTTPException(400, "args template must contain the {INPUT_FILE} placeholder")

        target = job.target_dir.strip().strip('"').strip("'")
        if not target:
            raise HTTPException(400, "target_dir is required")

        name = job.name.strip() or target.replace("\\", "/").rstrip("/").split("/")[-1] or target

        cur = conn.execute(
            """
            INSERT INTO jobs
                (name, target_dir, profile_id, profile_name, args_template, lang_code,
                 force, video_extensions, state, priority, dry_run, agent_allowlist,
                 webhook_url, webhook_headers, webhook_body, webhook_rel_path, webhook_states,
                 auto_delete_on_finish, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name, target, job.profile_id, prof["name"] if prof else "",
                args_template, lang_code, int(force), json.dumps(exts),
                "scanning" if job.start else "draft",
                job.priority, int(job.dry_run), json.dumps(job.agent_allowlist),
                job.webhook_url, json.dumps(job.webhook_headers), json.dumps(job.webhook_body),
                job.webhook_rel_path, json.dumps(job.webhook_states),
                int(job.auto_delete_on_finish),
                now, now,
            ),
        )
        job_id = cur.lastrowid
        db.log_event(conn, f"Job created: {name} -> {target}", job_id=job_id)
    return {"ok": True, "job_id": job_id}


@app.post("/api/jobs/{job_id}/agents")
def set_job_agents(job_id: int, body: JobAgents, _a=Auth):
    """Restrict (or unrestrict) which agents may claim this job's items."""
    now = db.now()
    with db.connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET agent_allowlist = ?, updated_at = ? WHERE id = ?",
            (json.dumps(body.agent_allowlist), now, job_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "no such job")
        db.log_event(
            conn,
            "Agents restricted to: " + (", ".join(body.agent_allowlist) or "any"),
            job_id=job_id,
        )
    return {"ok": True}


@app.post("/api/jobs/{job_id}/priority")
def set_job_priority(job_id: int, body: JobPriority, _a=Auth):
    """
    Re-prioritise a job that is already queued or running.

    Takes effect on the next claim, not the current one: an agent keeps the
    file it is working on and picks up the new ordering when it next asks for
    work.  Lower sorts first; see db.claim_next.
    """
    now = db.now()
    with db.connect() as conn:
        row = conn.execute("SELECT priority FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such job")
        if row["priority"] == body.priority:
            return {"ok": True, "priority": body.priority}
        conn.execute(
            "UPDATE jobs SET priority = ?, updated_at = ? WHERE id = ?",
            (body.priority, now, job_id),
        )
        db.log_event(
            conn, f"Priority changed: {row['priority']} -> {body.priority}", job_id=job_id
        )
    return {"ok": True, "priority": body.priority}


@app.post("/api/jobs/{job_id}/action")
def job_action(job_id: int, body: dict, _a=Auth):
    """
    start   -- (re)scan the target folder, then run
    resume  -- continue without rescanning
    pause   -- stop handing out work; in-flight files are abandoned and requeued
    stop    -- same as pause, and requeues everything not finished
    retry   -- move failed items back to pending
    reset   -- clear all item results and start over
    """
    action = (body or {}).get("action", "")
    now = db.now()
    with db.connect() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if job is None:
            raise HTTPException(404, "no such job")

        if action == "start":
            conn.execute(
                "UPDATE jobs SET state = 'scanning', scan_agent = NULL, error = '', updated_at = ? WHERE id = ?",
                (now, job_id),
            )
        elif action == "resume":
            conn.execute("UPDATE jobs SET state = 'running', updated_at = ? WHERE id = ?", (now, job_id))
        elif action in ("pause", "stop"):
            new_state = "paused" if action == "pause" else "stopped"
            conn.execute("UPDATE jobs SET state = ?, updated_at = ? WHERE id = ?", (new_state, now, job_id))
            # Release in-flight claims; the agents abort via their heartbeat.
            conn.execute(
                """
                UPDATE work_items
                   SET state = 'pending', claimed_by = NULL, claim_expires = 0,
                       progress = '', updated_at = ?
                 WHERE job_id = ? AND state IN ('claimed', 'running')
                """,
                (now, job_id),
            )
        elif action == "retry":
            conn.execute(
                """
                UPDATE work_items
                   SET state = 'pending', claimed_by = NULL, claim_expires = 0,
                       last_error = '', updated_at = ?
                 WHERE job_id = ? AND state = 'failed'
                """,
                (now, job_id),
            )
            conn.execute("UPDATE jobs SET state = 'running', updated_at = ? WHERE id = ?", (now, job_id))
        elif action == "reset":
            conn.execute("DELETE FROM work_items WHERE job_id = ?", (job_id,))
            conn.execute(
                "UPDATE jobs SET state = 'draft', scan_agent = NULL, error = '', updated_at = ? WHERE id = ?",
                (now, job_id),
            )
        else:
            raise HTTPException(400, f"unknown action: {action}")

        db.log_event(conn, f"Job action: {action}", job_id=job_id)
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: int, _a=Auth):
    with db.connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        db.log_event(conn, f"Job {job_id} deleted")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Fleet hold
# ---------------------------------------------------------------------------

class HoldIn(BaseModel):
    held: bool = True
    reason: str = ""
    ttl_seconds: Optional[float] = None
    set_by: str = ""
    abort_in_flight: bool = False   # also abort (requeue) files in progress
    extend_only: bool = False       # never move an existing deadline earlier


class HoldWindowIn(BaseModel):
    enabled: bool
    start: str = "02:55"            # HH:MM, hub local time
    minutes: int = 65
    reason: str = "maintenance window"
    abort_in_flight: bool = True


@app.get("/api/hold")
def get_hold(_a=Auth):
    """
    The effective hold, plus `busy`: the online agents still reporting a
    file in progress.  A caller that has just set an aborting hold (the
    SnapRAID hook) polls this until `busy` is empty before letting its own
    task start, so the array is actually quiet -- not merely told to be --
    by the time SnapRAID begins reading it.
    """
    now = db.now()
    with db.connect() as conn:
        hold = db.fleet_hold(conn)
        busy = []
        for row in conn.execute("SELECT agent_id, status_json, last_seen FROM agents"):
            if (now - row["last_seen"]) >= db.AGENT_OFFLINE_SECONDS:
                continue
            status = json.loads(row["status_json"] or "{}")
            if status.get("state") == "working" or (status.get("current") or {}).get("file"):
                busy.append(row["agent_id"])
    return {"hold": hold, "busy": sorted(busy)}


@app.post("/api/hold")
def set_hold(body: HoldIn, _a=Auth):
    """
    Fleet-wide hold: stop every agent from claiming new files, and with
    `abort_in_flight` also abort and requeue the ones in progress (nothing
    partial is ever written, so that is always safe).  Built for external
    events that want the array quiet -- the bundled contrib/snapraid-hook.sh
    sets it when a SnapRAID task begins and extends it past the last one --
    and for a manual hold from the dashboard.  The daily scheduled window
    (/api/hold/window) is the other source; fleet_hold() merges the two.

    Independent of per-agent Pause on purpose: releasing the hold never
    un-pauses an agent someone paused by hand, and vice versa.

    The hold always carries a deadline (`ttl_seconds`, default
    HOLD_DEFAULT_TTL_SECONDS).  That is the fail-open half of the contract:
    if the releaser never calls back, the fleet resumes on its own.
    Setting an active hold again just moves the deadline.
    """
    with db.connect() as conn:
        if body.held:
            ttl = body.ttl_seconds if body.ttl_seconds is not None else HOLD_DEFAULT_TTL_SECONDS
            if not (0 < ttl <= HOLD_MAX_TTL_SECONDS):
                raise HTTPException(400, f"ttl_seconds must be between 1 and {HOLD_MAX_TTL_SECONDS}")
            before = db.fleet_hold(conn)
            hold = db.set_fleet_hold(conn, body.reason.strip(), ttl, body.set_by.strip(),
                                     abort_in_flight=body.abort_in_flight,
                                     extend_only=body.extend_only)
            if before and before["until"] >= hold["until"] and before["reason"] == hold["reason"]:
                pass    # extend_only against a longer hold: nothing changed worth logging
            else:
                span = (f"{ttl / 3600:g} h" if ttl >= 3600 else
                        f"{int(ttl // 60)} min" if ttl >= 60 else f"{int(ttl)} s")
                db.log_event(conn, (f"Fleet hold {'extended' if before else 'set'}: "
                                    f"{body.reason.strip() or 'no reason given'} "
                                    f"(lifts in {span} unless released"
                                    f"{'; aborting files in progress' if body.abort_in_flight else ''})"),
                             level="warn")
        else:
            released = db.clear_fleet_hold(conn)
            if "window" in released:
                db.log_event(conn, "Fleet hold released; today's maintenance window skipped")
            elif released:
                db.log_event(conn, "Fleet hold released")
            hold = db.fleet_hold(conn)
    return {"ok": True, "hold": hold}


@app.get("/api/hold/window")
def get_hold_window(_a=Auth):
    with db.connect() as conn:
        return {"window": db.hold_window(conn)}


@app.post("/api/hold/window")
def set_hold_window(body: HoldWindowIn, _a=Auth):
    """
    A daily hold window in the hub's local time: the belt to the hook
    script's braces.  snapraidd only fires hooks around sync/scrub/diff/
    check/fix, not around spin-up, spin-down, probe or report, so a fixed
    window is the one way to cover the whole maintenance sequence.  Saving
    the schedule also forgets any "skip today" the Release button recorded.
    """
    with db.connect() as conn:
        try:
            win = db.set_hold_window(conn, enabled=body.enabled, start=body.start, minutes=body.minutes,
                                     reason=body.reason, abort_in_flight=body.abort_in_flight)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if win["enabled"]:
            db.log_event(conn, (f"Maintenance window set: daily {win['start']} for {win['minutes']} min "
                                f"({win['hub_tz']}); {win['reason']}"
                                f"{'; aborting files in progress' if win['abort_in_flight'] else ''}"))
        else:
            db.log_event(conn, "Maintenance window disabled")
    return {"ok": True, "window": win}


@app.post("/api/agents/{agent_id}/pause")
def pause_agent(agent_id: str, body: dict, _a=Auth):
    paused = bool((body or {}).get("paused", True))
    with db.connect() as conn:
        if paused:
            conn.execute("UPDATE agents SET paused = 1 WHERE agent_id = ?", (agent_id,))
        else:
            # Resuming always clears any stale Stop request: if the agent was
            # idle when Stopped, there was no in-flight heartbeat to consume
            # it, and it must not fire against whatever this agent claims next.
            conn.execute(
                "UPDATE agents SET paused = 0, abort_requested = 0 WHERE agent_id = ?",
                (agent_id,),
            )
        db.log_event(conn, f"Agent {'paused' if paused else 'resumed'}", agent_id=agent_id)
    return {"ok": True}


@app.post("/api/agents/{agent_id}/ignore_jasna")
def set_agent_ignore_jasna(agent_id: str, body: dict, _a=Auth):
    """Per-agent override: when on, this agent ignores Jasna GPU state and
    keeps claiming/working even while Jasna is active. Off (default) yields."""
    ignore = bool((body or {}).get("ignore", True))
    with db.connect() as conn:
        conn.execute("UPDATE agents SET ignore_jasna = ? WHERE agent_id = ?",
                     (1 if ignore else 0, agent_id))
        db.log_event(conn, f"Agent set to {'ignore' if ignore else 'respect'} Jasna GPU state",
                     agent_id=agent_id)
    return {"ok": True}


@app.post("/api/agents/{agent_id}/stop")
def stop_agent(agent_id: str, _a=Auth):
    """
    Pause this agent and abort whatever it is working on right now, rather
    than letting the current file finish first -- e.g. to reclaim its GPU
    for something else immediately.
    """
    with db.connect() as conn:
        cur = conn.execute(
            "UPDATE agents SET paused = 1, abort_requested = 1 WHERE agent_id = ?",
            (agent_id,),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "no such agent")
        db.log_event(conn, "Stop requested (immediate)", agent_id=agent_id)
    return {"ok": True}


@app.delete("/api/agents/{agent_id}")
def forget_agent(agent_id: str, _a=Auth):
    with db.connect() as conn:
        conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
    return {"ok": True}


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@app.post("/api/profiles")
def create_profile(p: ProfileIn, _a=Auth):
    now = db.now()
    with db.connect() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO profiles
                    (name, args_template, lang_code, force, video_extensions, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (p.name, p.args_template, p.lang_code, int(p.force),
                 json.dumps(p.video_extensions), p.notes, now, now),
            )
        except Exception as exc:
            raise HTTPException(400, f"could not create profile: {exc}")
    return {"ok": True, "id": cur.lastrowid}


@app.put("/api/profiles/{profile_id}")
def update_profile(profile_id: int, p: ProfileIn, _a=Auth):
    with db.connect() as conn:
        cur = conn.execute(
            """
            UPDATE profiles
               SET name = ?, args_template = ?, lang_code = ?, force = ?,
                   video_extensions = ?, notes = ?, updated_at = ?
             WHERE id = ?
            """,
            (p.name, p.args_template, p.lang_code, int(p.force),
             json.dumps(p.video_extensions), p.notes, db.now(), profile_id),
        )
        if not cur.rowcount:
            raise HTTPException(404, "no such profile")
    return {"ok": True}


@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: int, _a=Auth):
    with db.connect() as conn:
        conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
    return {"ok": True}


# ---------------------------------------------------------------------------
# Folder browsing (proxied through an agent, since the hub has no mount)
# ---------------------------------------------------------------------------

@app.post("/api/browse")
def browse(req: BrowseIn, _a=Auth):
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO browse_requests (agent_id, path, state, created_at) VALUES (?, ?, 'pending', ?)",
            (req.agent_id, req.path, db.now()),
        )
        # Keep this table small; old answers are of no interest.
        conn.execute("DELETE FROM browse_requests WHERE id < (SELECT MAX(id) - 50 FROM browse_requests)")
    return {"request_id": cur.lastrowid}


@app.get("/api/browse/{request_id}")
def browse_result(request_id: int, _a=Auth):
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM browse_requests WHERE id = ?", (request_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such request")
        payload = json.loads(row["result_json"] or "{}")
        return {"state": row["state"], "path": row["path"], **payload}


# ---------------------------------------------------------------------------
# Static dashboard
# ---------------------------------------------------------------------------

@app.get("/")
def index(_a=Auth):
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True})
