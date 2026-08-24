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
            "SELECT paused FROM agents WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        paused = bool(arow["paused"]) if arow else False

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

    return {"paused": paused, "browse": browse_task, "scan": scan_task}


@app.post("/api/agent/{agent_id}/claim")
def agent_claim(agent_id: str, _a=Auth):
    """Hand this agent exactly one work item, or nothing.  See db.claim_next."""
    with db.connect() as conn:
        row = conn.execute("SELECT paused FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
        if row and row["paused"]:
            return {"item": None, "reason": "agent paused"}

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
    up: if the job was paused or stopped from the dashboard, `abort` comes
    back true and the agent kills the in-flight whisperjav process.
    """
    with db.connect() as conn:
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
        if job["state"] != "running":
            return {"ok": True, "abort": True, "reason": f"job is {job['state']}"}

        ok = db.heartbeat_item(conn, beat.item_id, agent_id, beat.progress)
        if not ok:
            # Our claim was taken from us -- stop working on this file.
            return {"ok": False, "abort": True, "reason": "claim lost"}
    return {"ok": True, "abort": False, "reason": ""}


@app.post("/api/agent/{agent_id}/complete")
def agent_complete(agent_id: str, res: ItemDone, _a=Auth):
    with db.connect() as conn:
        row = conn.execute("SELECT job_id, rel_path FROM work_items WHERE id = ?", (res.item_id,)).fetchone()
        ok = db.complete_item(conn, res.item_id, agent_id, res.state, res.message, res.duration)
        if ok and row:
            level = "error" if res.state == "failed" else "info"
            db.log_event(
                conn, f"{res.state}: {row['rel_path']}" + (f" -- {res.message}" if res.message else ""),
                level=level, agent_id=agent_id, job_id=row["job_id"],
            )
            db.maybe_finish_job(conn, row["job_id"])
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

        agents = []
        for row in conn.execute("SELECT * FROM agents ORDER BY agent_id"):
            status = json.loads(row["status_json"] or "{}")
            agents.append({
                "agent_id": row["agent_id"],
                "online":   (now - row["last_seen"]) < db.AGENT_OFFLINE_SECONDS,
                "paused":   bool(row["paused"]),
                "last_seen_ago": round(now - row["last_seen"], 1),
                "status":   status,
            })

        jobs = []
        for row in conn.execute("SELECT * FROM jobs ORDER BY id DESC"):
            job = dict(row)
            job["force"]   = bool(job["force"])
            job["dry_run"] = bool(job["dry_run"])
            job["video_extensions"] = json.loads(job["video_extensions"] or "[]")
            job["agent_allowlist"] = json.loads(job["agent_allowlist"] or "[]")
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

    return {"now": now, "agents": agents, "jobs": jobs, "events": events, "profiles": profiles}


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
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name, target, job.profile_id, prof["name"] if prof else "",
                args_template, lang_code, int(force), json.dumps(exts),
                "scanning" if job.start else "draft",
                job.priority, int(job.dry_run), json.dumps(job.agent_allowlist),
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


@app.post("/api/agents/{agent_id}/pause")
def pause_agent(agent_id: str, body: dict, _a=Auth):
    paused = bool((body or {}).get("paused", True))
    with db.connect() as conn:
        conn.execute("UPDATE agents SET paused = ? WHERE agent_id = ?", (int(paused), agent_id))
        db.log_event(conn, f"Agent {'paused' if paused else 'resumed'}", agent_id=agent_id)
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
