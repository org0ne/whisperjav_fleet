# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A distributed job runner for WhisperJAV. It wraps the transcription behaviour of
`../whisperjav_batch/claude_whisperjav_batch.py` (NFO parsing, argument
templating, output naming, audit logs) in a hub/agent system so several GPU
machines can work one shared media directory without ever processing the same
video twice.

Adapted from that script, so **behavioural parity with it is a goal**. If you
change how a file is transcribed, named, or logged, check it against that script
first — divergence should be deliberate and noted in the README (there is
currently one, see "NFO selection" below).

## Commands

There is no build step, no linter config, and no automated test suite.

```bash
# Hub (needs the venv; FastAPI + uvicorn)
cd hub && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8787
HUB_DB_PATH=/tmp/scratch.sqlite3 .venv/bin/uvicorn app:app --port 8799   # throwaway DB

# Agent (stdlib only — no venv, no install)
python3 agent/agent.py --config /etc/whisperjav-agent/config.json
python3 agent/agent.py --hub http://127.0.0.1:8799 --agent-id test --once   # one poll cycle

# Syntax / import check
python3 -m py_compile hub/app.py hub/db.py agent/agent.py agent/runner.py
```

`HUB_USER` / `HUB_PASS` enable HTTP Basic on the hub (off when unset). Agent
settings also read from `FLEET_HUB_URL`, `FLEET_AGENT_ID`, `FLEET_HUB_USER`,
`FLEET_HUB_PASS`, `WHISPERJAV_BIN`.

## Manual verification

The only practical way to exercise the concurrency logic is a fake `whisperjav`
on `WHISPERJAV_BIN`: a shell script that answers `--version`, appends its `$1` to
a shared log under `flock`, sleeps, then writes `<basename>.ja.pass1.english.srt`.
Point two agents with different `--agent-id` at one hub and one media tree; the
invariant is **line count == distinct-file count** in that log. Vary it to cover
the interesting paths: a long sleep to test Pause mid-file, a pre-planted
`.whisper-lock` to test the lock guard, a background grandchild to test that the
process group is fully reaped.

## Architecture

```
browser → hub (SQLite queue, no filesystem)  ←── outbound HTTP only ──  agents (GPU + media mount)
```

**The hub must never touch the media filesystem.** It is designed to run in a
small LXC with no CIFS mount and no GPU. Scanning, directory browsing, and all
reads/writes happen on agents and are reported back — that is why the folder
picker is a request parked in `browse_requests` and answered on an agent's next
poll, rather than a plain `os.listdir` on the server. Preserve this constraint.

Agents only make outbound requests, so workers need no inbound ports.

The hub itself is otherwise receive-only, with one narrow exception: an
optional per-job webhook (`jobs.webhook_*`, see "Job and item semantics")
lets it fire a single opaque, best-effort outbound POST on item completion,
for external integrations. It carries no other knowledge of what's on the
other end.

### Preventing double-processing

Four independent layers. Understand all four before touching any of them:

1. **`db.claim_next()`** — the primary guard, in `hub/db.py`. One
   `BEGIN IMMEDIATE` transaction; the mark-as-claimed is a compare-and-swap
   (`UPDATE … WHERE id = ? AND state = 'pending'`) and a zero-rowcount result
   means another agent won, so it retries. SQLite's single-writer rule is what
   makes this safe. **Do not** split the select and update into separate
   transactions, or add a second concurrent writer path.
2. **Lease + heartbeat** — `claim_expires`; `release_expired_claims()` re-queues
   lapsed items. `heartbeat_item`/`complete_item`/`requeue_item` are all scoped
   `AND claimed_by = ?` so an agent that lost its claim cannot resurrect it.
3. **`runner.VideoLock`** — `<basename>.whisper-lock` via `O_CREAT|O_EXCL`
   (atomic on CIFS). Covers what the hub cannot see: the old batch script run by
   hand, a second hub, a reset DB. Stale after 6h.
4. **Last-moment output re-check** in `agent._process`, done twice: before
   locking and again after, because the other machine may have finished in
   between.

`runner.publish_srt` writes to a temp file and `os.replace`s it, so the other
machine's skip check can never see a half-written SRT.

### Agent threading

Three concerns, deliberately separated in `agent/agent.py`:

- **Presence thread** (`_presence_loop`) — the *only* caller of `checkin`. It runs
  independently of work because a single video can far exceed the hub's 30s
  offline threshold; when check-ins only happened between files, a busy machine
  showed as idle and then offline. It stashes `_paused` / `_pending_scan` for the
  work loop.
- **Work loop** (`run`) — scan, claim, process. Never calls `checkin`.
- **Per-item heartbeat thread** (inside `_run_one`) — refreshes the lease, touches
  the lock's mtime, and watches for `abort` in the heartbeat response. `abort` is
  how a dashboard Pause/Stop reaches an in-flight transcription — either because
  the *job* was paused/stopped, or because this specific *agent* was Stopped
  (see below). Either way `runner.run_whisperjav`'s `should_abort` check picks
  it up and kills the process group; `_run_one` treats an aborted run as a
  clean requeue, not a failure.

  This reaches everything whisperjav actually spawned, but a translate
  backend it talks to over HTTP (`--translate-provider ollama`) is a
  pre-existing daemon, not a child of that process group — killing the group
  never touches it, and Ollama keeps the model resident in VRAM regardless.
  `Agent._maybe_unload_ollama` is the opt-in mitigation: gated on
  `ollama_unload_on_abort`, it scans the already-built `wj_args` for
  `--translate-provider ollama` and its paired `--translate-model`, then
  POSTs `{"model": ..., "keep_alive": 0}` to `{ollama_url}/api/generate` —
  Ollama's documented way to unload a model without running inference. Pure
  best-effort: any failure is logged and swallowed, never allowed to affect
  the (already-aborted) transcription's outcome. Lives in `agent.py`, not
  `runner.py`, which is deliberately HTTP-free.

`runner._terminate_tree` kills the child's whole **process group** (children are
spawned with `start_new_session=True`), because WhisperJAV spawns helpers that
would otherwise be orphaned holding GPU memory. It captures the pgid *before*
`wait()` — once the child is reaped, `os.getpgid(proc.pid)` raises and the final
sweep would silently no-op.

### Job and item semantics

- Job states: `draft → scanning → running → paused|stopped|done|failed`. Only
  `running` jobs yield claims. Items: `pending → claimed → running →
  done|failed|skipped`.
- **Priority is strict, not weighted.** Claims are ordered
  `priority ASC, job id ASC, item id ASC`, so every pending item of a
  lower-numbered job is handed out before any of a higher-numbered one —
  a high-priority job starves the rest until its queue drains, rather than
  sharing agents proportionally. Equal numbers interleave by job id. Set at
  creation and changed later via `POST /api/jobs/{id}/priority`; it takes
  effect on the next claim, with no preemption of the file already running.
  The dashboard's `Priority…` dialog previews the resulting queue order.
- **Args are snapshotted onto the job at creation.** Profiles are only starting
  points; editing a profile must never change a queued job.
- `work_items.rel_path` is POSIX-style and relative to the job's `target_dir`, so
  the queue is portable across machines. Agents resolve it locally via
  `Agent.local_path` (`path_map` prefix rewrites; empty when all machines mount
  the share identically).
- Scan results use `INSERT OR IGNORE` on `(job_id, rel_path)`, making rescans
  idempotent — a scan reassigned after timeout cannot duplicate work.
- Terminal-state meanings matter: **`skipped`** = deliberately not processed
  (output exists, lock held, source gone, dry run); **`failed`** = tried and
  errored, writes a `.whisper-failed` marker; **requeue** = abandoned without
  prejudice (pause/stop), leaving the item `pending` and no marker.
- A lost completion report is safe by design: the lease expires and the item
  returns to the pool, costing a re-run at worst.
- **`jobs.agent_allowlist`** (JSON array of agent IDs, `[]` = unrestricted)
  gates both `db.claim_next()` and the scan handout in `agent_checkin`. Since
  SQLite's JSON1 extension isn't guaranteed present, matching happens in
  Python (`db.agent_allowed`) rather than in the SQL `WHERE`. **Filter per job,
  never per item:** the allowlist is a job-level property, so `claim_next`
  walks running jobs in `priority, id` order, skips those this agent may not
  run, and takes the first pending item of the first eligible one. An earlier
  version filtered a `LIMIT 500` batch of *items* and could hand an agent
  nothing at all when enough work routed elsewhere sat ahead of its next
  eligible item. Jobs number in the tens, so this needs no limit — which is
  what makes starvation structurally impossible rather than merely unlikely.
  All of it stays inside the same `BEGIN IMMEDIATE` transaction as the CAS
  update, so the primary claim guard (layer 1 above) is unchanged.
- **`agents.abort_requested`** is the immediate per-agent Stop (distinct from
  per-agent Pause, which only blocks future claims and lets the current file
  finish). `POST /api/agents/{id}/stop` sets both `paused` and
  `abort_requested`; `agent_heartbeat` consumes the flag **one-shot** — it
  clears it the instant any heartbeat from that agent arrives, so it can only
  ever abort the file in flight *right now*, never a later one. The
  `paused = 0, abort_requested = 0` resume path in `pause_agent` is a second
  safety net for the case where Stop was clicked on an idle agent: with no
  in-flight heartbeat to consume it there, a stale flag would otherwise
  detonate against whatever that agent claims next.
- **`jobs.webhook_*`** is a generic, opaque completion notifier for external
  integrations (e.g. a Stash plugin that submitted the job wanting to know
  when to rescan). `agent_complete` fires `webhook_url` with exactly
  `webhook_headers`/`webhook_body` as given at job creation — the hub never
  interprets them, which is what lets one mechanism serve any caller without
  the hub knowing anything about what's on the other end. `webhook_rel_path`
  optionally scopes it to one file in the job (blank = every item);
  `webhook_states` (default `["done", "skipped"]`) picks which outcomes
  trigger it. Fired from a detached thread in `_fire_webhook` — the request
  and all its failure modes are swallowed, so a slow or dead target can never
  delay or fail the agent's `/complete` call.

### Constraints to preserve

- **The agent must stay standard-library only** — workers install nothing beyond
  WhisperJAV. Do not add `requests` or similar to `agent/`.
- **Cross-platform.** A worker may be Windows: `runner` branches on `os.name` for
  process-group kill (`taskkill /T`) and drive-letter roots in `list_directory`.
- **NFO selection deliberately differs from the batch script.** `runner.find_nfo`
  prefers an exact basename match, falls back to the original "only .nfo in the
  folder" behaviour, and returns `None` rather than guessing when several exist
  and none matches — the original's `nfo_files[0]` silently fed the wrong
  title/actress/plot to the translator in multi-video folders.
- WhisperJAV has **no native named-profile store** (`--config` is a JSON file of
  internal tuning blocks, not a saved invocation), which is why profiles live
  here.

### Dashboard

`hub/static/index.html` is a single dependency-free page: polls `/api/state`
every 2.5s and re-renders. All UI state lives in that one payload; adding a panel
usually means extending `get_state` rather than adding an endpoint.
