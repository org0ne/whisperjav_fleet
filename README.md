# WhisperJAV Fleet

Turns `claude_whisperjav_batch.py` into a two-machine (or n-machine) fleet
driven from one always-on web dashboard.

The transcription behaviour is unchanged from the batch script — same `.nfo`
parsing, same argument templating, same output naming, same audit logs. What
is new is the coordination: a hub owns the queue, agents pull one file at a
time, and **two machines pointed at the same folder can never process the same
video**.

```
   browser
      │
      ▼
 ┌──────────────┐        outbound HTTP only        ┌─────────────────┐
 │   Hub        │◀─────────────────────────────────│ Agent (debeast) │
 │  LXC / any   │   checkin · claim · heartbeat     └─────────────────┘
 │  small box   │◀─────────────────────────────────┌─────────────────┐
 │  no mount    │                                   │ Agent (box two) │
 └──────────────┘                                   └─────────────────┘
                                                            │
                                        both mount the same share, e.g.
                                             /storage/pool/...
```

**The hub never touches the media filesystem.** Scanning, browsing, reading
and writing are all done by agents and reported back. That is what lets the
hub live in a small LXC with no CIFS mount and no GPU.

---

## How double-processing is prevented

Four independent layers, so no single failure re-runs a file or corrupts an
output:

1. **Atomic claim (the primary guard).** The hub hands out work from SQLite
   inside a `BEGIN IMMEDIATE` transaction, with the mark-as-claimed written as
   a compare-and-swap (`UPDATE … WHERE state = 'pending'`). SQLite permits one
   writer at a time, so simultaneous pollers are serialised by the database:
   the loser matches zero rows and asks again.

2. **Lease + heartbeat.** A claim expires after 15 minutes without a
   heartbeat. If an agent is unplugged mid-file, the lease lapses and the file
   returns to the queue — no manual cleanup, and no file silently stranded.

3. **On-disk lock file.** Before touching a video, the agent creates
   `<basename>.whisper-lock` with `O_CREAT|O_EXCL` (atomic on local disks and
   on SMB/CIFS). This covers what the hub cannot see: someone running the old
   batch script by hand, a second hub, or a hub database that was reset while
   work was in flight. Locks older than 6 hours are treated as abandoned.

4. **Last-moment output re-check.** A scan may be minutes old, so immediately
   before running — and again after taking the lock — the agent re-checks for
   `<basename>.<lang>.srt` and skips if the other machine got there first.

On top of that, **the output SRT is published atomically**: the audit comment
block is prepended in memory, written to a temp file in the same directory,
and moved into place with `os.replace`. A reader — or the other machine's skip
check — sees either the complete file or no file, never a half-written one.

---

## Setup

### Hub — one machine (the Proxmox LXC)

Needs Python 3.10+ and nothing else. No media mount, no GPU.

The repo is private, so cloning it on a new machine needs GitHub credentials
(`gh auth login`, or an SSH deploy key) — same requirement on every agent
machine below.

```bash
git clone https://github.com/org0ne/whisperjav_fleet.git /opt/whisperjav-fleet
cd /opt/whisperjav-fleet/hub
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

export HUB_USER=someuser        # strongly recommended if reachable off-LAN
export HUB_PASS=somepassword
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8787
```

Open `http://<hub>:8787`. It will be empty until an agent checks in.

To run it as a service, edit `User=`, `WorkingDirectory=` and the credentials
in `hub/whisperjav-hub.service`, then:

```bash
sudo cp hub/whisperjav-hub.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now whisperjav-hub
```

### Agent — every worker machine

**No pip install required** — the agent uses only the standard library. It
needs WhisperJAV on the machine and access to the media share.

```bash
git clone https://github.com/org0ne/whisperjav_fleet.git /opt/whisperjav-fleet
cd /opt/whisperjav-fleet

sudo mkdir -p /etc/whisperjav-agent
sudo cp agent/config.example.json /etc/whisperjav-agent/config.json
sudo nano /etc/whisperjav-agent/config.json     # set hub_url

python3 agent/agent.py --config /etc/whisperjav-agent/config.json
```

The machine appears on the dashboard within a few seconds. Then:

```bash
sudo cp agent/whisperjav-agent.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now whisperjav-agent
```

### Agent config

| Field | Meaning |
|---|---|
| `agent_id` | Name on the dashboard. `null` → hostname. |
| `hub_url` | e.g. `http://192.168.11.50:8787` |
| `hub_user` / `hub_pass` | Only if the hub sets `HUB_USER`/`HUB_PASS`. |
| `whisperjav_bin` | Blank → auto-detect (`$WHISPERJAV_BIN`, then the known venv paths, then `PATH`). |
| `heartbeat_seconds` | Lease refresh interval, and how fast a Pause reaches an in-flight file. |
| `allow_browse` | Whether this agent answers the dashboard's folder picker. |
| `path_map` | Prefix rewrites for a machine that mounts the share elsewhere. Empty when every machine uses the same path. |
| `log_output_lines` | Echo WhisperJAV's output into the agent log. |
| `ollama_unload_on_abort` | On any abort, ask Ollama to drop the job's translate model from VRAM (`false` by default). See "Notes and limits" below. |
| `ollama_url` | Where that request goes. Default `http://127.0.0.1:11434`. |
| `storage_retry_seconds` | When the media share itself becomes unreachable (host down, stale mount), the item in hand is requeued rather than failed and the agent stops claiming for this long before trying again. Default `60`. |

Every field can also come from the environment: `FLEET_HUB_URL`,
`FLEET_AGENT_ID`, `FLEET_HUB_USER`, `FLEET_HUB_PASS`, `WHISPERJAV_BIN`.

---

## Using it

**Jobs.** A job is a target folder plus the exact WhisperJAV arguments to run
against it. Create one, and the first idle agent scans the folder; every agent
then pulls files from the shared queue until it is empty. Both machines work
the same job concurrently, each on a different file.

**Invocation criteria.** Each job carries its own free-form argument box —
the same template format as `WHISPERJAV_ARGS` in the batch script, including
`{INPUT_FILE}`, `{TRANSLATE_TITLE}`, `{TRANSLATE_ACTRESS}` and
`{TRANSLATE_PLOT}`. **Profiles** are named starting points for that box; the
batch script's current recipe is seeded as `qwen-galgame-translate`. The
arguments are *snapshotted onto the job at creation*, so editing a profile
later never changes a job already queued.

**Pause / Stop (job).** WhisperJAV cannot resume a partial transcription, so
Pause kills the in-flight process — along with the helpers it spawned, via its
process group, so nothing is left holding GPU memory — and returns the file to
the queue. Nothing partial is written, so restarting is always safe. Stop does
the same and requeues everything unfinished.

**Pause / Stop now (machine).** Each machine on the **Machines** panel has its
own, separate controls. **Pause** just stops that agent from picking up new
files — whatever it's already transcribing keeps running to completion.
**Stop now** is immediate: it aborts the current file right away (same clean
requeue as above, nothing partial written) and pauses the agent, freeing its
GPU without waiting for that file to finish — handy when you need the compute
for something else. Click **Resume** when you want it back in the pool.

**Re-running is free.** A rescan skips any video that already has its
`.<lang>.srt`, exactly like the batch script. Point a job at a large tree
repeatedly and it only picks up what is new.

**Folder picker.** "Browse…" lists directories *as a chosen worker sees them*,
since the hub has no mount of its own.

**Agent routing.** By default any online, unpaused agent may claim any file
from any running job. To dedicate a job to specific machines, set "Restrict
to agents" when creating it (or click **Agents…** on an existing job) to a
comma-separated list of agent IDs — every other agent then skips that job's
queue entirely. Leave it blank for the default shared-queue behaviour.

**Completion webhooks (API only, no dashboard UI).** `POST /api/jobs` accepts
`webhook_url`, `webhook_headers`, and `webhook_body` — when a work item
finishes, the hub fires exactly that request, as-is, off-thread and
best-effort. It never inspects or interprets the payload, so it can notify
anything: for example, a Stash plugin that submitted the job can set these to
Stash's own GraphQL endpoint and a `metadataScan` mutation, so a completed
subtitle gets picked up automatically. Two optional filters narrow when it
fires: `webhook_rel_path` (a specific file in the job; blank fires for every
item) and `webhook_states` (which outcomes trigger it; default
`["done", "skipped"]`). A slow or unreachable target never delays or fails
the agent's completion report.

`auto_delete_on_finish` (also API-only) deletes a job once it finishes with
zero failed items, instead of leaving it in the list — for callers that
submit many small one-off jobs (again, a Stash plugin submitting one job per
scene is the motivating case) and have no use for a persistent history. Any
failure keeps the job around regardless, so nothing about the failure is
ever silently lost.

---

## Output, per video

Unchanged from the batch script, plus two provenance lines:

```
<basename>.<lang>.srt              final subtitle, with the audit block prepended as {/ … /}
<basename>-whisper-command.txt     audit log — now also records Processed By and Fleet Job
<basename>.whisper-failed          written only on failure
<basename>.whisper-lock            present only while a machine is working on it
```

---

## Notes and limits

- **`.nfo` selection differs slightly from the batch script.** The original
  took the first `.nfo` in the directory, which silently attaches the wrong
  title/actress/plot when a folder holds several videos — and that metadata is
  fed straight to the translator. The agent prefers an exact basename match,
  falls back to the old behaviour when a folder holds exactly one `.nfo`, and
  supplies no metadata rather than guessing when there are several and none
  matches. For a one-video-per-folder library the behaviour is identical.

- **WhisperJAV has no native profile store.** `--config` takes a JSON file of
  internal tuning blocks (VAD thresholds, model-load params), and flags like
  `--pass1-sensitivity` / `--qwen-postprocess-preset` select named blocks
  *within* it. There is no way to save and recall a whole invocation by name —
  that is what this project's profiles provide.

- **A detached helper can outlive a Pause.** Killing the process group reaches
  everything WhisperJAV spawns normally. A process that deliberately escapes
  into its own session (or a pre-existing daemon such as an `ollama serve` you
  started yourself) is by definition out of reach — an aborted run frees
  whisperjav's own GPU memory immediately, but not a model Ollama is still
  holding. Set `ollama_unload_on_abort: true` in that agent's config and it
  will ask Ollama to drop the job's `--translate-model` (`keep_alive: 0`) the
  moment an abort happens, best-effort. `ollama_url` defaults to
  `http://127.0.0.1:11434`.

- **Command delivery is at-most-once.** A command issued exactly as the hub or
  agent restarts can be lost — click the button again. Fine for a manual
  control panel.

- **Auth is HTTP Basic and optional.** Put the hub behind HTTPS
  (Caddy/nginx/Tailscale) before exposing it beyond a trusted LAN. Note that
  `allow_browse` lets anyone with dashboard access list directories on the
  worker machines.
