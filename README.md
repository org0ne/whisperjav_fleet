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
| `hub_url` | e.g. `http://hub.lan:8787` |
| `hub_user` / `hub_pass` | Only if the hub sets `HUB_USER`/`HUB_PASS`. |
| `whisperjav_bin` | Blank → auto-detect (`$WHISPERJAV_BIN`, then the known venv paths, then `PATH`). |
| `heartbeat_seconds` | Lease refresh interval, and how fast a Pause reaches an in-flight file. |
| `allow_browse` | Whether this agent answers the dashboard's folder picker. |
| `path_map` | Prefix rewrites for a machine that mounts the share elsewhere. Empty when every machine uses the same path. |
| `log_output_lines` | Echo WhisperJAV's output into the agent log. |
| `ollama_unload_on_abort` | On any abort, ask Ollama to drop the job's translate model from VRAM (`false` by default). See "Notes and limits" below. |
| `ollama_url` | Where that request goes. Default `http://127.0.0.1:11434`. |
| `storage_retry_seconds` | When the media share itself becomes unreachable (host down, stale mount), the item in hand is requeued rather than failed and the agent stops claiming for this long before trying again. Default `60`. |
| `env_retry_seconds` | When whisperjav fails for a reason that is this worker's rather than the file's (expired Hugging Face token, GPU driver gone after a reboot, broken venv, Ollama down -- see "A broken worker is not a failed file" below), the item is requeued rather than failed and the agent stops claiming for this long. Default `300`. |
| `gpu_hold_url` | Blank = off. The stash-jasna-bridge health URL, e.g. `http://127.0.0.1:8770/jasna/health`. While Jasna needs the GPU -- a viewer session, a batch restoration, or the managed process still resident -- the agent stays off it. See "Sharing the GPU with Jasna". |
| `gpu_hold_process` | Blank = off. A process name, e.g. `jasna`: hold while `nvidia-smi` lists a GPU compute process of that name using at least `gpu_hold_process_min_mb` (default `1024`). The way to see a Jasna *desktop app* started by hand on a machine with no bridge. Combine with `gpu_hold_url` or use alone. |
| `gpu_hold_mode` | `yield` (default): abort and requeue the file in hand the moment Jasna starts. `finish`: let the current file complete, only stop claiming new ones. |
| `gpu_hold_while_resident` | Also hold while the Jasna *process* is resident (it keeps its models in VRAM until the bridge's idle stop), not only while someone is watching. Default `true`. |
| `gpu_hold_poll_seconds` / `gpu_hold_release_seconds` | How often to re-check (`3`), and how long the hold must stay clear before claiming resumes (`30`), so a quick OFF/ON does not restart a long file just to yield it again. |

Every field can also come from the environment: `FLEET_HUB_URL`,
`FLEET_AGENT_ID`, `FLEET_HUB_USER`, `FLEET_HUB_PASS`, `WHISPERJAV_BIN`,
`FLEET_GPU_HOLD_URL`, `FLEET_GPU_HOLD_PROCESS`.

### Sharing the GPU with Jasna

On a machine that also runs [Jasna](https://github.com/Kruk2/jasna) through
[stash-jasna-bridge](https://github.com/org0ne/stash-jasna-bridge), the two do
not fit on one card together: Jasna's hq preset is ~14 GB and the Ollama
translate model ~9 GB more, on a 24 GB GPU; on a 12 GB card nothing fits
beside Jasna at all. So Jasna wins. Set `gpu_hold_url` to the bridge's health
endpoint (no auth needed) and the agent:

- **does not claim** a file while a viewer session is active or starting,
  while the bridge's batch queue is restoring a file, or while the Jasna
  process is still resident holding VRAM;
- in `yield` mode, **aborts the file in hand** within a few seconds of Jasna
  starting, kills whisperjav's process group, asks Ollama to unload the
  translate model (`ollama_unload_on_abort: true` recommended), and requeues
  the file as "yielded to Jasna"; the dashboard shows the agent as *holding*;
- **resumes on its own** once the bridge reports Jasna gone for
  `gpu_hold_release_seconds`.

It is a race, not a lock: the bridge flags `preparing` before Jasna loads its
models, and the agent polls every `gpu_hold_poll_seconds`, so VRAM is usually
free in time. If it is not, the toggle in Stash errors once and works on retry.
Two knobs outside this repo tighten it: a shorter `process_idle_minutes` in the
bridge config so a lingering Jasna frees the card sooner, and a short
`OLLAMA_KEEP_ALIVE` on the Ollama service. An unreachable bridge is treated as
"no Jasna", so the agent keeps working if the bridge is down.

**A Jasna nothing manages.** A worker where Jasna is the desktop app, started
by hand with no bridge in front of it, has no health URL to poll. Set
`gpu_hold_process: jasna` instead: the agent asks `nvidia-smi` for the GPU's
compute processes on the same cadence and holds while one whose name matches
uses at least `gpu_hold_process_min_mb` of VRAM. The idle app sits on the
card at a few hundred MB, under the default 1024 MB floor, so merely having
it open does not stop the fleet; a restoration pushes it into the gigabytes
within seconds and the agent yields. This is a later signal than the bridge's
`preparing` flag -- the process has already started allocating -- so a
restoration may fail once on a tight card before the agent is out of the
way; Jasna retries cleanly. Both signals can be set together on a bridge
machine; the bridge is asked first. A missing or failing `nvidia-smi` is
fail-open, the same as a bridge that is down.

---

### Holding the fleet through SnapRAID maintenance

If the media share sits on a SnapRAID array managed by
[snapraid-daemon](https://github.com/amadvance/snapraid-daemon), two
mechanisms together keep the fleet off the array for the *whole* nightly
sequence — spin-up, sync, scrub, probe, spin-down, report — not just the
sync:

1. **A daily maintenance window on the hub** (dashboard: **Maintenance
   window…** on the Machines panel, or `POST /api/hold/window`). Hub local
   time, e.g. `02:55` for `65` minutes around a 03:00 `maintenance_schedule`.
   This is the floor: it covers the fixed slot whether or not anything
   fires, including the spin-up before sync and the spin-down after scrub
   that snapraidd never tells a hook about. By default it also **aborts
   files in progress** when it opens (they requeue; nothing partial is
   written) — the intent is that nothing is transcribing during
   maintenance, and a file claimed at 02:50 would otherwise run through the
   entire sync. Untick it to let in-flight files finish instead.
   **Release (skip today)** on the banner sits out the rest of one
   occurrence; the schedule fires again tomorrow.

2. **The hook script** [`contrib/snapraid-hook.sh`](contrib/snapraid-hook.sh)
   as snapraidd's `hook_script`, which makes the hold *grow* with the array.
   The daemon runs it around sync/scrub (and diff/check/fix); the script
   only ever sets or extends the hold, never releases it: a task's begin
   arms `HOLD_TTL_SECONDS` (2 h), and its end extends the hold
   `HOLD_TAIL_SECONDS` (15 min) past the last hooked task to cover probe,
   spin-down and report. Every call is extend-only, so it can never cut the
   window or a manual hold short. When a sync one day outgrows the window,
   the hook is what keeps the fleet held until it is done.

On the SnapRAID host:

```bash
sudo install -o root -g root -m 0755 contrib/snapraid-hook.sh /usr/local/bin/snapraid-hook.sh
sudo nano /usr/local/bin/snapraid-hook.sh   # set HUB_URL (and HUB_USER/HUB_PASS if the hub has auth)
```

Then in `snapraidd.conf`:

```
hook_script = /usr/local/bin/snapraid-hook.sh
# optional: arm the hold before the disks even spin up, not once sync begins
notify_start = /usr/local/bin/snapraid-hook.sh maintenance-start
```

and reload the daemon (SIGHUP). Points worth knowing:

- **The script always exits 0.** snapraid-daemon aborts a task whose
  `task-begin` hook fails, and a hub that is down must never cancel the sync.
  An unreachable hub is logged to syslog and the sync proceeds unheld.
- **Fail-open, both halves.** The window closes on the clock; the hook's
  hold carries a deadline. Neither needs a release call to arrive, so a
  host that reboots mid-sync or a hub that is unreachable at the end cannot
  park the fleet for longer than the later deadline.
- **`HOLD_COMMANDS`** defaults to `sync scrub`. `diff`, `check` and `fix`
  are also hooked; add them if you run those on a schedule.
- **Daemon version.** The script needs snapraid-daemon built from master
  on or after 2026-06-17 (the 2.0 line): that is when the hook gained
  `SNAPRAID_COMMAND` / `SNAPRAID_HIGH_COMMAND` and the daemon gained
  `notify_start`. No tagged release up to v1.14 has either. An older daemon
  passes only the event name; the script then logs
  `snapraidd passed no SNAPRAID_COMMAND ... upgrade it` to syslog and holds
  nothing (the hub's daily window still covers the fixed slot). To upgrade
  a daemon you built from git, in its checkout:

  ```bash
  sudo apt-get install -y autoconf automake pkg-config   # autogen.sh needs autoreconf
  git pull
  ./autogen.sh && ./configure --prefix=/usr/local   # the prefix you used before; /usr/local/etc/snapraidd.conf means /usr/local
  make && sudo make install                          # keeps an existing snapraidd.conf
  sudo systemctl restart snapraidd && snapraidd --version
  ```

  To prove the wiring without waiting for the night, schedule a `diff`
  through the daemon's REST API (`POST /snapraid/v1/schedule` with
  `{"tasks":[{"command":"diff"}]}`) after temporarily adding `diff` to
  `HOLD_COMMANDS` and setting `HOLD_ABORT_IN_FLIGHT=0` in
  `/usr/local/etc/snapraid-hook.conf`; the hub's Activity panel should show
  "Fleet hold set: snapraid diff". Release the hold and delete the conf file
  afterwards.

  The daemon's own notes say its 1.11+ heartbeat handling needs the
  SnapRAID engine at 14.6 or later, and it currently ships against 14.9;
  an engine at 14.4 is worth bringing up at the same time.
- **Daemon rules.** The script must be root- or daemon-owned, not group- or
  world-writable, executable, without setuid, and a single hard link; the
  `install` line above satisfies them. The daemon scrubs the environment
  before running it, so settings live in the script or in
  `/usr/local/etc/snapraid-hook.conf`, which it sources if present.
- **Time zones.** The window runs on the hub's clock; the dialog shows the
  hub's zone and current time so a UTC container is obvious. Set `TZ` in
  the hub's service environment if it should follow the NAS.

The hold is generic: anything that can POST JSON can use it (see below).

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

**Hold the fleet (SnapRAID maintenance, disk swaps, …).** **Hold fleet…**
on the Machines panel and `POST /api/hold` stop *every* agent from claiming
new files at once, without touching the per-machine Pause state: releasing
the hold never resumes a machine you paused by hand. By default files
already transcribing run to completion; with `abort_in_flight` they are
aborted and requeued instead (nothing partial is ever written, so this is
always safe — it costs the GPU time spent so far). Every hold carries a
deadline (default 6 h) and lifts by itself when it passes, so a caller that
never gets to release cannot park the fleet indefinitely. Scans and folder
browses still run during a hold; they only read the share, and a job created
while held is queued and ready the moment it lifts. A daily scheduled window
(`/api/hold/window`) is the second source; the two merge — held while either
is active, until the later deadline, aborting if either says so. See
"Holding the fleet through SnapRAID maintenance" above.

```bash
curl -X POST http://hub:8787/api/hold -H 'Content-Type: application/json' \
     -d '{"held": true, "reason": "disk swap", "ttl_seconds": 7200, "abort_in_flight": false, "set_by": "scott"}'
curl -X POST http://hub:8787/api/hold -H 'Content-Type: application/json' -d '{"held": false}'
curl http://hub:8787/api/hold          # {"hold": {...}} or {"hold": null}
curl -X POST http://hub:8787/api/hold/window -H 'Content-Type: application/json' \
     -d '{"enabled": true, "start": "02:55", "minutes": 65, "reason": "snapraid maintenance", "abort_in_flight": true}'
```

Setting an active hold again moves its deadline and updates the reason;
`"extend_only": true` lets a caller only ever move the deadline *later*, which
is how the hook script layers on top of the window without shortening it.
`{"held": false}` releases whatever is holding right now — the ad-hoc hold,
and, if the window is open, the rest of today's occurrence.

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
the agent's completion report, but every attempt is recorded on the job's
events — `webhook delivered: <url> -> HTTP 200`, or an error-level
`webhook FAILED: <url> -> <reason>` — so a callback aimed at the wrong
address (a hub on another machine calling `localhost`, say) shows up
instead of vanishing. Remember the URL is used *from the hub*: it has to be
an address the hub can reach, not one that only works on the caller.

`auto_delete_on_finish` (also API-only) deletes a job once it finishes with
zero failed items, instead of leaving it in the list — for callers that
submit many small one-off jobs (again, a Stash plugin submitting one job per
scene is the motivating case) and have no use for a persistent history. Any
failure keeps the job around regardless, so nothing about the failure is
ever silently lost. A job with a webhook is likewise kept until that webhook
has actually been delivered; if the delivery fails the job stays, with the
error on its events, rather than taking the only record of itself away.

---

## Output, per video

Unchanged from the batch script, plus two provenance lines:

```
<basename>.<lang>.srt              final subtitle, with the audit block prepended as {/ … /}
<basename>-whisper-command.txt     audit log — now also records Processed By and Fleet Job
<basename>.whisper-failed          written only on failure
<basename>.whisper-lock            present only while a machine is working on it
```

When the agent recognises the failure, `.whisper-failed` carries a
`--- Diagnosis ---` section ahead of the raw output that names the cause in
plain words and what to check (the batch script has no equivalent). The same
one-line cause is the item's error on the dashboard and in the agent's log.

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

- **A broken worker is not a failed file.** `runner.diagnose_failure` matches
  whisperjav's output against the failures met so far. Those that belong to
  the machine rather than the video -- Hugging Face refusing to resolve a
  cached model because the token in `~/.cache/huggingface/token` expired
  (the OAuth tokens `hf auth login` issues last 30 days), a GPU driver that
  did not survive a kernel update, a venv missing a package, Ollama not
  running or missing the translate model, ffmpeg gone, disk full -- do not
  fail the item. It goes back to the queue with the cause as its error,
  another worker may take it, and this agent shows *degraded* on its card
  with the cause, stops claiming for `env_retry_seconds`, and clears itself
  once a run succeeds. Without this, one expired token failed every item in
  the queue at one per two minutes and planted a `.whisper-failed` beside
  each. Causes that can be the file's (GPU out of memory) still fail the
  item, with the diagnosis in the marker. To take Hugging Face out of the
  picture entirely on a worker whose models are all cached, uncomment
  `HF_HUB_OFFLINE=1` in `agent/whisperjav-agent.service`.

- **Command delivery is at-most-once.** A command issued exactly as the hub or
  agent restarts can be lost — click the button again. Fine for a manual
  control panel.

- **Auth is HTTP Basic and optional.** Put the hub behind HTTPS
  (Caddy/nginx/Tailscale) before exposing it beyond a trusted LAN. Note that
  `allow_browse` lets anyone with dashboard access list directories on the
  worker machines.
