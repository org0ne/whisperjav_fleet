#!/usr/bin/env python3
"""
agent.py -- WhisperJAV fleet agent.

Runs on every machine that has a WhisperJAV install and a GPU.  It polls the
hub over plain HTTP, claims one video at a time, transcribes it, and reports
the result.  Because it only makes outbound requests, a worker needs no open
ports and can sit behind NAT or a VPN.

Deliberately depends on nothing outside the Python standard library, so a
worker machine needs no pip install beyond WhisperJAV itself.

    python3 agent.py --config /etc/whisperjav-agent/config.json
    python3 agent.py --hub http://hub.lan:8787 --once      # smoke test
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner
from runner import LockBusy, VideoLock

DEFAULT_CONFIG = {
    "agent_id": None,                 # None -> hostname
    "hub_url": "http://localhost:8787",
    "hub_user": "",
    "hub_pass": "",
    "poll_interval_seconds": 3,
    "idle_poll_interval_seconds": 3,
    "heartbeat_seconds": 20,
    "whisperjav_bin": "",             # blank -> auto-detect
    "allow_browse": True,             # answer dashboard folder-picker requests
    "path_map": [],                   # [{"from": "/storage/pool", "to": "P:\\"}]
    "log_output_lines": False,        # echo whisperjav output to the agent log
    "ollama_unload_on_abort": False,  # best-effort: drop the translate model from VRAM on abort
    "ollama_url": "http://127.0.0.1:11434",
}


def _flag_value(args: list[str], flag: str) -> Optional[str]:
    """The token immediately following `flag` in an argv-style list, if any."""
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def log(message: str, level: str = "INFO") -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"{stamp} {level:<5} {message}", flush=True)


# ---------------------------------------------------------------------------
# Hub client
# ---------------------------------------------------------------------------

class HubClient:
    def __init__(self, base_url: str, agent_id: str, user: str = "", password: str = ""):
        self.base = base_url.rstrip("/")
        self.agent_id = agent_id
        self._auth = ""
        if user:
            raw = f"{user}:{password}".encode()
            self._auth = "Basic " + base64.b64encode(raw).decode()

    def _request(self, method: str, path: str, payload: Any = None, timeout: float = 30.0) -> Any:
        url = f"{self.base}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self._auth:
            req.add_header("Authorization", self._auth)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}

    def post(self, path: str, payload: Any = None, timeout: float = 30.0) -> Any:
        return self._request("POST", path, payload, timeout)

    # -- typed helpers ------------------------------------------------------

    def checkin(self, status: dict) -> dict:
        return self.post(f"/api/agent/{self.agent_id}/checkin", status)

    def claim(self) -> Optional[dict]:
        return self.post(f"/api/agent/{self.agent_id}/claim", {}).get("item")

    def heartbeat(self, item_id: int, progress: str = "") -> dict:
        return self.post(f"/api/agent/{self.agent_id}/heartbeat",
                         {"item_id": item_id, "progress": progress[:400]})

    def complete(self, item_id: int, state: str, message: str = "", duration: float = 0.0) -> dict:
        return self.post(f"/api/agent/{self.agent_id}/complete",
                         {"item_id": item_id, "state": state,
                          "message": message[:4000], "duration": duration})

    def requeue(self, item_id: int, reason: str = "") -> dict:
        return self.post(f"/api/agent/{self.agent_id}/requeue",
                         {"item_id": item_id, "reason": reason[:400]})

    def scan_result(self, job_id: int, files: list[str], error: str = "") -> dict:
        return self.post(f"/api/agent/{self.agent_id}/scan_result",
                         {"job_id": job_id, "files": files, "error": error},
                         timeout=120.0)

    def browse_result(self, request_id: int, entries: list[dict], error: str = "") -> dict:
        return self.post(f"/api/agent/{self.agent_id}/browse_result",
                         {"request_id": request_id, "entries": entries, "error": error})


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, config: dict):
        self.cfg = config
        self.agent_id = config["agent_id"] or socket.gethostname()
        self.hub = HubClient(config["hub_url"], self.agent_id,
                             config.get("hub_user", ""), config.get("hub_pass", ""))
        self.binary = runner.find_whisperjav_bin(config.get("whisperjav_bin", ""))
        self.version = runner.get_whisperjav_version(self.binary)

        self.started_at = time.time()
        self.shutdown = threading.Event()
        self.current: dict = {}          # description of the in-flight item
        self.counts = {"done": 0, "failed": 0, "skipped": 0}
        self._abort = threading.Event()
        self._progress = ""

        # Set by the presence thread, read by the work loop.
        self._paused = False
        self._pending_scan: Optional[dict] = None
        self._connected = threading.Event()

    # -- status -------------------------------------------------------------

    def status(self) -> dict:
        return {
            "host":        socket.gethostname(),
            "state":       "working" if self.current else "idle",
            "current":     self.current,
            "progress":    self._progress[:300],
            "counts":      dict(self.counts),
            "whisperjav":  self.version,
            "binary":      self.binary,
            "platform":    sys.platform,
            "started_at":  self.started_at,
        }

    # -- path mapping -------------------------------------------------------

    def local_path(self, canonical: str) -> Path:
        """
        Translate the hub's canonical target directory into this machine's
        local path.  With identical mounts everywhere `path_map` is empty and
        this is a no-op; it exists so one odd machine cannot force every path
        in the system to be rewritten.
        """
        result = canonical
        for rule in self.cfg.get("path_map") or []:
            src, dst = rule.get("from", ""), rule.get("to", "")
            if src and result.startswith(src):
                result = dst + result[len(src):]
                break
        return Path(result)

    # -- Ollama -------------------------------------------------------------

    def _maybe_unload_ollama(self, wj_args: list[str]) -> None:
        """
        Killing whisperjav's process group frees everything *it* spawned, but
        `--translate-provider ollama` talks to a pre-existing `ollama serve`
        daemon over HTTP -- that process is never a child of whisperjav's, so
        no signal sent to its process group can reach it, and it holds the
        translate model in VRAM regardless of what happens to whisperjav.
        This is opt-in and best-effort: any failure here must not affect the
        outcome of the (already aborted) transcription.
        """
        if not self.cfg.get("ollama_unload_on_abort"):
            return
        if _flag_value(wj_args, "--translate-provider") != "ollama":
            return
        model = _flag_value(wj_args, "--translate-model")
        if not model:
            return

        url = self.cfg.get("ollama_url", "http://127.0.0.1:11434").rstrip("/") + "/api/generate"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
            log(f"asked Ollama to unload {model}")
        except Exception as exc:
            log(f"could not unload Ollama model {model}: {exc}", "WARN")

    # -- main loop ----------------------------------------------------------

    def _presence_loop(self) -> None:
        """
        Keep the hub informed that we are alive, whatever else we are doing.

        This runs in its own thread precisely because a single video can take
        far longer than the hub's offline threshold: if check-ins only happened
        between files, a busy machine would show up on the dashboard as idle
        and then vanish as offline halfway through a transcription.
        """
        interval = float(self.cfg.get("poll_interval_seconds", 3))
        backoff = 1.0

        while not self.shutdown.is_set():
            try:
                directives = self.hub.checkin(self.status())
                backoff = 1.0
                if not self._connected.is_set():
                    log("connected to hub")
                self._connected.set()
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
                if self._connected.is_set():
                    log(f"hub unreachable ({exc}); retrying", "WARN")
                self._connected.clear()
                self.shutdown.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

            self._apply_directives(directives)
            self.shutdown.wait(interval)

    def _apply_directives(self, directives: dict) -> None:
        self._paused = bool(directives.get("paused"))

        if directives.get("browse") and self.cfg.get("allow_browse", True):
            self._handle_browse(directives["browse"])

        # Only take a scan while idle -- scanning a large tree from a
        # machine that is mid-transcription just delays both.
        if directives.get("scan") and not self.current and self._pending_scan is None:
            self._pending_scan = directives["scan"]

    def _check_in_now_idle(self) -> None:
        """
        Called from the main loop the instant an item finishes, while
        `self.current` is guaranteed already cleared. The presence thread
        only samples idleness once every `poll_interval_seconds`, which on a
        continuously busy agent (back-to-back items, near-zero gap between
        them) almost never lands during that brief idle window -- starving
        any waiting scan job indefinitely even though it outranks everything
        else in the queue. This closes that race with a synchronous check-in
        at the one moment idleness is certain.
        """
        if self._paused or not self._connected.is_set():
            return
        try:
            directives = self.hub.checkin(self.status())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            return
        self._apply_directives(directives)

    def run(self, once: bool = False) -> int:
        log(f"agent '{self.agent_id}' starting")
        log(f"hub       : {self.cfg['hub_url']}")
        log(f"whisperjav: {self.binary} ({self.version})")

        presence = threading.Thread(target=self._presence_loop, daemon=True)
        presence.start()

        # Give the first check-in a moment so --once has something to act on.
        self._connected.wait(timeout=15)

        idle_sleep = float(self.cfg.get("idle_poll_interval_seconds", 3))

        while not self.shutdown.is_set():
            did_work = False

            if self._pending_scan is not None:
                task, self._pending_scan = self._pending_scan, None
                self._handle_scan(task)
                did_work = True

            elif not self._paused and self._connected.is_set():
                try:
                    item = self.hub.claim()
                except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
                    log(f"claim failed: {exc}", "WARN")
                    item = None
                if item:
                    self._process(item)
                    did_work = True
                    self._check_in_now_idle()

            if once:
                break
            if not did_work:
                self.shutdown.wait(idle_sleep)

        log("agent stopped")
        return 0

    # -- directives ---------------------------------------------------------

    def _handle_browse(self, task: dict) -> None:
        try:
            entries = runner.list_directory(task.get("path", ""))
            self.hub.browse_result(task["request_id"], entries)
        except Exception as exc:
            try:
                self.hub.browse_result(task["request_id"], [], str(exc))
            except Exception:
                pass

    def _handle_scan(self, task: dict) -> None:
        job_id = task["job_id"]
        target = self.local_path(task["target_dir"])
        log(f"scanning job {job_id}: {target}")
        try:
            if not target.is_dir():
                raise NotADirectoryError(f"'{target}' is not a directory or is not reachable")
            files = runner.scan_target(
                target,
                task.get("video_extensions") or [],
                task.get("lang_code", "qu"),
                bool(task.get("force")),
            )
            log(f"scan found {len(files)} file(s) needing work")
            self.hub.scan_result(job_id, files)
        except Exception as exc:
            log(f"scan failed: {exc}", "ERROR")
            try:
                self.hub.scan_result(job_id, [], str(exc))
            except Exception:
                pass

    # -- the work -----------------------------------------------------------

    def _process(self, item: dict) -> None:
        job       = item["job"]
        item_id   = item["item_id"]
        target    = self.local_path(job["target_dir"])
        video     = target / item["rel_path"]
        lang_code = job["lang_code"]
        basename  = video.stem
        video_dir = video.parent

        srt_out    = video_dir / f"{basename}.{lang_code}.srt"
        audit_out  = video_dir / f"{basename}-whisper-command.txt"
        failed_out = video_dir / f"{basename}.whisper-failed"

        self.current = {"job_id": job["id"], "job_name": job["name"],
                        "item_id": item_id, "file": item["rel_path"]}
        self._progress = ""
        self._abort.clear()
        started = time.time()

        try:
            # -- Pre-flight ---------------------------------------------------
            if not video_dir.exists():
                self._finish(item_id, "skipped", f"directory no longer exists: {video_dir}")
                return
            if not video.is_file():
                self._finish(item_id, "skipped", "source file no longer exists")
                return

            # Re-check the output at the last moment.  The scan may be minutes
            # old and the other machine may have produced this SRT since.
            if srt_out.exists() and not job["force"]:
                self._finish(item_id, "skipped", f"'{srt_out.name}' already exists")
                return

            # -- Exclusive lock ----------------------------------------------
            try:
                lock = VideoLock(video, owner=self.agent_id,
                                 extra={"job": job["name"], "hub_item": item_id})
                lock.acquire()
            except LockBusy as exc:
                # Something outside this hub's queue is working on the file.
                log(f"skip (locked): {item['rel_path']} -- {exc}", "WARN")
                self._finish(item_id, "skipped", f"lock held: {exc}")
                return

            try:
                # Between the existence check and the lock, the other side may
                # have finished.  One last look, now that we hold the lock.
                if srt_out.exists() and not job["force"]:
                    self._finish(item_id, "skipped", f"'{srt_out.name}' appeared during locking")
                    return

                self._run_one(item, job, video, video_dir, basename,
                              srt_out, audit_out, failed_out, lock, started)
            finally:
                lock.release()

        except OSError as exc:
            log(f"filesystem error on {item['rel_path']}: {exc}", "ERROR")
            self._finish(item_id, "failed", f"filesystem error: {exc}",
                         duration=time.time() - started)
        except Exception as exc:
            log(f"unexpected error on {item['rel_path']}: {exc}", "ERROR")
            self._finish(item_id, "failed", f"agent error: {exc}",
                         duration=time.time() - started)
        finally:
            self.current = {}
            self._progress = ""

    def _run_one(self, item, job, video, video_dir, basename,
                 srt_out, audit_out, failed_out, lock, started) -> None:
        item_id   = item["item_id"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # -- NFO ------------------------------------------------------------
        meta = runner.parse_nfo(runner.find_nfo(video))

        # -- Command --------------------------------------------------------
        wj_args  = runner.build_args(job["args_template"], video, meta)
        cmd_line = runner.command_display("whisperjav", wj_args)
        log(f"[{self.agent_id}] {item['rel_path']}")

        if job.get("dry_run"):
            self._finish(item_id, "skipped", f"dry run: {cmd_line}",
                         duration=time.time() - started)
            return

        # -- Heartbeat / abort watchdog -------------------------------------
        # One background thread keeps the hub's claim lease alive, refreshes
        # the on-disk lock's mtime, and watches for a Pause/Stop instruction.
        stop_beat = threading.Event()

        def beat():
            interval = float(self.cfg.get("heartbeat_seconds", 20))
            while not stop_beat.wait(interval):
                lock.touch()
                try:
                    resp = self.hub.heartbeat(item_id, self._progress)
                except Exception:
                    continue                      # transient; keep working
                if resp.get("abort"):
                    log(f"abort requested: {resp.get('reason', '')}", "WARN")
                    self._abort.set()
                    return

        beat_thread = threading.Thread(target=beat, daemon=True)
        beat_thread.start()

        def on_line(line: str) -> None:
            self._progress = line.strip()
            if self.cfg.get("log_output_lines"):
                log(f"    {line}", "OUT")

        try:
            exit_code, output, aborted = runner.run_whisperjav(
                self.binary, wj_args,
                on_line=on_line,
                should_abort=lambda: self._abort.is_set() or self.shutdown.is_set(),
            )
        finally:
            stop_beat.set()
            beat_thread.join(timeout=5)

        elapsed = time.time() - started

        # -- Aborted --------------------------------------------------------
        if aborted:
            # No SRT was produced, so nothing is half-done on disk: put the
            # file back and let whoever picks it up next start it cleanly.
            log(f"aborted after {runner.format_duration(elapsed)}: {item['rel_path']}", "WARN")
            self._maybe_unload_ollama(wj_args)
            try:
                self.hub.requeue(item_id, "aborted by pause/stop")
            except Exception:
                pass
            return

        # -- Failure --------------------------------------------------------
        if exit_code != 0:
            err = f"whisperjav exited with code {exit_code}.\n{output[-4000:]}"
            fail_text = runner.build_audit_text(
                self.version, timestamp, video, cmd_line, elapsed, err,
                agent_id=self.agent_id, job_name=job["name"],
            )
            runner.safe_write(failed_out, fail_text)
            self._finish(item_id, "failed", f"exit code {exit_code}", duration=elapsed)
            return

        # -- Locate output ---------------------------------------------------
        found_srt = runner.find_output_srt(video_dir, basename, srt_out)
        if not found_srt:
            err = "whisperjav exited 0 but no .srt output was found."
            fail_text = runner.build_audit_text(
                self.version, timestamp, video, cmd_line, elapsed, err,
                agent_id=self.agent_id, job_name=job["name"],
            )
            runner.safe_write(failed_out, fail_text)
            self._finish(item_id, "failed", err, duration=elapsed)
            return

        # -- Publish ----------------------------------------------------------
        audit_text = runner.build_audit_text(
            self.version, timestamp, video, cmd_line, elapsed,
            agent_id=self.agent_id, job_name=job["name"],
        )
        try:
            runner.publish_srt(found_srt, srt_out, audit_text)
        except OSError as exc:
            self._finish(item_id, "failed", f"could not write output SRT: {exc}", duration=elapsed)
            return

        runner.safe_write(audit_out, audit_text)
        try:
            failed_out.unlink()          # clear a stale failure marker
        except OSError:
            pass

        log(f"ok: {srt_out.name} in {runner.format_duration(elapsed)}")
        self._finish(item_id, "done", "", duration=elapsed)

    def _finish(self, item_id: int, state: str, message: str = "", duration: float = 0.0) -> None:
        if state in self.counts:
            self.counts[state] += 1
        if state == "skipped" and message:
            log(f"skip: {message}")
        try:
            self.hub.complete(item_id, state, message, duration)
        except Exception as exc:
            # The lease will expire and the item returns to the pool, so a lost
            # completion costs a re-run at worst -- never a lost file.
            log(f"could not report completion ({exc}); item will be re-queued", "WARN")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(path: Optional[str]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        cfg.update({k: v for k, v in raw.items() if v is not None or k == "agent_id"})
    # Environment overrides make container/systemd deployment easy.
    for key, env in (("hub_url", "FLEET_HUB_URL"), ("agent_id", "FLEET_AGENT_ID"),
                     ("hub_user", "FLEET_HUB_USER"), ("hub_pass", "FLEET_HUB_PASS"),
                     ("whisperjav_bin", "WHISPERJAV_BIN")):
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="WhisperJAV fleet agent.")
    parser.add_argument("--config", default=os.environ.get("FLEET_AGENT_CONFIG"),
                        help="Path to config.json.")
    parser.add_argument("--hub", default=None, help="Hub URL (overrides config).")
    parser.add_argument("--agent-id", default=None, help="Agent name (defaults to hostname).")
    parser.add_argument("--once", action="store_true",
                        help="Run a single poll cycle and exit (smoke test).")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not read config: {exc}", file=sys.stderr)
        return 1

    if args.hub:
        cfg["hub_url"] = args.hub
    if args.agent_id:
        cfg["agent_id"] = args.agent_id

    agent = Agent(cfg)

    def handle_signal(signum, _frame):
        log(f"signal {signum} received; finishing up", "WARN")
        agent.shutdown.set()
        agent._abort.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle_signal)
        except (ValueError, AttributeError):
            pass

    return agent.run(once=args.once)


if __name__ == "__main__":
    sys.exit(main())
