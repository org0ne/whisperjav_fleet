#!/usr/bin/env python3
"""
runner.py -- the per-file work of the original claude_whisperjav_batch.py,
refactored so the fleet agent can drive it one claimed file at a time.

The transcription behaviour is deliberately unchanged from the batch script:
same .nfo parsing, same {PLACEHOLDER} argument templating, same output SRT
selection, same audit-log format.  What is new is everything needed to make it
safe when two machines share one directory:

  * an exclusive on-disk lock file taken with O_CREAT|O_EXCL before any work
  * a re-check, at the last possible moment, that the output does not exist
  * atomic output publication via a temp file + os.replace

Nothing here imports the hub or knows about HTTP -- it is pure local work.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# A lock whose owning process died leaves a file behind.  After this long with
# no modification we assume the owner is gone and take it over.  Must exceed
# the longest plausible single-file transcription.
STALE_LOCK_SECONDS = 6 * 3600

LOCK_SUFFIX = ".whisper-lock"


# ---------------------------------------------------------------------------
# WhisperJAV binary resolution  (unchanged logic from the batch script)
# ---------------------------------------------------------------------------

def find_whisperjav_bin(configured: str = "") -> str:
    """
    Locate the WhisperJAV executable so the agent runs unmodified on any
    machine.  Resolution order:

        1. explicit config value      -- per-machine override from config.json
        2. $WHISPERJAV_BIN            -- per-machine env override
        3. known venv locations       -- the normal case
        4. 'whisperjav' on PATH       -- last resort

    PATH is deliberately last: on these machines a stale pyenv shim shadows
    the real binary and fails with "pyenv: whisperjav: command not found".
    """
    if configured:
        return os.path.expanduser(configured)

    override = os.environ.get("WHISPERJAV_BIN")
    if override:
        return os.path.expanduser(override)

    candidates = [
        "~/Applications/WhisperJav/.venv/bin/whisperjav",
        "~/Applications/WhsiperJav/.venv/bin/whisperjav",   # legacy misspelling
        "~/Applications/WhisperJav/.venv/Scripts/whisperjav.exe",
    ]
    for candidate in candidates:
        path = os.path.expanduser(candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    on_path = shutil.which("whisperjav")
    if on_path:
        return on_path

    return os.path.expanduser(candidates[0])


def get_whisperjav_version(binary: str) -> str:
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15)
        first_line = (result.stdout or result.stderr or "").splitlines()[0]
        return first_line.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# NFO parsing  (unchanged from the batch script)
# ---------------------------------------------------------------------------

def parse_nfo(nfo_path: Optional[Path]) -> dict:
    """
    Parse a Kodi/Jellyfin-style .nfo XML file and return:
        title     (str | None)  -- cleaned title
        actresses (list[str])   -- names from <actor><name> blocks
        plot      (str | None)  -- <plot> text
    """
    meta: dict = {"title": None, "actresses": [], "plot": None}

    if not nfo_path or not nfo_path.is_file():
        return meta

    try:
        root = ET.parse(nfo_path).getroot()

        title_el = root.find("title")
        if title_el is not None and title_el.text:
            raw = title_el.text.strip()
            # Strip everything up to and including the first " - "
            cleaned = re.sub(r"^[^-]*-\s*", "", raw)
            # Strip the LAST bracketed/parenthesised chunk, e.g. " [ABC-123]"
            cleaned = re.sub(r"\s*[\[\(][^\]\)]*[\]\)]\s*$", "", cleaned)
            meta["title"] = cleaned.strip() or None

        meta["actresses"] = [
            name_el.text.strip()
            for actor_el in root.findall(".//actor")
            for name_el in actor_el.findall("name")
            if name_el.text and name_el.text.strip()
        ]

        plot_el = root.find("plot")
        if plot_el is not None and plot_el.text and plot_el.text.strip():
            meta["plot"] = plot_el.text.strip()

    except Exception as exc:
        meta["_error"] = f"failed to parse {nfo_path.name}: {exc}"

    return meta


def find_nfo(video: Path) -> Optional[Path]:
    """
    Pick the .nfo describing this video.

    The original batch script took the first .nfo in the directory, which is
    correct for a one-video-per-folder library but silently attaches the wrong
    title/actress/plot when a folder holds several videos -- and that metadata
    is fed straight to the translator.  Prefer an exact basename match, then
    fall back to the old behaviour when the folder holds exactly one .nfo.
    """
    exact = video.with_suffix(".nfo")
    if exact.is_file():
        return exact

    try:
        candidates = sorted(video.parent.glob("*.nfo"))
    except OSError:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Several .nfo files and none matches this video by name: guessing would
    # mean mislabelling the translation, so supply no metadata at all.
    return None


# ---------------------------------------------------------------------------
# Argument templating  (unchanged from the batch script, template now a param)
# ---------------------------------------------------------------------------

def build_args(args_template: str, input_file: Path, meta: dict) -> list[str]:
    """
    Parse the args template, substitute runtime placeholders, and return a flat
    list of strings suitable for subprocess.  No shell involved -- each token
    becomes one list element regardless of spaces or special characters.

        {INPUT_FILE}        -> absolute path to the video file
        {TRANSLATE_TITLE}   -> cleaned title from .nfo  (flag dropped if absent)
        {TRANSLATE_ACTRESS} -> comma-separated actresses (flag dropped if absent)
        {TRANSLATE_PLOT}    -> plot text from .nfo       (flag dropped if absent)
    """
    flat_plot = re.sub(r"\r\n|\r|\n", " ", meta["plot"]) if meta.get("plot") else None
    substitutions = {
        "{INPUT_FILE}":        str(input_file),
        "{TRANSLATE_TITLE}":   meta.get("title"),
        "{TRANSLATE_ACTRESS}": ",".join(meta["actresses"]) if meta.get("actresses") else None,
        "{TRANSLATE_PLOT}":    flat_plot,
    }

    cleaned_lines = []
    for raw_line in args_template.splitlines():
        line = raw_line.strip().rstrip("\\").strip()
        if not line or line.startswith("#"):
            continue
        cleaned_lines.append(line)

    tokens = shlex.split(" ".join(cleaned_lines))

    result: list[str] = []
    for token in tokens:
        has_placeholder = any(p in token for p in substitutions)
        if has_placeholder:
            resolved = token
            drop = False
            for placeholder, value in substitutions.items():
                if placeholder in resolved:
                    if value is None:
                        drop = True
                        break
                    resolved = resolved.replace(placeholder, value)
            if drop:
                # Also drop the preceding flag (e.g. --translate-title)
                if result and result[-1].startswith("--"):
                    result.pop()
                continue
            result.append(resolved)
        else:
            result.append(token)

    return result


def command_display(binary_label: str, args: list[str]) -> str:
    """Human-readable command string for the audit log."""
    parts = [binary_label]
    for a in args:
        parts.append(f'"{a}"' if " " in a else a)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Audit log  (unchanged format from the batch script, plus the agent name)
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def build_audit_text(version: str, timestamp: str, source_file: Path, cmd_display: str,
                     duration_seconds: float = 0.0, extra_info: str = "",
                     agent_id: str = "", job_name: str = "") -> str:
    lines = [
        f"WhisperJAV Version : {version}",
        f"Timestamp          : {timestamp}",
        f"Duration           : {format_duration(duration_seconds)} ({duration_seconds:.1f}s)",
        f"Source File        : {source_file}",
    ]
    if agent_id:
        lines.append(f"Processed By       : {agent_id}")
    if job_name:
        lines.append(f"Fleet Job          : {job_name}")
    lines.append(f"Command            : {cmd_display}")
    if extra_info:
        lines += ["", "--- Output / Error ---", extra_info]
    return "\n".join(lines)


def safe_write(path: Path, text: str) -> bool:
    """Write text, tolerating a directory that vanished (NAS disconnect)."""
    try:
        path.write_text(text, encoding="utf-8")
        return True
    except OSError:
        return False


def _read_text_tolerant(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Exclusive per-video lock  --  defence in depth against double processing
# ---------------------------------------------------------------------------

class LockBusy(Exception):
    """Raised when another process holds a live lock on this video."""


class VideoLock:
    """
    An advisory lock file created next to the video with O_CREAT|O_EXCL, which
    is atomic on local filesystems and on SMB/CIFS shares.

    The hub's claim is already the authoritative guard -- this exists to cover
    the cases the hub cannot see: someone running the old batch script by hand,
    a second hub, or a hub database that was reset while work was in flight.
    """

    def __init__(self, video: Path, owner: str, extra: Optional[dict] = None):
        self.video = video
        self.path = video.with_name(video.stem + LOCK_SUFFIX)
        self.owner = owner
        self.extra = extra or {}
        self.acquired = False

    def _payload(self) -> str:
        return json.dumps({
            "owner":      self.owner,
            "host":       socket.gethostname(),
            "pid":        os.getpid(),
            "video":      self.video.name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            **self.extra,
        }, indent=2)

    def read_holder(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _is_stale(self) -> bool:
        try:
            return (time.time() - self.path.stat().st_mtime) > STALE_LOCK_SECONDS
        except OSError:
            return False

    def acquire(self) -> None:
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._is_stale():
                # Owner is long gone.  Remove and retry exactly once; if the
                # retry also collides, someone else won the race -- back off.
                try:
                    self.path.unlink()
                except OSError:
                    pass
                try:
                    fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except OSError as exc:
                    raise LockBusy(f"lock contended after stale cleanup: {exc}")
            else:
                holder = self.read_holder()
                who = holder.get("owner") or holder.get("host") or "another process"
                raise LockBusy(f"locked by {who}")
        except OSError as exc:
            raise LockBusy(f"could not create lock file: {exc}")

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(self._payload())
        except OSError:
            pass  # the lock exists, which is what matters
        self.acquired = True

    def touch(self) -> None:
        """Keep the mtime fresh so a long job is never mistaken for stale."""
        if not self.acquired:
            return
        try:
            os.utime(self.path, None)
        except OSError:
            pass

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self.acquired = False

    def __enter__(self) -> "VideoLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_target(target: Path, video_extensions: list[str], lang_code: str,
                force: bool) -> list[str]:
    """
    Recursively find candidate videos under `target`, returning POSIX-style
    paths relative to it.  Relative paths keep the queue portable: the hub
    stores one canonical target directory and each agent resolves it locally.

    Files that already have their output SRT are filtered out here, and checked
    again just before processing -- the other machine may have produced one in
    the meantime.
    """
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in video_extensions}
    found: list[str] = []

    for path in sorted(target.rglob("*")):
        try:
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            if not force:
                srt_out = path.with_name(f"{path.stem}.{lang_code}.srt")
                if srt_out.exists():
                    continue
            found.append(path.relative_to(target).as_posix())
        except OSError:
            continue

    return found


def list_directory(path_str: str) -> list[dict]:
    """
    Directory listing for the dashboard's folder picker.  An empty path lists
    plausible roots for whichever OS the agent runs on, so the picker works
    without the user knowing the machine's layout.
    """
    if not path_str:
        roots: list[dict] = []
        if os.name == "nt":
            import string
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    roots.append({"name": drive, "path": drive, "dir": True})
        else:
            for candidate in ("/", "/mnt", "/media", "/storage", "/srv",
                              os.path.expanduser("~")):
                if os.path.isdir(candidate):
                    roots.append({"name": candidate, "path": candidate, "dir": True})
        return roots

    base = Path(path_str)
    entries: list[dict] = []
    if base.parent != base:
        entries.append({"name": "..", "path": str(base.parent), "dir": True})

    for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        try:
            if child.is_dir():
                entries.append({"name": child.name, "path": str(child), "dir": True})
        except OSError:
            continue
    return entries


# ---------------------------------------------------------------------------
# Running WhisperJAV
# ---------------------------------------------------------------------------

def _terminate_tree(proc: subprocess.Popen) -> None:
    """
    Stop `proc` and everything it spawned.

    WhisperJAV cannot resume a partial transcription, so a Pause or Stop means
    discarding the in-flight work -- the point here is to make sure no orphan
    is left behind holding the GPU.  Ask politely first, then insist.

    The process-group id is captured up front, while the child is still alive:
    once `wait()` has reaped it, `os.getpgid(proc.pid)` raises and the final
    sweep would silently do nothing, leaving the grandchildren running.
    """
    pgid = None
    if os.name != "nt":
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None

    def signal_group(sig) -> None:
        if pgid is None:
            return
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    if os.name == "nt":
        # taskkill walks the child tree, which send_signal alone will not do.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        signal_group(signal.SIGTERM)

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()

    # Whether or not the direct child went quietly, sweep the group: helpers it
    # spawned do not necessarily die with their parent.
    if os.name != "nt":
        signal_group(signal.SIGKILL)

    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def run_whisperjav(binary: str, args: list[str],
                   on_line: Optional[Callable[[str], None]] = None,
                   should_abort: Optional[Callable[[], bool]] = None,
                   poll_seconds: float = 2.0) -> tuple[int, str, bool]:
    """
    Run whisperjav, capturing output and streaming it to `on_line`.

    `should_abort` is polled while the process runs; when it returns True the
    child is terminated (then killed if it ignores that).  This is how a Pause
    or Stop from the dashboard reaches an in-flight transcription -- WhisperJAV
    has no resume, so the partial work is simply discarded and the file goes
    back on the queue.

    Returns (exit_code, combined_output, aborted).
    """
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    captured: list[str] = []
    lock = threading.Lock()

    # Put the child in its own process group / session.  WhisperJAV spawns
    # helpers of its own (translation backends, workers), and signalling only
    # the direct child would orphan those -- leaving them holding GPU memory
    # long after a Pause.  Owning the whole group lets us take them all down.
    spawn_kwargs: dict = {}
    if os.name == "nt":
        spawn_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        spawn_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(
            [binary] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            **spawn_kwargs,
        )
    except FileNotFoundError:
        return 1, f"whisperjav executable not found at '{binary}'", False
    except OSError as exc:
        return 1, f"could not start whisperjav: {exc}", False

    def drain(stream):
        for line in stream:
            line = line.rstrip("\n")
            with lock:
                captured.append(line)
                # Bound memory on very chatty runs; keep the most recent output.
                if len(captured) > 4000:
                    del captured[:1000]
            if on_line and line.strip():
                try:
                    on_line(line)
                except Exception:
                    pass

    threads = [
        threading.Thread(target=drain, args=(proc.stdout,), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr,), daemon=True),
    ]
    for t in threads:
        t.start()

    aborted = False
    while True:
        try:
            proc.wait(timeout=poll_seconds)
            break
        except subprocess.TimeoutExpired:
            if should_abort and should_abort():
                aborted = True
                _terminate_tree(proc)
                break

    for t in threads:
        t.join(timeout=5)

    # The child may also die *because* an abort is under way without us being
    # the ones to kill it: a service stop delivers SIGTERM to whisperjav at the
    # same instant as to the agent, so it is often gone before the next poll
    # above can notice `should_abort`.  A non-zero exit while an abort was
    # requested says nothing about the file, so report it as the abort it is
    # rather than a failure (which would also plant a .whisper-failed marker).
    # A clean exit 0 in the same window is still a genuine result.
    if not aborted and proc.returncode != 0 and should_abort and should_abort():
        aborted = True

    with lock:
        output = "\n".join(captured)
    return proc.returncode if proc.returncode is not None else 1, output, aborted


def find_output_srt(video_dir: Path, basename: str, exclude: Path) -> Optional[Path]:
    """
    WhisperJAV names its output like:
        [basename].ja.pass1.english.srt   (translated -- preferred)
        [basename].ja.pass1.srt           (native)
    Returns the best match, or None.
    """
    try:
        candidates = sorted(
            p for p in video_dir.glob("*.srt")
            if p.name.startswith(basename + ".") and p != exclude
        )
    except OSError:
        return None

    for p in candidates:
        if p.name.endswith(".english.srt"):
            return p
    for p in candidates:
        if p.name.endswith(".pass1.srt"):
            return p
    return candidates[0] if candidates else None


def publish_srt(source_srt: Path, dest_srt: Path, audit_text: str) -> None:
    """
    Write the final subtitle atomically: the audit comment block is prepended
    in memory, the result goes to a temp file in the same directory, and
    os.replace swaps it into place in one step.

    This matters for a shared directory -- a reader (or the other machine's
    skip check) can only ever see the complete file or no file, never a
    half-written one.
    """
    original = _read_text_tolerant(source_srt)
    wrapped = "\n".join(f"{{/ {line} /}}" for line in audit_text.splitlines())
    content = wrapped + "\n\n" + original

    tmp = dest_srt.with_name(dest_srt.name + f".tmp-{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(str(tmp), str(dest_srt))
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
