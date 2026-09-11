"""GPU hold / yield-to-Jasna tests.

Runs the real Agent in-process against a fake hub, a fake stash-jasna-bridge
health endpoint, and a fake `whisperjav` that just sleeps. Exercises: the
claim gate, mid-file yield (abort + requeue as "yielded to Jasna"), the
"finish" mode, the release hysteresis, and fail-open when the bridge is down.

    python3 -m unittest -v tests.test_gpu_hold
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
import agent as agent_mod  # noqa: E402
from agent import Agent, GpuHold  # noqa: E402


def _serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class FakeHealth:
    """Mimics GET /jasna/health from the bridge."""
    def __init__(self):
        self.active = False
        self.preparing = None
        self.running = False
        self.up = True

    def start(self):
        state = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if not state.up:
                    self.send_response(503); self.end_headers(); return
                body = json.dumps({"active": state.active, "preparing": state.preparing,
                                   "jasna": {"running": state.running}}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        self.srv = _serve(H)
        return f"http://127.0.0.1:{self.srv.server_address[1]}/jasna/health"


class FakeHub:
    """Enough of the hub API for one item: checkin/claim/heartbeat/requeue/complete."""
    def __init__(self, item):
        self.item = item
        self.ignore_jasna = False
        self.claims = 0
        self.heartbeats = 0
        self.requeues: list[str] = []
        self.completes: list[dict] = []
        self.lock = threading.Lock()

    def start(self):
        hub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
                path = self.path
                with hub.lock:
                    if path.endswith("/checkin"):
                        return self._json({"paused": False, "ignore_jasna": hub.ignore_jasna, "browse": None, "scan": None})
                    if path.endswith("/claim"):
                        hub.claims += 1
                        item, hub.item = hub.item, None   # hand it out once
                        return self._json({"item": item})
                    if path.endswith("/heartbeat"):
                        hub.heartbeats += 1
                        return self._json({"ok": True, "abort": False, "reason": ""})
                    if path.endswith("/requeue"):
                        hub.requeues.append(payload.get("reason", ""))
                        return self._json({"ok": True})
                    if path.endswith("/complete"):
                        hub.completes.append(payload)
                        return self._json({"ok": True})
                return self._json({})
        self.srv = _serve(H)
        return f"http://127.0.0.1:{self.srv.server_address[1]}"


def make_fake_whisperjav(tmp: str) -> str:
    """A whisperjav that answers --version, records its pid, then sleeps (must be killed)."""
    bin_path = os.path.join(tmp, "fake-whisperjav")
    with open(bin_path, "w") as fh:
        fh.write(f"""#!/bin/sh
if [ "$1" = "--version" ]; then echo "fake-whisperjav 1.0"; exit 0; fi
echo $$ > "{tmp}/whisperjav.pid"
sleep 300
""")
    os.chmod(bin_path, os.stat(bin_path).st_mode | stat.S_IEXEC)
    return bin_path


def pid_alive(pidfile: str) -> bool:
    try:
        pid = int(open(pidfile).read().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def wait_for(cond, timeout: float, step: float = 0.2) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return False


class Harness:
    def __init__(self, mode="yield", release=1.0, while_resident=True, health_url=None):
        self.tmp = tempfile.mkdtemp(prefix="gpuhold-")
        target = os.path.join(self.tmp, "media"); os.makedirs(target)
        open(os.path.join(target, "clip.mp4"), "wb").write(b"\x00" * 16)
        self.pidfile = os.path.join(self.tmp, "whisperjav.pid")
        self.health = FakeHealth()
        hurl = health_url if health_url is not None else self.health.start()
        item = {"item_id": 7, "job_id": 1, "rel_path": "clip.mp4", "attempts": 0,
                "job": {"id": 1, "name": "t", "target_dir": target, "args_template": "{INPUT_FILE}",
                        "lang_code": "qu", "force": False, "dry_run": False, "video_extensions": [".mp4"]}}
        self.hub = FakeHub(item)
        cfg = dict(agent_mod.DEFAULT_CONFIG)
        cfg.update({"agent_id": "t-agent", "hub_url": self.hub.start(), "whisperjav_bin": make_fake_whisperjav(self.tmp),
                    "poll_interval_seconds": 0.5, "idle_poll_interval_seconds": 0.5, "heartbeat_seconds": 1,
                    "ollama_unload_on_abort": False,
                    "gpu_hold_url": hurl, "gpu_hold_mode": mode, "gpu_hold_poll_seconds": 1,
                    "gpu_hold_while_resident": while_resident, "gpu_hold_release_seconds": release})
        self.agent = Agent(cfg)
        self.thread = threading.Thread(target=self.agent.run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.agent.shutdown.set(); self.agent._abort.set()
        self.thread.join(timeout=10)
        for s in (getattr(self.health, "srv", None), self.hub.srv):
            if s: s.shutdown(); s.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)


class GpuHoldUnit(unittest.TestCase):
    def test_fail_open_when_bridge_unreachable(self):
        g = GpuHold({"gpu_hold_url": "http://127.0.0.1:1/jasna/health", "gpu_hold_release_seconds": 0})
        self.assertEqual(g.probe(), (False, ""))
        self.assertFalse(g.blocking())

    def test_signals_and_hysteresis(self):
        h = FakeHealth(); url = h.start()
        g = GpuHold({"gpu_hold_url": url, "gpu_hold_release_seconds": 0.6, "gpu_hold_while_resident": True})
        self.assertEqual(g.probe(), (False, ""))
        h.running = True;   self.assertEqual(g.probe()[1], "Jasna resident, holding VRAM")
        h.preparing = {"scene_id": "1"}; self.assertEqual(g.probe()[1], "session starting")
        h.active = True;    self.assertEqual(g.probe()[1], "viewer active")
        h.active = False; h.preparing = None; h.running = False
        self.assertFalse(g.probe()[0])
        self.assertTrue(g.blocking(), "release window should still block right after release")
        time.sleep(0.7)
        self.assertFalse(g.blocking())
        g2 = GpuHold({"gpu_hold_url": url, "gpu_hold_while_resident": False, "gpu_hold_release_seconds": 0})
        h.running = True; self.assertFalse(g2.probe()[0], "resident-only must not hold when disabled")
        h.srv.shutdown(); h.srv.server_close()

    def test_disabled_when_url_blank(self):
        g = GpuHold({"gpu_hold_url": ""})
        self.assertFalse(g.enabled); self.assertFalse(g.blocking())


class YieldMode(unittest.TestCase):
    def test_mid_file_yield_then_no_reclaim_then_resume(self):
        h = Harness(mode="yield", release=1.0).start()
        try:
            self.assertTrue(wait_for(lambda: h.hub.claims >= 1 and os.path.exists(h.pidfile), 15), "should claim and start whisperjav")
            self.assertTrue(pid_alive(h.pidfile))
            self.assertTrue(wait_for(lambda: h.hub.heartbeats >= 1, 5))
            # Jasna starts: a session is being prepared
            h.health.preparing = {"scene_id": "34487"}; h.health.running = True
            self.assertTrue(wait_for(lambda: h.hub.requeues, 15), "file should be requeued")
            self.assertEqual(h.hub.requeues[0], "yielded to Jasna")
            self.assertTrue(wait_for(lambda: not pid_alive(h.pidfile), 10), "whisperjav should be killed")
            self.assertEqual(h.agent.counts["yielded"], 1)
            # still held: no new claim
            c = h.hub.claims; time.sleep(3)
            self.assertEqual(h.hub.claims, c, "must not claim while Jasna holds the GPU")
            self.assertEqual(h.agent.status()["state"], "holding")
            # Jasna gone: claims resume after the release window
            h.health.preparing = None; h.health.running = False; h.health.active = False
            self.assertTrue(wait_for(lambda: h.hub.claims > c, 10), "claims should resume once released")
            self.assertEqual(h.agent.status()["state"], "idle")
        finally:
            h.stop()

    def test_active_viewer_also_yields(self):
        h = Harness(mode="yield", release=0).start()
        try:
            self.assertTrue(wait_for(lambda: os.path.exists(h.pidfile), 15))
            h.health.active = True
            self.assertTrue(wait_for(lambda: h.hub.requeues, 15))
            self.assertEqual(h.hub.requeues[0], "yielded to Jasna")
        finally:
            h.stop()


class FinishMode(unittest.TestCase):
    def test_finish_mode_lets_current_file_run(self):
        h = Harness(mode="finish", release=0).start()
        try:
            self.assertTrue(wait_for(lambda: os.path.exists(h.pidfile), 15))
            h.health.active = True
            time.sleep(5)
            self.assertEqual(h.hub.requeues, [], "finish mode must not abort the in-flight file")
            self.assertTrue(pid_alive(h.pidfile), "whisperjav should still be running")
            # finish mode runs no mid-file watcher, so the cached flag only
            # refreshes at the claim gate; probe explicitly to confirm the
            # signal itself is visible.
            self.assertTrue(h.agent._gpu_hold.probe()[0])
        finally:
            h.stop()
            self.assertTrue(wait_for(lambda: h.hub.requeues, 10))
            self.assertEqual(h.hub.requeues[0], "aborted by pause/stop")


class ClaimGate(unittest.TestCase):
    def test_no_claim_while_held_from_the_start(self):
        h = Harness(mode="yield", release=0)
        h.health.running = True
        h.start()
        try:
            time.sleep(4)
            self.assertEqual(h.hub.claims, 0, "must not claim while Jasna is resident")
            self.assertEqual(h.agent.status()["state"], "holding")
            self.assertIn("resident", h.agent.status()["gpu_hold"])
        finally:
            h.stop()

    def test_ignore_override_claims_while_jasna_held(self):
        h = Harness(mode="yield", release=0)
        h.health.running = True          # Jasna resident -> would normally hold
        h.start()
        try:
            time.sleep(3)
            self.assertEqual(h.hub.claims, 0, "held before the override")
            h.hub.ignore_jasna = True     # hub flips the per-agent override
            self.assertTrue(wait_for(lambda: h.hub.claims >= 1, 10),
                            "with ignore on, the agent claims despite Jasna holding")
            self.assertTrue(wait_for(lambda: os.path.exists(h.pidfile), 10))
        finally:
            h.stop()

    def test_ignore_override_prevents_mid_file_yield(self):
        h = Harness(mode="yield", release=0)
        h.hub.ignore_jasna = True
        h.start()
        try:
            self.assertTrue(wait_for(lambda: os.path.exists(h.pidfile), 15), "should start a file")
            h.health.active = True        # Jasna starts; normally this yields
            time.sleep(5)
            self.assertEqual(h.hub.requeues, [], "ignore override must not yield the in-flight file")
            self.assertTrue(pid_alive(h.pidfile), "whisperjav keeps running")
        finally:
            h.stop()

    def test_bridge_down_does_not_block(self):
        h = Harness(mode="yield", release=0)
        h.health.up = False
        h.start()
        try:
            self.assertTrue(wait_for(lambda: h.hub.claims >= 1, 10), "unreachable bridge must fail open")
        finally:
            h.stop()


def _fake_nvidia_smi(line: str) -> str:
    """Write a fake nvidia-smi to a temp dir and return its path."""
    d = tempfile.mkdtemp(prefix="fakesmi-")
    p = os.path.join(d, "nvidia-smi")
    with open(p, "w") as f:
        f.write("#!/bin/sh\nprintf '%s\\n' " + repr(line) + "\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


class GpuSamplerTest(unittest.TestCase):
    def test_samples_and_snapshots(self):
        smi = _fake_nvidia_smi("73, 12000, 24564, NVIDIA GeForce RTX 4090")
        s = agent_mod.GpuSampler({"nvidia_smi_bin": smi, "gpu_sample_seconds": 1})
        sd = threading.Event()
        s.start(sd)
        try:
            self.assertTrue(wait_for(lambda: s.snapshot() is not None, 5), "should produce a sample")
            snap = s.snapshot()
            self.assertEqual(snap["util"], 73)
            self.assertEqual(snap["mem_used"], 12000)
            self.assertEqual(snap["mem_total"], 24564)
            self.assertIn("RTX 4090", snap["name"])
            self.assertTrue(snap["history"] and snap["history"][-1] == 73)
        finally:
            sd.set()

    def test_history_is_bounded(self):
        smi = _fake_nvidia_smi("50, 1, 24564, GPU")
        s = agent_mod.GpuSampler({"nvidia_smi_bin": smi, "gpu_sample_seconds": 0.05})
        s.HISTORY = 5
        sd = threading.Event()
        s.start(sd)
        try:
            self.assertTrue(wait_for(lambda: (s.snapshot() or {}).get("history") and
                                     len(s.snapshot()["history"]) == 5, 5),
                            "ring buffer must cap at HISTORY")
            time.sleep(0.3)
            self.assertLessEqual(len(s.snapshot()["history"]), 5, "must not grow past cap")
        finally:
            sd.set()

    def test_missing_binary_reports_unavailable(self):
        s = agent_mod.GpuSampler({"nvidia_smi_bin": "/no/such/nvidia-smi"})
        sd = threading.Event()
        s.start(sd)                 # thread must not spin without a binary
        time.sleep(0.3)
        self.assertIsNone(s.snapshot(), "no binary -> no graphic")
        sd.set()

    def test_disabled_reports_nothing(self):
        smi = _fake_nvidia_smi("50, 1, 24564, GPU")
        s = agent_mod.GpuSampler({"nvidia_smi_bin": smi, "gpu_sample": False})
        sd = threading.Event()
        s.start(sd)
        time.sleep(0.3)
        self.assertIsNone(s.snapshot(), "gpu_sample=false -> no graphic")
        sd.set()

    def test_bad_output_reports_unavailable(self):
        smi = _fake_nvidia_smi("No devices were found")
        s = agent_mod.GpuSampler({"nvidia_smi_bin": smi, "gpu_sample_seconds": 0.1})
        sd = threading.Event()
        s.start(sd)
        time.sleep(0.4)
        self.assertIsNone(s.snapshot(), "unparseable output must not crash or emit")
        sd.set()


if __name__ == "__main__":
    unittest.main()
