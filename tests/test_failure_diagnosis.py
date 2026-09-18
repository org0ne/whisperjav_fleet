"""Failure diagnosis tests.

`runner.diagnose_failure` on real-world whisperjav output, and the agent's
handling of the two kinds of diagnosis: an environmental one (this worker is
broken -- requeue, back off, show degraded) and a file-level one (fail the
item, with the explanation in the .whisper-failed marker).

    python3 -m unittest -v tests.test_failure_diagnosis
"""
from __future__ import annotations

import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
import runner  # noqa: E402

from tests.test_gpu_hold import Harness, wait_for  # noqa: E402


HF_EXPIRED_TOKEN = """\
  File ".../transformers/utils/hub.py", line 511, in cached_files
    raise OSError(
OSError: jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
2026-09-18 11:39:05 - whisperjav - WARNING - [Worker 60936] Pass 1 had 1 failed file(s)
2026-09-18 11:39:05 - whisperjav - ERROR - Failed: DLDSS-478 - Unknown error
"""


class DiagnoseUnit(unittest.TestCase):
    def test_expired_hf_token_is_environmental_and_names_the_model(self):
        d = runner.diagnose_failure(HF_EXPIRED_TOKEN)
        self.assertIsNotNone(d)
        self.assertTrue(d.environmental)
        self.assertIn("jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame", d.label)
        self.assertIn("token", d.label)
        self.assertIn("huggingface/token", d.detail)
        self.assertIn("HF_HUB_OFFLINE", d.detail)
        self.assertNotIn("\n", d.label)

    def test_gpu_oom_is_not_environmental(self):
        d = runner.diagnose_failure("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
        self.assertIsNotNone(d)
        self.assertFalse(d.environmental)
        self.assertIn("out of memory", d.label)

    def test_missing_driver_after_reboot(self):
        d = runner.diagnose_failure("RuntimeError: Found no NVIDIA driver on your system.")
        self.assertTrue(d.environmental)
        self.assertIn("GPU not usable", d.label)

    def test_ollama_refused(self):
        out = ("requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=11434): "
               "Max retries exceeded with url: /api/chat (Caused by NewConnectionError(... Connection refused))")
        d = runner.diagnose_failure(out)
        self.assertTrue(d.environmental)
        self.assertIn("Ollama", d.label)

    def test_ollama_model_not_pulled(self):
        d = runner.diagnose_failure("ollama error: model 'translategemma:12b' not found, try pulling it first")
        self.assertTrue(d.environmental)
        self.assertIn("translategemma:12b", d.label)

    def test_broken_venv(self):
        d = runner.diagnose_failure("Traceback...\nModuleNotFoundError: No module named 'qwen_asr'\n")
        self.assertTrue(d.environmental)
        self.assertIn("qwen_asr", d.label)

    def test_unknown_output_is_none(self):
        self.assertIsNone(runner.diagnose_failure("something unexpected happened\nexit 1"))
        self.assertIsNone(runner.diagnose_failure(""))

    def test_audit_text_puts_diagnosis_before_output(self):
        from pathlib import Path
        text = runner.build_audit_text("v", "t", Path("/x/a.mp4"), "whisperjav a.mp4", 1.0,
                                       "raw output", diagnosis="the cause")
        self.assertLess(text.index("--- Diagnosis ---"), text.index("--- Output / Error ---"))
        self.assertIn("the cause", text)
        plain = runner.build_audit_text("v", "t", Path("/x/a.mp4"), "whisperjav a.mp4", 1.0, "raw output")
        self.assertNotIn("Diagnosis", plain)


def _script(text: str) -> str:
    return "cat <<'WJEOF'\n" + text + "WJEOF\nexit 1\n"


class DiagnoseAgent(unittest.TestCase):
    def test_environmental_failure_requeues_and_degrades(self):
        h = Harness(mode="finish", release=0, whisperjav_body=_script(HF_EXPIRED_TOKEN),
                    cfg_extra={"env_retry_seconds": 60}).start()
        try:
            self.assertTrue(wait_for(lambda: h.hub.requeues, 10), "item was not requeued")
            reason = h.hub.requeues[0]
            self.assertIn("degraded", reason)
            self.assertIn("jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame", reason)
            self.assertEqual(h.hub.completes, [], "an environmental failure must not be reported as failed")
            self.assertFalse(os.path.exists(os.path.join(h.media, "clip.whisper-failed")),
                             "no failure marker for a worker-side cause")
            st = h.agent.status()
            self.assertIn("token", st["degraded"])
            # The back-off: give the item back to the hub and make sure the
            # agent does not come for it within the hold.
            h.hub.item = {"item_id": 8, "job_id": 1, "rel_path": "clip.mp4", "attempts": 1,
                          "job": {"id": 1, "name": "t", "target_dir": h.media, "args_template": "{INPUT_FILE}",
                                  "lang_code": "qu", "force": False, "dry_run": False, "video_extensions": [".mp4"]}}
            claims = h.hub.claims
            time.sleep(2.5)
            self.assertEqual(h.hub.claims, claims, "agent claimed again inside env_retry_seconds")
            self.assertGreater(h.agent._env_down_until, time.time() + 30)
        finally:
            h.stop()

    def test_file_level_failure_fails_with_diagnosis_in_marker(self):
        h = Harness(mode="finish", release=0,
                    whisperjav_body=_script("torch.OutOfMemoryError: CUDA out of memory.\n")).start()
        try:
            self.assertTrue(wait_for(lambda: h.hub.completes, 10), "item was not completed")
            done = h.hub.completes[0]
            self.assertEqual(done["state"], "failed")
            self.assertEqual(done["message"], "exit code 1: GPU out of memory")
            self.assertEqual(h.hub.requeues, [])
            marker = os.path.join(h.media, "clip.whisper-failed")
            self.assertTrue(os.path.exists(marker))
            with open(marker) as fh:
                text = fh.read()
            self.assertIn("--- Diagnosis ---", text)
            self.assertIn("--- Output / Error ---", text)
            self.assertLess(text.index("--- Diagnosis ---"), text.index("--- Output / Error ---"))
            self.assertEqual(h.agent.status()["degraded"], "")
            self.assertEqual(h.agent._env_down_until, 0.0)
        finally:
            h.stop()

    def test_unrecognised_failure_is_unchanged(self):
        h = Harness(mode="finish", release=0, whisperjav_body="echo boom; exit 3").start()
        try:
            self.assertTrue(wait_for(lambda: h.hub.completes, 10))
            done = h.hub.completes[0]
            self.assertEqual(done["state"], "failed")
            self.assertEqual(done["message"], "exit code 3")
            with open(os.path.join(h.media, "clip.whisper-failed")) as fh:
                text = fh.read()
            self.assertNotIn("Diagnosis", text)
            self.assertIn("boom", text)
        finally:
            h.stop()


if __name__ == "__main__":
    unittest.main()
