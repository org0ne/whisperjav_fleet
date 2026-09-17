"""Per-agent runtime by job (`db.agent_job_runtimes`), and the per-attempt
start clock it depends on.  Stdlib only, throwaway DB, frozen clock.

    python3 -m unittest -v tests.test_agent_runtime
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hub"))
import db  # noqa: E402


class RuntimeByJob(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hubrt-")
        db.configure(Path(self.tmp) / "t.sqlite3")
        self.clock = [1_000_000.0]
        self._real_now = db.now
        db.now = lambda: self.clock[0]
        with db.connect() as c:
            for jid, name in ((1, "alpha"), (2, "beta")):
                c.execute("INSERT INTO jobs (id, name, target_dir, args_template, lang_code, state, created_at, updated_at)"
                          " VALUES (?, ?, '/m', '', 'ja', 'running', ?, ?)", (jid, name, db.now(), db.now()))
                for i in range(3):
                    c.execute("INSERT INTO work_items (job_id, rel_path, updated_at) VALUES (?, ?, ?)",
                              (jid, f"{name}/{i}.mp4", db.now()))

    def tearDown(self):
        db.now = self._real_now

    def tick(self, s):
        self.clock[0] += s

    def runtimes(self):
        with db.connect() as c:
            return db.agent_job_runtimes(c, db.now())

    def test_breakdown_current_first_then_recent(self):
        # A finishes two alpha files (100s, 300s) and one beta file (50s).
        for dur_s in (100, 300):
            it = db.claim_next("A")
            self.assertEqual(it["job_id"], 1)
            self.tick(dur_s)
            with db.connect() as c:
                self.assertTrue(db.complete_item(c, it["item_id"], "A", "done", "", dur_s))
        it = db.claim_next("A"); self.tick(50)  # alpha/2 (strict priority drains alpha first)
        with db.connect() as c:
            db.complete_item(c, it["item_id"], "A", "done", "", 50)
        it = db.claim_next("A"); self.tick(50)  # beta/0
        self.assertEqual(it["job_id"], 2)
        with db.connect() as c:
            db.complete_item(c, it["item_id"], "A", "done", "", 50)
        # B is mid-file on beta for 40s, and has never finished anything.
        itb = db.claim_next("B"); self.assertEqual(itb["job_id"], 2)
        self.tick(40)

        rt = self.runtimes()
        self.assertEqual(sorted(rt), ["A", "B"])
        a = rt["A"]
        self.assertEqual([j["job_name"] for j in a], ["beta", "alpha"], "most recently finished first when idle")
        alpha = a[1]
        self.assertEqual((alpha["done"], alpha["total_duration"], alpha["avg_duration"]), (3, 450, 150))
        self.assertIsNone(alpha["current"])
        self.assertEqual(a[0]["done"], 1)
        b = rt["B"]
        self.assertEqual(len(b), 1)
        self.assertEqual((b[0]["job_name"], b[0]["done"], b[0]["avg_duration"]), ("beta", 0, None))
        self.assertEqual(b[0]["current"], {"rel_path": itb["rel_path"], "elapsed": 40.0})

        # A picks up another beta file: its beta row moves first and goes live.
        ita = db.claim_next("A"); self.tick(7)
        a = self.runtimes()["A"]
        self.assertEqual(a[0]["job_name"], "beta")
        self.assertEqual(a[0]["current"], {"rel_path": ita["rel_path"], "elapsed": 7.0})
        self.assertEqual(a[0]["done"], 1)

    def test_start_clock_is_per_attempt(self):
        it = db.claim_next("A"); self.tick(500)
        with db.connect() as c:
            self.assertEqual(self.runtimes()["A"][0]["current"]["elapsed"], 500.0)
            self.assertTrue(db.requeue_item(c, it["item_id"], "A", "paused"))
            self.assertIsNone(c.execute("SELECT started_at FROM work_items WHERE id = ?", (it["item_id"],)).fetchone()[0])
        self.assertEqual(self.runtimes(), {}, "a requeued item is nobody's runtime")
        self.tick(3600)
        again = db.claim_next("A")
        self.assertEqual(again["item_id"], it["item_id"])
        self.tick(12)
        self.assertEqual(self.runtimes()["A"][0]["current"]["elapsed"], 12.0,
                         "elapsed counts from the new claim, not the paused attempt an hour ago")
        # A lapsed lease resets it the same way.
        with db.connect() as c:
            c.execute("UPDATE work_items SET claim_expires = ? WHERE id = ?", (db.now() - 1, it["item_id"]))
            self.assertEqual(db.release_expired_claims(c), 1)
            self.assertIsNone(c.execute("SELECT started_at FROM work_items WHERE id = ?", (it["item_id"],)).fetchone()[0])


if __name__ == "__main__":
    unittest.main()
