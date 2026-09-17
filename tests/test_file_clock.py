"""The per-attempt clock behind the agent card's "This file" row
(`db.current_item_started`).  Stdlib only, throwaway DB, frozen clock.

    python3 -m unittest -v tests.test_file_clock
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


class FileClock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hubclock-")
        db.configure(Path(self.tmp) / "t.sqlite3")
        self.clock = [1_000_000.0]
        self._real_now = db.now
        db.now = lambda: self.clock[0]
        with db.connect() as c:
            c.execute("INSERT INTO jobs (id, name, target_dir, args_template, lang_code, state, created_at, updated_at)"
                      " VALUES (1, 'j', '/m', '', 'ja', 'running', ?, ?)", (db.now(), db.now()))
            for i in range(2):
                c.execute("INSERT INTO work_items (job_id, rel_path, updated_at) VALUES (1, ?, ?)", (f"{i}.mp4", db.now()))

    def tearDown(self):
        db.now = self._real_now

    def elapsed(self, agent):
        with db.connect() as c:
            st = db.current_item_started(c, agent)
        return None if st is None else db.now() - st

    def test_counts_from_claim_and_stops_on_completion(self):
        self.assertIsNone(self.elapsed("A"))
        it = db.claim_next("A")
        self.clock[0] += 90
        self.assertEqual(self.elapsed("A"), 90)
        self.assertIsNone(self.elapsed("B"))
        with db.connect() as c:
            self.assertTrue(db.complete_item(c, it["item_id"], "A", "done", "", 90))
        self.assertIsNone(self.elapsed("A"))

    def test_per_attempt_after_requeue_and_lease_expiry(self):
        it = db.claim_next("A")
        self.clock[0] += 500
        with db.connect() as c:
            self.assertTrue(db.requeue_item(c, it["item_id"], "A", "paused"))
            self.assertIsNone(c.execute("SELECT started_at FROM work_items WHERE id = ?", (it["item_id"],)).fetchone()[0])
        self.assertIsNone(self.elapsed("A"))
        self.clock[0] += 3600
        again = db.claim_next("A")
        self.assertEqual(again["item_id"], it["item_id"])
        self.clock[0] += 12
        self.assertEqual(self.elapsed("A"), 12, "counts from the new claim, not the attempt paused an hour ago")
        with db.connect() as c:
            c.execute("UPDATE work_items SET claim_expires = ? WHERE id = ?", (db.now() - 1, it["item_id"]))
            self.assertEqual(db.release_expired_claims(c), 1)
        self.assertIsNone(self.elapsed("A"))


if __name__ == "__main__":
    unittest.main()
