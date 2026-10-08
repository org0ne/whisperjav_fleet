"""maybe_finish_job: a running job finishes once nothing is outstanding --
including a job the scan left with no items at all (every video already
subtitled).  Stdlib only, on a throwaway DB.

    python3 -m unittest -v tests.test_job_finish
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


class JobFinish(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        db.configure(Path(self.tmp) / "t.sqlite3")

    def job(self, state="running", auto_delete=0) -> int:
        with db.connect() as c:
            cur = c.execute(
                """
                INSERT INTO jobs (name, target_dir, args_template, lang_code, state,
                                  auto_delete_on_finish, created_at, updated_at)
                VALUES ('j', '/x', '{INPUT_FILE}', 'qu', ?, ?, 1.0, 1.0)
                """,
                (state, auto_delete),
            )
            return cur.lastrowid

    def item(self, job_id: int, state: str) -> None:
        with db.connect() as c:
            c.execute(
                "INSERT INTO work_items (job_id, rel_path, state, updated_at) VALUES (?, ?, ?, 1.0)",
                (job_id, f"{state}-{os.urandom(2).hex()}.mp4", state),
            )

    def state(self, job_id: int):
        with db.connect() as c:
            row = c.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return row["state"] if row else None

    def finish(self, job_id: int) -> None:
        with db.connect() as c:
            db.maybe_finish_job(c, job_id)

    def test_empty_running_job_is_done(self):
        j = self.job()
        self.finish(j)
        self.assertEqual(self.state(j), "done")

    def test_empty_running_job_auto_deletes(self):
        j = self.job(auto_delete=1)
        self.finish(j)
        self.assertIsNone(self.state(j))

    def test_pending_item_keeps_it_running(self):
        j = self.job()
        self.item(j, "pending")
        self.finish(j)
        self.assertEqual(self.state(j), "running")

    def test_all_finished_is_done(self):
        j = self.job()
        self.item(j, "done"); self.item(j, "skipped")
        self.finish(j)
        self.assertEqual(self.state(j), "done")

    def test_failure_blocks_auto_delete(self):
        j = self.job(auto_delete=1)
        self.item(j, "failed")
        self.finish(j)
        self.assertEqual(self.state(j), "done")

    def test_only_running_jobs_finish(self):
        for st in ("draft", "scanning", "paused", "stopped"):
            j = self.job(state=st)
            self.finish(j)
            self.assertEqual(self.state(j), st)


if __name__ == "__main__":
    unittest.main()
