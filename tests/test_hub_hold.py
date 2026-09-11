"""Hub-side fleet hold: the daily window math and how it merges with the
ad-hoc hold.  Stdlib only (hub/db.py needs no FastAPI), on a throwaway DB
with a frozen clock.

    python3 -m unittest -v tests.test_hub_hold
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hub"))
import db  # noqa: E402


def local_epoch(hh: int, mm: int, day_offset: int = 0) -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + day_offset, hh, mm, 0, 0, 0, -1))


class WindowMath(unittest.TestCase):
    def test_inside_and_outside(self):
        t = local_epoch(3, 10)
        self.assertIsNotNone(db.window_bounds("02:55", 65, t))
        st, en = db.window_bounds("02:55", 65, t)
        self.assertEqual((st, en), (local_epoch(2, 55), local_epoch(2, 55) + 65 * 60))
        self.assertIsNone(db.window_bounds("02:55", 65, local_epoch(4, 0)), "end is exclusive")
        self.assertIsNone(db.window_bounds("02:55", 65, local_epoch(2, 54, 0) + 59))
        self.assertIsNotNone(db.window_bounds("02:55", 65, local_epoch(2, 55)), "start is inclusive")

    def test_crosses_midnight(self):
        t = local_epoch(0, 15)
        b = db.window_bounds("23:30", 90, t)
        self.assertIsNotNone(b, "yesterday's 23:30 window still covers 00:15")
        self.assertEqual(b[0], local_epoch(23, 30, -1))

    def test_bad_input(self):
        self.assertIsNone(db.window_bounds("25:00", 10, time.time()))
        self.assertIsNone(db.window_bounds("nope", 10, time.time()))
        self.assertIsNone(db.window_bounds("02:00", 0, time.time()))

    def test_next_start(self):
        self.assertEqual(db.window_next_start("02:55", local_epoch(1, 0)), local_epoch(2, 55))
        self.assertEqual(db.window_next_start("02:55", local_epoch(3, 0)), local_epoch(2, 55, 1))


class HoldMerge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hubhold-")
        db.configure(Path(self.tmp) / "t.sqlite3")
        self.clock = [local_epoch(1, 0)]
        self._real_now = db.now
        db.now = lambda: self.clock[0]

    def tearDown(self):
        db.now = self._real_now

    def hold(self):
        with db.connect() as c:
            return db.fleet_hold(c)

    def test_nothing_by_default(self):
        self.assertIsNone(self.hold())

    def test_manual_hold_expires_and_logs(self):
        with db.connect() as c:
            db.set_fleet_hold(c, "disk swap", 600, "scott")
        h = self.hold()
        self.assertEqual(h["sources"], ["manual"]); self.assertEqual(h["remaining"], 600.0)
        self.assertFalse(h["abort_in_flight"])
        self.clock[0] += 601
        self.assertIsNone(self.hold())
        with db.connect() as c:
            msgs = [r["message"] for r in c.execute("SELECT message FROM events ORDER BY id DESC LIMIT 1")]
        self.assertIn("expired without release", msgs[0])

    def test_window_opens_closes_and_skips(self):
        with db.connect() as c:
            db.set_hold_window(c, enabled=True, start="02:55", minutes=65, reason="snapraid", abort_in_flight=True)
        self.assertIsNone(self.hold(), "01:00 is outside")
        self.clock[0] = local_epoch(3, 0)
        h = self.hold()
        self.assertEqual(h["sources"], ["window"]); self.assertTrue(h["abort_in_flight"])
        self.assertEqual(h["until"], local_epoch(2, 55) + 65 * 60); self.assertEqual(h["set_by"], "schedule")
        with db.connect() as c:
            self.assertEqual(db.clear_fleet_hold(c), ["window"])
        self.assertIsNone(self.hold(), "Release during the window skips today")
        with db.connect() as c:
            w = db.hold_window(c)
        self.assertTrue(w["skipped_today"]); self.assertEqual(w["next_start"], local_epoch(2, 55, 1))
        self.clock[0] = local_epoch(3, 0, 1)
        self.assertIsNotNone(self.hold(), "tomorrow's occurrence is not skipped")
        self.clock[0] = local_epoch(4, 0, 1)
        self.assertIsNone(self.hold(), "closed after 65 min")

    def test_merge_takes_later_deadline_and_either_abort(self):
        with db.connect() as c:
            db.set_hold_window(c, enabled=True, start="02:55", minutes=65, reason="window", abort_in_flight=False)
        self.clock[0] = local_epoch(3, 30)
        with db.connect() as c:
            db.set_fleet_hold(c, "hook", 3600, "hook", abort_in_flight=True)
        h = self.hold()
        self.assertEqual(h["sources"], ["manual", "window"])
        self.assertEqual(h["until"], local_epoch(3, 30) + 3600, "the hook's later deadline wins")
        self.assertTrue(h["abort_in_flight"]); self.assertIn("hook", h["reason"]); self.assertIn("window", h["reason"])
        with db.connect() as c:
            self.assertEqual(sorted(db.clear_fleet_hold(c)), ["manual", "window"])
        self.assertIsNone(self.hold())

    def test_extend_only_never_shortens(self):
        with db.connect() as c:
            db.set_fleet_hold(c, "manual 2h", 7200, "scott")
            db.set_fleet_hold(c, "hook tail", 900, "hook", extend_only=True)
            h = db.fleet_hold(c)
            self.assertEqual(h["remaining"], 7200.0); self.assertEqual(h["reason"], "manual 2h")
            db.set_fleet_hold(c, "hook long", 10000, "hook", extend_only=True)
            h = db.fleet_hold(c)
            self.assertEqual(h["remaining"], 10000.0); self.assertEqual(h["reason"], "hook long")
            # extend_only with abort still turns abort on for the standing hold
            db.set_fleet_hold(c, "x", 10, "hook", extend_only=True, abort_in_flight=True)
            self.assertTrue(db.fleet_hold(c)["abort_in_flight"])

    def test_window_validation(self):
        with db.connect() as c:
            with self.assertRaises(ValueError): db.set_hold_window(c, enabled=True, start="2:99", minutes=5, reason="", abort_in_flight=True)
            with self.assertRaises(ValueError): db.set_hold_window(c, enabled=True, start="02:00", minutes=0, reason="", abort_in_flight=True)


if __name__ == "__main__":
    unittest.main()
