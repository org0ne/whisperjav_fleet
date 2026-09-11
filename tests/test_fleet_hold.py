"""Fleet-wide hold tests.

The hub tells every agent to stop claiming (e.g. while a SnapRAID sync runs)
through the `hold` field of the check-in response.  Runs the real Agent
against the fake hub from test_gpu_hold, with the Jasna bridge reporting
nothing so only the fleet hold is in play.

    python3 -m unittest -v tests.test_fleet_hold
"""
from __future__ import annotations

import os
import time
import unittest

from tests.test_gpu_hold import Harness, pid_alive, wait_for

HOLD = {"reason": "snapraid sync", "until": time.time() + 3600, "remaining": 3600.0,
        "set_at": time.time(), "set_by": "test"}


class FleetHold(unittest.TestCase):
    def test_no_claim_while_held_then_resume_on_release(self):
        h = Harness(mode="finish", release=0)
        h.hub.hold = dict(HOLD)
        h.start()
        try:
            time.sleep(4)
            self.assertEqual(h.hub.claims, 0, "must not claim while the fleet is held")
            st = h.agent.status()
            self.assertEqual(st["state"], "holding")
            self.assertEqual(st["fleet_hold"], "snapraid sync")
            self.assertEqual(st["gpu_hold"], "", "the hold is the fleet's, not Jasna's")
            h.hub.hold = None
            self.assertTrue(wait_for(lambda: h.hub.claims >= 1, 10), "claims resume once released")
            self.assertTrue(wait_for(lambda: os.path.exists(h.pidfile), 10))
            self.assertEqual(h.agent.status()["fleet_hold"], "")
        finally:
            h.stop()

    def test_hold_arriving_mid_file_lets_it_finish(self):
        h = Harness(mode="finish", release=0)
        h.start()
        try:
            self.assertTrue(wait_for(lambda: os.path.exists(h.pidfile), 15), "should start a file")
            h.hub.hold = dict(HOLD)
            time.sleep(4)
            self.assertEqual(h.hub.requeues, [], "a fleet hold never aborts the file in hand")
            self.assertTrue(pid_alive(h.pidfile), "whisperjav keeps running")
            # It is reported as working, not holding, while the file runs.
            self.assertEqual(h.agent.status()["state"], "working")
            self.assertEqual(h.agent.status()["fleet_hold"], "snapraid sync")
        finally:
            h.stop()

    def test_abort_in_flight_requeues_and_does_not_reclaim(self):
        h = Harness(mode="finish", release=0)
        h.start()
        try:
            self.assertTrue(wait_for(lambda: os.path.exists(h.pidfile), 15), "should start a file")
            h.hub.hold = dict(HOLD, abort_in_flight=True)
            self.assertTrue(wait_for(lambda: h.hub.requeues, 10), "the heartbeat abort requeues the file")
            self.assertIn("fleet hold: snapraid sync", h.hub.requeues[0])
            self.assertTrue(wait_for(lambda: not pid_alive(h.pidfile), 10), "whisperjav is killed")
            claims = h.hub.claims
            time.sleep(3)
            self.assertEqual(h.hub.claims, claims, "nothing is reclaimed while the hold stands")
            self.assertEqual(h.agent.status()["state"], "holding")
            self.assertEqual(h.agent.counts["yielded"], 0, "a fleet-hold abort is not a Jasna yield")
        finally:
            h.stop()

    def test_hold_without_reason_still_holds(self):
        h = Harness(mode="finish", release=0)
        h.hub.hold = {"reason": "", "until": time.time() + 60, "remaining": 60.0}
        h.start()
        try:
            time.sleep(3)
            self.assertEqual(h.hub.claims, 0)
            self.assertEqual(h.agent.status()["fleet_hold"], "no reason given")
        finally:
            h.stop()


if __name__ == "__main__":
    unittest.main()
