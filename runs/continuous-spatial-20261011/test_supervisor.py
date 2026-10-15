"""State-machine acceptance tests; fake backend never starts research."""
import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock
import supervisor as S

class Fake:
    def __init__(self):
        self.info = {"LoadState": "not-found", "ActiveState": "inactive"}
        self.starts = []
        self.stops = 0
        self.populated = False
    def show(self):
        return dict(self.info)
    def empty(self, info):
        return not self.populated
    def start(self, root, run):
        assert not self.populated
        self.starts.append(run.name)
        self.populated = True
        self.info = {"LoadState": "loaded", "ActiveState": "active", "SubState": "running",
                     "Result": "success", "ExecMainStatus": "0", "InvocationID": run.name}
        return self.show()
    def stop(self):
        self.stops += 1
        self.populated = False
        self.info = {"LoadState": "not-found", "ActiveState": "inactive"}
        return self.show()
    def finish(self, status=0):
        self.populated = False
        self.info.update(ActiveState="active" if not status else "failed",
                         SubState="exited" if not status else "failed",
                         Result="success" if not status else "exit-code", ExecMainStatus=str(status))

class Acceptance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "runs").mkdir()
        (self.root / "cancel").mkdir()
        self.plan = copy.deepcopy(json.loads(pathlib.Path("PLAN.json").read_text()))
        # Unit tests isolate resource gates; scientific validator uses real frozen configs.
        self.plan["code_sha256"] = {}
        self.plan["dataset_files"] = {}
        self.plan["protected_stops"] = {}
        for config in self.plan["experiments"]:
            config["code_sha256"] = {}
            config["fingerprint"] = S.fingerprint(config)
        S.atomic(self.root / "PLAN.json", self.plan)
        self.backend = Fake()
        self.queue = S.Queue(self.root, self.backend)
        self.disk = mock.patch.object(S.shutil, "disk_usage", return_value=type("Usage", (), {"free": 20 * 1024 ** 3})())
        self.disk.start()
    def tearDown(self):
        self.disk.stop()
        self.tmp.cleanup()
    def launch(self):
        self.queue.tick()
        return self.queue.state["active"]
    def finish(self):
        fp = self.queue.state["active"]
        S.atomic(self.root / "runs" / fp / "SUMMARY.json", {"fingerprint": fp})
        self.backend.finish()
        self.queue.tick()
        return fp
    def test_distinct_sequential_jobs_and_atomic_terminal(self):
        first = self.launch()
        self.queue.tick()
        self.assertEqual(len(self.backend.starts), 1)
        self.finish()
        receipt = S.read(self.root / "runs" / first / "TERMINAL.json")
        self.assertTrue(receipt["job_tree_empty"])
        second = self.launch()
        self.assertNotEqual(first, second)
        self.assertNotEqual(self.plan["experiments"][0]["params"], self.plan["experiments"][1]["params"])
    def test_duplicate_global_flock(self):
        lockfile = self.root / "lock"
        first = S.acquire_lock(lockfile)
        self.assertIsNotNone(first)
        self.assertIsNone(S.acquire_lock(lockfile))
        first.close()
        third = S.acquire_lock(lockfile)
        self.assertIsNotNone(third)
        third.close()
    def test_restart_adopts_live_job_without_second_launch(self):
        first = self.launch()
        restarted = S.Queue(self.root, self.backend)
        restarted.tick()
        self.assertEqual(restarted.state["active"], first)
        self.assertEqual(len(self.backend.starts), 1)
    def test_unowned_orphan_blocks_new_job(self):
        self.backend.populated = True
        self.backend.info = {"ActiveState": "active", "SubState": "running"}
        self.queue.tick()
        self.assertEqual(self.queue.state["status"], "blocked_orphan")
        self.assertFalse(self.backend.starts)
    def test_stop_survives_restart_and_kills_job(self):
        first = self.launch()
        (self.root / "STOP").touch()
        restarted = S.Queue(self.root, self.backend)
        restarted.tick()
        self.assertFalse(self.backend.populated)
        self.assertEqual(restarted.state["status"], "paused_stop")
        self.assertEqual(restarted.state["outcomes"][first], "cancelled")
        restarted.tick()
        self.assertEqual(len(self.backend.starts), 1)
    def test_individual_cancel_skips_and_advances(self):
        fp = self.plan["experiments"][0]["fingerprint"]
        (self.root / "cancel" / fp).touch()
        second = self.launch()
        self.assertNotEqual(fp, second)
        self.assertEqual(self.queue.state["outcomes"][fp], "cancelled_before_start")
    def test_crash_between_receipt_and_state_never_repeats(self):
        fp = self.launch()
        run = self.root / "runs" / fp
        S.atomic(run / "TERMINAL.json", {"fingerprint": fp, "outcome": "completed"})
        restarted = S.Queue(self.root, self.backend)
        restarted.tick()
        restarted.tick()
        self.assertEqual(restarted.state["outcomes"][fp], "completed")
        self.assertEqual(self.backend.starts.count(fp), 1)
    def test_failed_job_is_not_retried_and_three_failures_open_circuit(self):
        for _ in range(3):
            self.queue.state["not_before"] = 0
            self.launch()
            self.backend.finish(status=1)
            self.queue.tick()
        self.queue.state["not_before"] = 0
        self.queue.tick()
        self.assertEqual(self.queue.state["status"], "blocked_failure_circuit")
        self.assertEqual(len(self.backend.starts), 3)
        self.assertEqual(len(set(self.backend.starts)), 3)
    def test_failure_backoff(self):
        self.launch()
        self.backend.finish(status=1)
        self.queue.tick()
        self.queue.tick()
        self.assertEqual(self.queue.state["status"], "failure_backoff")
        self.assertEqual(len(self.backend.starts), 1)
    def test_queue_exhaustion_is_paused(self):
        self.queue.state["outcomes"] = {c["fingerprint"]: "completed" for c in self.plan["experiments"]}
        self.queue.tick()
        self.assertEqual(self.queue.state["status"], "paused_exhausted")
        self.assertFalse(self.backend.starts)
    def test_duplicate_scientific_fingerprint_blocks(self):
        self.queue.plan["experiments"].append(self.queue.plan["experiments"][0])
        self.queue.tick()
        self.assertEqual(self.queue.state["status"], "blocked_invalid_queue")
    def test_disk_floor_blocks_without_launch(self):
        self.disk.stop()
        with mock.patch.object(S.shutil, "disk_usage", return_value=type("Usage", (), {"free": 1})()):
            self.queue.tick()
        self.assertEqual(self.queue.state["reason"], "disk_floor")
        self.assertFalse(self.backend.starts)
    def test_unsafe_model_call_rejected(self):
        config = self.queue.plan["experiments"][0]
        config["remote_maker_calls"] = 1
        config["fingerprint"] = S.fingerprint(config)
        self.queue.tick()
        self.assertEqual(self.queue.state["status"], "blocked_invalid_queue")
    def test_incomplete_launch_intent_is_interrupted_not_replayed(self):
        config = self.plan["experiments"][0]
        run = self.root / "runs" / config["fingerprint"]
        run.mkdir()
        S.atomic(run / "CONFIG.json", config)
        self.queue.tick()
        self.assertEqual(self.queue.state["outcomes"][config["fingerprint"]], "interrupted")
        self.assertFalse(self.backend.starts)

if __name__ == "__main__":
    unittest.main(verbosity=2)

