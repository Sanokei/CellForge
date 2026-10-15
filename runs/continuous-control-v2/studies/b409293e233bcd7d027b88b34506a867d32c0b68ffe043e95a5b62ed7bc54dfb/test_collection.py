"""Independent failure/collection regressions; backend preserves failed metadata."""
import pathlib
import unittest
from unittest.mock import patch
import test_review as F
import supervisor as V
from common import *

class FailureCollection(unittest.TestCase):
    setUp=F.ControllerCrashPoints.setUp
    tearDown=F.ControllerCrashPoints.tearDown
    enable=F.ControllerCrashPoints.enable
    def fail_active(self):
        self.enable(); self.controller.tick()
        self.backend.value.update(ActiveState="failed",SubState="failed",MainPID="0",ControlGroup="",
                                  Result="exit-code",ExecMainStatus="2")
    def test_backend_stop_preserves_failed_metadata_until_owned_collection(self):
        self.fail_active(); s=read(self.root/"REGISTRY.json"); entry=s["conditions"][self.fp]
        self.backend.stop(entry["token"],entry["invocation"])
        self.assertEqual(self.backend.show()["LoadState"],"loaded")
        self.assertEqual(self.backend.show()["ActiveState"],"failed"); self.assertEqual(self.backend.collections,0)
        self.backend.collect(entry["token"],entry["invocation"])
        self.assertEqual(self.backend.show()["LoadState"],"not-found"); self.assertEqual(self.backend.collections,1)
    def test_failure_backoff_then_next_distinct_condition(self):
        self.fail_active(); self.controller.tick()
        s=read(self.root/"REGISTRY.json"); run=self.controller.context(s,self.fp)[3]
        receipt=read(run/"TERMINAL.json"); receipt_hash=filehash(run/"TERMINAL.json")
        self.assertEqual(s["consecutive_failures"],1); self.assertIsNone(s["active"])
        self.assertEqual(self.backend.collections,1); self.assertEqual(self.backend.show()["LoadState"],"not-found")
        self.assertTrue(read(run/"COLLECTION.json")["unit_absent"])
        self.controller.tick()
        self.assertEqual(read(self.root/"REGISTRY.json")["status"],"failure_backoff")
        self.assertEqual(self.backend.starts,1)
        with patch("common.time.time",return_value=receipt["not_before_after"]+1): self.controller.tick()
        after=read(self.root/"REGISTRY.json")
        self.assertEqual(self.backend.starts,2); self.assertNotEqual(after["active"],self.fp)
        self.assertEqual(after["consecutive_failures"],1); self.assertEqual(after["sequence"],1)
        self.assertEqual(after["studies"][self.ph]["attempted"],2)
        self.assertEqual(filehash(run/"TERMINAL.json"),receipt_hash)
    def crash_case(self,point,prior_failures=0):
        self.fail_active()
        state=read(self.root/"REGISTRY.json"); state["consecutive_failures"]=prior_failures; atomic(self.root/"REGISTRY.json",state)
        def crash(actual):
            if actual==point: raise F.Crash(point)
        self.controller.crash=crash
        with self.assertRaises(F.Crash): self.controller.tick()
        state=read(self.root/"REGISTRY.json"); run=self.controller.context(state,self.fp)[3]
        receipt=read(run/"TERMINAL.json"); h=filehash(run/"TERMINAL.json")
        self.assertEqual(state["active"],self.fp); self.assertEqual(state["consecutive_failures"],prior_failures)
        V.Controller(self.root,self.backend).tick()
        after=read(self.root/"REGISTRY.json")
        self.assertIsNone(after["active"]); self.assertEqual(after["sequence"],1)
        self.assertEqual(after["consecutive_failures"],prior_failures+1)
        self.assertEqual(after["not_before"],receipt["not_before_after"])
        self.assertEqual(filehash(run/"TERMINAL.json"),h)
        self.assertEqual(self.backend.collections,1); self.assertEqual(self.backend.show()["LoadState"],"not-found")
        return after
    def test_crash_after_receipt_before_collection(self): self.crash_case("after_receipt_before_collection")
    def test_crash_immediately_before_reset(self): self.crash_case("before_failed_unit_collection")
    def test_crash_immediately_after_reset(self): self.crash_case("after_failed_unit_reset")
    def test_crash_after_collection_before_accounting(self): self.crash_case("after_collection_before_registry")
    def test_third_failure_after_reset_crash_keeps_circuit(self):
        self.crash_case("after_failed_unit_reset",prior_failures=2)
        V.Controller(self.root,self.backend).tick()
        self.assertEqual(read(self.root/"REGISTRY.json")["status"],"blocked_failure_circuit")
        self.assertEqual(self.backend.starts,1)
    def test_unknown_status_refuses_collection(self):
        self.fail_active()
        with patch.object(self.backend,"show",side_effect=Blocked("unknown transport")),self.assertRaises(Blocked):
            self.backend.collect("owner","invocation")
        self.assertEqual(self.backend.collections,0)
    def test_unowned_and_mismatched_invocation_refuse_collection(self):
        self.fail_active(); state=read(self.root/"REGISTRY.json"); entry=state["conditions"][self.fp]
        with self.assertRaises(Blocked): self.backend.collect("unowned",entry["invocation"])
        with self.assertRaises(Blocked): self.backend.collect(entry["token"],"another-invocation")
        self.backend.value["InvocationID"]=""
        with self.assertRaises(Blocked): self.backend.collect(entry["token"],entry["invocation"])
        self.assertEqual(self.backend.collections,0)
    def test_populated_or_unreadable_cgroup_refuses_collection(self):
        self.fail_active(); s=read(self.root/"REGISTRY.json"); entry=s["conditions"][self.fp]
        group=self.root/"cgroups"/V.GROUP.lstrip("/"); group.mkdir(parents=True)
        with self.assertRaises(Blocked): self.backend.collect(entry["token"],entry["invocation"])
        (group/"cgroup.events").write_text("populated 1\n")
        with self.assertRaises(Blocked): self.backend.collect(entry["token"],entry["invocation"])
        self.assertEqual(self.backend.collections,0)
    def test_owner_replacement_at_checkpoint_refuses_reset(self):
        self.fail_active(); s=read(self.root/"REGISTRY.json"); entry=s["conditions"][self.fp]
        def replaced(point):
            if point=="before_failed_unit_collection": self.backend.value["Description"]="cellforge-v2:someone-else"
        with self.assertRaises(Blocked): self.backend.collect(entry["token"],entry["invocation"],checkpoint=replaced)
        self.assertEqual(self.backend.collections,0)

if __name__=="__main__": unittest.main(verbosity=2)

