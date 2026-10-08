"""Temporary adapter fixtures only; no model calls or real panel evaluation."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("comparison_adapter", ROOT / "scripts/nca_prompt_comparison_intake.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
TIME = datetime(2026, 10, 8, 8, 0, tzinfo=timezone.utc).timestamp()


def iso(seconds=0):
    return datetime.fromtimestamp(TIME + seconds, timezone.utc).isoformat()


class ComparisonAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.supplement = self.root / "ADAPTER.json"
        self.sha = M.freeze_supplement(ROOT, self.supplement)
        self.a = M.Adapter(ROOT, self.supplement, self.sha, self.root / "execution")
        self.clock = patch.object(M, "now", return_value=TIME)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def route(self, arm="A"):
        return {"model": "gpt-6.1-sol", "reasoning_effort": "low", "wall_seconds": 120,
                "memory_bytes": 2147483648, "concurrency": 1, "request_retries": 0, "stream_retries": 0,
                "websocket": False, "audit_mode": "complete_event_stream_conservative_allowlist",
                "input_isolation_status": "verified_packet_only_no_repo_global_context",
                "preventive_isolation_status": "not_claimed", "new_independent_session": True,
                "prior_weights_exposed": False, "controls_exposed": False, "cross_arm_feedback_exposed": False,
                "scientific_input_sha256": [self.a.packet_sha(arm), self.a.p.SCHEMA_SHA],
                "identity": "fixture-independent-" + arm, "attestation_reference": "fixture-route-" + arm,
                "isolated_role_directory": "/srv/cellforge-inbox/research-20261008/input-only-author-stable-v2",
                "observed_at": iso(), "capture_code_sha256": M.CAPTURE_SHA,
                "capture_config_sha256": M.CONFIG_SHAS[arm]}

    def quota(self):
        return {"authoritative": True, "ordinaryUsageAllowed": True, "usage_percent": 15,
                "period_id": "fixture-week", "reset_at": iso(3600), "observed_at": iso(),
                "attestation_reference": "fixture-only-quota"}

    def approval(self, arm="A"):
        return {"schema": "nca-comparison-parent-approval/1", "parent_approved": True,
                "parent_identity": "fixture-parent-no-real-authorization", "approval_reference": "fixture-approval-" + arm,
                "arm": arm, "preparation_contract_sha256": M.PREPARATION_SHA,
                "adapter_supplement_sha256": self.sha, "packet_sha256": self.a.packet_sha(arm),
                "response_schema_sha256": self.a.p.SCHEMA_SHA, "approved_max_invocations": 2,
                "approved_positions_each_arm": 2976, "route": self.route(arm), "quota": self.quota()}

    def initialize(self):
        return self.a.initialize(self.approval())

    def begin(self, arm="A"):
        if arm == "A":
            self.initialize()
        self.a.reserve(arm, self.approval(arm))
        self.a.submit(arm, "fixture-host-" + arm, self.route(arm), self.quota())

    def raw(self):
        tensors = {}
        for item in self.a.intake.tensor_schema():
            values = [0] * item["positions"]
            tensors[item["name"]] = [values] if len(item["shape"]) == 2 else values
        return self.a.p.json_bytes({"schema": "nca-author-tensors/1", "tensors": tensors})

    def receipts(self, raw=None, arm="A"):
        raw = self.raw() if raw is None else raw
        events, _, _ = self.a._read()
        reservation = self.a.event(events, "reserve", arm)
        submission = self.a.event(events, "submit", arm)
        metadata = [
            {"sequence": 0, "event_type": "thread.started", "item_type": None, "classification": "lifecycle", "event_sha256": "0" * 64, "final_raw_sha256": None},
            {"sequence": 1, "event_type": "turn.started", "item_type": None, "classification": "lifecycle", "event_sha256": "1" * 64, "final_raw_sha256": None},
            {"sequence": 2, "event_type": "item.completed", "item_type": "agent_message", "classification": "final_message", "event_sha256": "2" * 64, "final_raw_sha256": M.raw_hash(raw)},
            {"sequence": 3, "event_type": "turn.completed", "item_type": None, "classification": "lifecycle", "event_sha256": "3" * 64, "final_raw_sha256": None}]
        audit = {"schema": "nca-host-response-audit/1", "status": "completed", "host_capture": True,
                 "parent_identity": reservation["parent_identity"], "packet_sha256": reservation["packet_sha256"],
                 "raw_sha256": M.raw_hash(raw), "submission_sha256": M.digest(submission),
                 "route_attestation_reference": submission["route"]["attestation_reference"],
                 "quota_attestation_reference": submission["quota"]["attestation_reference"],
                 "tools_audit_status": "observed_zero_tool_use", "forbidden_tool_uses": [], "audit_reference": "fixture-audit",
                 "event_stream_status": "complete", "event_metadata": metadata,
                 "restricted_trace_sha256": "a" * 64, "restricted_trace_reference": "fixture-restricted-trace",
                 "usage": {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 5, "credits": "unknown"},
                 "input_isolation_status": "verified_packet_only_no_repo_global_context", "preventive_isolation_status": "not_claimed",
                 "timed_out": False, "completion_certain": True}
        native = {"native_usage": {"input_tokens": 10, "cached_input_tokens": 2, "cache_write_input_tokens": 1,
                                    "output_tokens": 5, "reasoning_output_tokens": 2},
                  "raw_turn_completed_line_sha256": "3" * 64, "full_stdout_trace_sha256": "a" * 64,
                  "raw_turn_completed_line_index": 3}
        native_raw = json.dumps(native, indent=2).encode() + b"\n"
        lines = [{"raw_line_index": n, "raw_line_bytes": 100, "raw_line_sha256": str(n) * 64} for n in range(4)]
        result = {"accepted": True, "rejection": None, "exit_code": 0, "stdout_EOF": True, "stderr_EOF": True,
                  "wall_seconds": 15.5, "peak_owned_RSS_KiB": 150000, "raw_stdout_sha256": "a" * 64,
                  "raw_stderr_sha256": M.raw_hash(b""), "raw_line_audit": lines, "unterminated_tail_bytes": 0,
                  "validated_prefix_metadata": [{**row, "type": metadata[n]["event_type"]} for n, row in enumerate(lines)],
                  "charged_attempts": 1, "no_automatic_wrapper_retry": True,
                  "native_usage_receipt_sha256": M.raw_hash(native_raw),
                  "stderr_classification": "empty required; no observed benign byte sequence approved"}
        return raw, audit, native_raw, self.a.p.json_bytes(result)

    def send_capture(self, arm, raw, audit, native, host):
        audit_raw = self.a.p.json_bytes(audit) if isinstance(audit, dict) else audit
        return self.a.capture(arm, raw, audit_raw, native, host)

    def capture(self, arm="A"):
        return self.send_capture(arm, *self.receipts(arm=arm))

    def fake_worker(self, arm, kind, output, module, tasks):
        # Deliberately synthetic predictions and summaries; never run a predictor.
        value = {"arm": kind, "summary": {"valid": True, "exact_demonstrations": 0, "exact_tasks": 0,
                    "mean_normalized_pixel_error": .5, "solved_demo_keys": [], "runtime_errors": 0},
                 "demonstrations": [{"prediction": [[0]]} for _ in range(17)], "fixture_only": True}
        self.a.p.write_once(output / (kind + ".json"), self.a.p.json_bytes(value))
        return value

    def evaluate_fixture(self, arm="A", worker=None):
        with patch.object(self.a.intake, "frozen_runtime", return_value=(SimpleNamespace(), {}, [])), \
             patch.object(self.a, "run_worker", side_effect=worker or self.fake_worker):
            return self.a.evaluate(arm)

    def test_supplement_binds_exact_preparation_sources_and_actual_V5_configs(self):
        value = json.loads(self.supplement.read_text())
        self.assertFalse(value["parent_execution_approval"])
        self.assertEqual(value["preparation_contract_sha256"], M.PREPARATION_SHA)
        self.assertEqual(value["actual_V5_route_bindings"]["PROSPECTIVE_V5_COMPARISON_A.json"]["sha256"], M.CONFIG_SHAS["A"])
        self.assertFalse(self.a.root.exists())
        self.a.verify_bindings()

    def test_relative_repo_supplement_payload_matches_absolute_repo(self):
        with patch.object(M.os, "getcwd", return_value=str(ROOT)):
            self.assertEqual(M.supplement_payload(Path(".")), M.supplement_payload(ROOT))

    def test_initialize_requires_approval_and_never_overwrites(self):
        invalid = self.approval(); invalid["parent_approved"] = False
        with self.assertRaises(ValueError):
            self.a.initialize(invalid)
        self.assertFalse(self.a.root.exists())
        self.initialize()
        with self.assertRaisesRegex(ValueError, "new execution directory"):
            self.initialize()

    def test_A_then_B_new_founders_full_finite_budget(self):
        self.begin()
        self.assertEqual(self.a.status()["author_charged_positions"], 1488)
        self.capture()
        self.evaluate_fixture()
        self.begin("B")
        self.capture("B")
        self.evaluate_fixture("B")
        status = self.a.status()
        self.assertEqual(status["stages"], {"A": "evaluated", "B": "evaluated"})
        self.assertEqual(status["consumed_attempts"], 2)
        self.assertEqual(status["author_charged_positions"], 2976)
        self.assertEqual(status["paired_random_charged_positions"], 2976)
        events, _, _ = self.a._read()
        for arm in ("A", "B"):
            reserve = self.a.event(events, "reserve", arm)
            self.assertIsNone(reserve["parent_tensor_sha256"])
            self.assertEqual(reserve["role"], "new_independent_founder")
        comparison = json.loads((self.a.root / "evaluation/B/COMPARISON.json").read_text())
        self.assertTrue(comparison["comparison_complete"])
        self.assertEqual(comparison["descriptive_B_minus_A_exact_demonstrations"], 0)
        self.assertFalse(comparison["positive_feasibility"])
        with self.assertRaises(ValueError):
            self.a.reserve("B", self.approval("B"))

    def test_no_B_before_A_fully_completed_including_random(self):
        self.initialize()
        with self.assertRaises(ValueError):
            self.a.reserve("B", self.approval("B"))
        self.a.reserve("A", self.approval())
        with self.assertRaises(ValueError):
            self.a.reserve("B", self.approval("B"))
        self.a.submit("A", "fixture-host", self.route(), self.quota())
        self.capture()
        with self.assertRaises(ValueError):
            self.a.reserve("B", self.approval("B"))

    def test_duplicate_reserve_and_submit_never_consume_again(self):
        self.begin()
        with self.assertRaises(ValueError):
            self.a.reserve("A", self.approval())
        with self.assertRaises(ValueError):
            self.a.submit("A", "second-host", self.route(), self.quota())
        self.assertEqual(self.a.status()["consumed_attempts"], 1)

    def test_invalid_literal_consumes_full_charge_then_STOP_no_random(self):
        self.begin()
        result = self.send_capture("A", *self.receipts(raw=b'{"invalid":"fixture"}'))
        self.assertEqual(result["payload"]["status"], "invalid")
        status = self.a.status()
        self.assertTrue(status["terminal_stop"])
        self.assertEqual(status["paired_random_charged_positions"], 1488)
        self.assertFalse((self.a.root / "evaluation").exists())
        with self.assertRaises(ValueError):
            self.a.random_response("A")
        with self.assertRaises(ValueError):
            self.a.reserve("B", self.approval("B"))

    def test_failed_audit_consumes_and_does_not_publish_private_fields(self):
        self.begin()
        raw, audit, native, host = self.receipts()
        audit["reasoning_text"] = "fixture-private-body-never-persist"
        self.send_capture("A", raw, audit, native, host)
        self.assertTrue(self.a.status()["terminal_stop"])
        for path in self.a.root.rglob("*"):
            if path.is_file() and "restricted_raw_evidence" not in path.parts:
                self.assertNotIn(b"fixture-private-body-never-persist", path.read_bytes())
        restricted = self.a.root / "restricted_raw_evidence/A/SUBMITTED_AUDIT.bin"
        self.assertIn(b"fixture-private-body-never-persist", restricted.read_bytes())

    def test_original_V5_result_and_native_usage_bytes_preserved(self):
        self.begin()
        raw, audit, native, host = self.receipts()
        result = self.send_capture("A", raw, audit, native, host)
        self.assertEqual(result["payload"]["status"], "valid")
        self.assertEqual((self.a.root / "capture/A/NATIVE_USAGE.json").read_bytes(), native)
        self.assertEqual((self.a.root / "capture/A/HOST_RESULT.json").read_bytes(), host)

    def test_host_timeout_memory_limit_line_mismatch_and_usage_fail(self):
        self.begin()
        raw, audit, native_raw, host_raw = self.receipts()
        original = json.loads(host_raw)
        for changes in ({"wall_seconds": 121}, {"peak_owned_RSS_KiB": 2097153}, {"accepted": False},
                        {"raw_stdout_sha256": "b" * 64}, {"stdout_EOF": False}, {"native_usage_receipt_sha256": "f" * 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                M.Adapter.host_check({**original, **changes}, audit, json.loads(native_raw), native_raw)
        bad = copy.deepcopy(original); bad["validated_prefix_metadata"][2]["raw_line_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            M.Adapter.host_check(bad, audit, json.loads(native_raw), native_raw)
        usage = json.loads(native_raw); usage["native_usage"].pop("reasoning_output_tokens")
        altered = self.a.p.json_bytes(usage)
        with self.assertRaises(ValueError):
            M.Adapter.host_check({**original, "native_usage_receipt_sha256": M.raw_hash(altered)}, audit, usage, altered)

    def test_stale_unmetered_85_percent_and_false_ordinary_usage_block(self):
        self.initialize()
        for change in ({"usage_percent": 85}, {"usage_percent": True}, {"ordinaryUsageAllowed": False},
                       {"authoritative": False}, {"observed_at": iso(-301)}, {"observed_at": iso(1)}, {"reset_at": iso(-1)}):
            approval = self.approval(); approval["quota"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.a.reserve("A", approval)
        self.assertEqual(self.a.status()["reservations"], 0)

    def test_refresh_quota_at_submission_and_period_reset_drift_blocks(self):
        self.initialize(); self.a.reserve("A", self.approval())
        with patch.object(M, "now", return_value=TIME + 301), self.assertRaises(ValueError):
            self.a.submit("A", "fixture", self.route(), self.quota())
        for change in ({"period_id": "another"}, {"reset_at": iso(7200)}, {"usage_percent": 85}):
            quota = self.quota(); quota.update(change)
            with self.assertRaises(ValueError):
                self.a.submit("A", "fixture", self.route(), quota)
        self.assertEqual(self.a.status()["consumed_attempts"], 0)

    def test_packet_isolation_actual_config_and_limits_fail_closed(self):
        self.initialize()
        changes = [{"controls_exposed": True}, {"cross_arm_feedback_exposed": True}, {"prior_weights_exposed": True},
                   {"model": "other"}, {"wall_seconds": 121}, {"memory_bytes": 3 * 1024 ** 3},
                   {"capture_config_sha256": M.CONFIG_SHAS["B"]}, {"scientific_input_sha256": ["f" * 64]},
                   {"new_independent_session": False}, {"request_retries": 1}]
        for change in changes:
            approval = self.approval(); approval["route"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.a.reserve("A", approval)
        wrong = self.approval(); wrong["packet_sha256"] = self.a.packet_sha("B")
        with self.assertRaises(ValueError):
            self.a.reserve("A", wrong)

    def test_reserved_crash_cannot_reset_or_create_new_reservation(self):
        self.initialize(); self.a.reserve("A", self.approval())
        restarted = M.Adapter(ROOT, self.supplement, self.sha, self.a.root)
        self.assertEqual(restarted.status()["unresolved_no_retry"], ["A"])
        for arm in ("A", "B"):
            with self.assertRaises(ValueError):
                restarted.reserve(arm, self.approval(arm))
        restarted.stop("A", "fixture crash before dispatch; no relaunch", "fixture-original-host-check")
        self.assertEqual(restarted.status()["consumed_attempts"], 0)
        with self.assertRaises(ValueError):
            restarted.submit("A", "fixture", self.route(), self.quota())

    def test_unknown_consumed_attempt_STOP_no_refund_or_retry(self):
        self.begin()
        self.a.stop("A", "fixture terminal timeout", "fixture-original-timeout-receipt")
        self.assertTrue(self.a.status()["terminal_stop"])
        self.assertEqual(self.a.status()["consumed_attempts"], 1)
        with self.assertRaises(ValueError):
            self.a.submit("A", "relaunch", self.route(), self.quota())
        with self.assertRaises(ValueError):
            self.capture()

    def test_partial_capture_crash_is_not_repairable(self):
        self.begin()
        receipts = self.receipts()
        with patch.object(self.a, "_append", side_effect=OSError("fixture disk failure")), self.assertRaises(OSError):
            self.send_capture("A", *receipts)
        self.assertEqual((self.a.root / "capture/A/RAW_RESPONSE.bin").read_bytes(), receipts[0])
        with self.assertRaisesRegex(ValueError, "partial/duplicate"):
            self.send_capture("A", *receipts)
        self.a.stop("A", "fixture partial capture", "fixture-disk-failure")
        self.assertEqual(self.a.status()["consumed_attempts"], 1)

    def test_partial_hashchain_corruption_never_resumes(self):
        self.begin()
        path = self.a.root / "LEDGER.jsonl"
        original = path.read_bytes()
        path.write_bytes(original + b'{"partial":')
        with self.assertRaisesRegex(ValueError, "partial"):
            self.a.status()
        path.write_bytes(original.replace(b'"consumed_author_positions":1488', b'"consumed_author_positions":1489'))
        with self.assertRaisesRegex(ValueError, "hash chain"):
            self.a.status()

    def test_evaluation_crash_terminal_blocks_B_and_no_random(self):
        self.begin(); self.capture()
        with self.assertRaises(RuntimeError):
            self.evaluate_fixture(worker=lambda *args: (_ for _ in ()).throw(RuntimeError("fixture-only evaluation crash")))
        self.assertTrue(self.a.status()["terminal_stop"])
        self.assertFalse((self.a.root / "evaluation/A/random.json").exists())
        with self.assertRaises(ValueError):
            self.a.reserve("B", self.approval("B"))

    def test_duplicate_capture_and_evaluation_rejected(self):
        self.begin(); self.capture()
        with self.assertRaises(ValueError):
            self.capture()
        self.evaluate_fixture()
        with self.assertRaises(ValueError):
            self.evaluate_fixture()

    def test_random_construction_rejects_preconsumption_and_invalid(self):
        self.initialize()
        with self.assertRaises(ValueError):
            self.a.random_response("A")
        self.a.reserve("A", self.approval())
        with self.assertRaises(ValueError):
            self.a.random_response("A")
        self.a.submit("A", "fixture", self.route(), self.quota())
        with self.assertRaises(ValueError):
            self.a.random_response("A")

    def test_literal_random_seed03_and04_only_consumed_evaluation(self):
        self.begin(); self.capture()
        events, _, _ = self.a._read()
        self.a._append(events, "evaluate_begin", "A", {"fixture_only": True})
        # Synthetic unit fixture only: never applied to the training panel.
        value = self.a.random_response("A")
        self.a.intake.validate_response(self.a.p.json_bytes(value))
        rng = M.random.Random(2026100803)
        first = self.a.intake.tensor_schema()[0]
        expected = [rng.choice(self.a.intake.VOCABULARY) for _ in range(first["positions"])]
        self.assertEqual(value["tensors"][first["name"]], [expected])
        self.assertEqual(self.a.p.SEEDS, {"A": 2026100803, "B": 2026100804})

    def test_saved_artifact_tampering_blocks_B(self):
        self.begin(); self.capture(); self.evaluate_fixture()
        path = self.a.root / "capture/A/RAW_RESPONSE.bin"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.a.reserve("B", self.approval("B"))

    def test_fsync_is_used_for_ledger_and_artifacts(self):
        with patch.object(M.os, "fsync", wraps=M.os.fsync) as fsync:
            self.begin()
            self.capture()
        self.assertGreaterEqual(fsync.call_count, 10)

    def test_real_frozen_preparation_unchanged(self):
        self.a.p.verify(ROOT, self.a.preparation, M.PREPARATION_SHA)
        self.assertFalse((self.a.preparation / "LEDGER.jsonl").exists())
        self.assertFalse((self.a.preparation / "attempts").exists())

    def test_STOP_and_dangling_STOP_guard_every_mutating_transition(self):
        self.initialize()
        stop = self.a.root / "STOP"
        stop.write_bytes(b"fixture stop")
        with self.assertRaisesRegex(ValueError, "STOP"):
            self.a.reserve("A", self.approval())
        stop.unlink()
        self.a.reserve("A", self.approval())
        stop.symlink_to(self.root / "absent-stop-target")
        self.assertTrue(self.a.status()["terminal_stop"])
        with self.assertRaisesRegex(ValueError, "STOP"):
            self.a.submit("A", "fixture", self.route(), self.quota())
        stop.unlink()
        self.a.submit("A", "fixture", self.route(), self.quota())
        receipts = self.receipts()
        stop.write_bytes(b"fixture stop")
        with self.assertRaisesRegex(ValueError, "STOP"):
            self.send_capture("A", *receipts)
        self.assertFalse((self.a.root / "capture").exists())
        stop.unlink()
        self.capture()
        stop.symlink_to(self.root / "absent-again")
        for action in (lambda: self.a.evaluate("A"), lambda: self.a.worker("A", "candidate"), lambda: self.a.random_response("A")):
            with self.assertRaisesRegex(ValueError, "STOP"):
                action()
        self.assertFalse((self.a.root / "evaluation").exists())
        self.a.stop("A", "fixture explicit stop", "fixture-receipt")
        self.assertTrue(self.a.status()["terminal_stop"])

    def test_dotdot_paths_cannot_enter_frozen_or_historical_directories(self):
        for target in (self.a.p.HISTORY, self.a.p.PHASE1, self.a.p.RUN_NAME):
            path = ROOT / "never-created" / ".." / target / "attempt-injection"
            with self.subTest(target=target), self.assertRaises(ValueError):
                M.Adapter(ROOT, self.supplement, self.sha, path)
            self.assertFalse((ROOT / target / "attempt-injection").exists())

    def test_repository_traversal_or_symlink_cannot_hide_protected_roots(self):
        repo = ROOT.parent / "never-created-parent" / ".." / ROOT.name
        for target in (self.a.p.HISTORY, self.a.p.PHASE1, self.a.p.RUN_NAME):
            with self.subTest(target=target), self.assertRaises(ValueError):
                M.Adapter(repo, self.supplement, self.sha, ROOT / target / "attempt-injection")
        linked = self.root / "linked-repository"
        linked.symlink_to(ROOT, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "nonsymlink repository"):
            M.Adapter(linked, self.supplement, self.sha, self.root / "unused")

    def test_invalid_receipt_original_bytes_are_durable_restricted_and_hash_bound(self):
        self.begin()
        raw, _, native, host = self.receipts()
        audit_raw = b'\xff{"private":"fixture-invalid-original"}'
        event = self.a.capture("A", raw, audit_raw, native, host)
        self.assertEqual(event["payload"]["status"], "invalid")
        self.assertTrue(event["payload"]["restricted_evidence_not_publishable"])
        for name, original in (("SUBMITTED_AUDIT.bin", audit_raw), ("NATIVE_USAGE.bin", native), ("HOST_RESULT.bin", host)):
            relative = "restricted_raw_evidence/A/" + name
            self.assertEqual((self.a.root / relative).read_bytes(), original)
            self.assertEqual(event["payload"]["restricted_artifacts"][relative], M.raw_hash(original))
        path = self.a.root / "restricted_raw_evidence/A/HOST_RESULT.bin"
        path.write_bytes(host + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.a.status()


if __name__ == "__main__":
    unittest.main()
