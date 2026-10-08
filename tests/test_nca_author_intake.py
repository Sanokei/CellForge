"""Strict phase2 intake regressions; no real author or phase2 panel scoring."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("nca_author_intake", ROOT / "scripts/nca_author_intake.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
BASE_SPEC = importlib.util.spec_from_file_location("nca_fixture_runtime", ROOT / "scripts/nca_feasibility_preflight.py")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)


def response():
    tensors, offset = {}, 0
    for item in M.tensor_schema():
        values = [M.VOCABULARY[(offset + i) % 7] for i in range(item["positions"])]
        tensors[item["name"]] = [values] if len(item["shape"]) == 2 else values
        offset += item["positions"]
    return {"schema": "nca-author-tensors/1", "tensors": tensors}


def task():
    return {"task_id": "opaque-evaluator-id", "train": [
        {"input": [[1, 2]], "output": [[1, 2]]},
        {"input": [[2]], "output": [[1]]}]}


def arm(name, exact, solved=(), valid=True):
    return {"arm": name, "summary": {"valid": valid, "exact_demonstrations": exact if valid else None,
        "solved_demo_keys": list(solved)}, "demonstrations": []}


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.expected = "fixture-contract"
        self.contract = {"phase1_path": "not-used"}
        # Persistence state tests deliberately use a separate mock freeze.
        self.guard = patch.object(M, "verify", return_value=(self.contract, {}, {}))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        M.write_json(self.run / "FOUNDER_PACKET.json", {"fixture": "no live author"})

    def approval(self, number=1):
        return {"schema": "nca-parent-approval/1", "parent_approved": True,
            "parent_identity": "mock-parent", "contract_sha256": self.expected,
            "packet_sha256": M.file_hash(M.packet_path(self.run, number)), "approval_reference": "mock:approval",
            "route": {"identity": "mock-input-only", "attestation_reference": "mock:route", "audit_mode": "complete_event_stream_conservative_allowlist",
                      "input_isolation_status": "verified_packet_only_no_repo_global_context", "preventive_isolation_status": "not_claimed"},
            "quota": {"attestation_reference": "mock:quota", "approved_max_attempts": 2, "approved_positions_each_arm": 2976,
                      "credit_accounting": "mock-only; no credits used"}}

    def begin(self):
        reservation = M.reserve_attempt(self.run, self.expected, 1, self.approval())
        submission = M.mark_submission_intent(self.run, self.expected, 1, "mock:submission")
        return reservation, submission

    def audit(self, raw, status="completed", number=1):
        directory = M.attempt_dir(self.run, number)
        reservation = M.read_json(directory / "RESERVATION.json")
        submission = M.read_json(directory / "SUBMISSION.json")
        return {"schema": "nca-host-response-audit/1", "status": status, "host_capture": True,
            "parent_identity": reservation["parent_identity"], "packet_sha256": reservation["packet_sha256"],
            "raw_sha256": M.hashlib.sha256(raw).hexdigest(), "submission_sha256": M.digest(submission),
            "route_attestation_reference": reservation["route_attestation_reference"],
            "quota_attestation_reference": reservation["quota_attestation_reference"],
            "tools_audit_status": "observed_zero_tool_use", "forbidden_tool_uses": [], "audit_reference": "mock:audit",
            "event_stream_status": "complete", "timed_out": False, "completion_certain": True,
            "input_isolation_status": "verified_packet_only_no_repo_global_context", "preventive_isolation_status": "not_claimed",
            "restricted_trace_sha256": "a" * 64, "restricted_trace_reference": "mock:restricted-parent-trace",
            "usage": {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 50, "credits": "unknown"},
            "event_metadata": [
                {"sequence": 0, "event_type": "thread.started", "item_type": None, "classification": "lifecycle", "event_sha256": "0" * 64, "final_raw_sha256": None},
                {"sequence": 1, "event_type": "turn.started", "item_type": None, "classification": "lifecycle", "event_sha256": "1" * 64, "final_raw_sha256": None},
                {"sequence": 2, "event_type": "item.completed", "item_type": "agent_message", "classification": "final_message", "event_sha256": "2" * 64, "final_raw_sha256": M.hashlib.sha256(raw).hexdigest()},
                {"sequence": 3, "event_type": "turn.completed", "item_type": None, "classification": "lifecycle", "event_sha256": "3" * 64, "final_raw_sha256": None}]}

    def complete(self, raw=None):
        self.begin()
        raw = M.canonical(response()) if raw is None else raw
        return M.capture_response(self.run, self.expected, 1, raw, self.audit(raw))

    def test_schema_exact_96_arrays_1488_positions_sorted(self):
        schema = M.tensor_schema()
        self.assertEqual(len(schema), 96)
        self.assertEqual(sum(item["positions"] for item in schema), 1488)
        self.assertEqual([item["name"] for item in schema], sorted(item["name"] for item in schema))
        self.assertEqual(M.validate_response(M.canonical(response())), response())

    def test_reject_invalid_json_duplicate_nonfinite_utf8_bool_and_source(self):
        invalid = [b'{"schema":"a","schema":"b"}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff',
                   b'```json\n{}\n```', b'null', b'[]', b'{"source":"print(1)"}']
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                M.validate_response(raw)
        for literal in (True, False, None, "0", 2, float("inf")):
            value = response()
            value["tensors"]["nca.h0.c0.b"] = [literal]
            with self.subTest(literal=literal), self.assertRaises((ValueError, TypeError)):
                M.validate_response(json.dumps(value).encode())

    def test_reject_wrong_shape_missing_extra_and_executable_fields(self):
        for mutate in (lambda v: v.update(program="x"), lambda v: v["tensors"].pop("nca.o0.W"),
                       lambda v: v["tensors"].update(extra=[0]), lambda v: v["tensors"].update({"nca.o0.W": [0] * 12}),
                       lambda v: v["tensors"].update({"nca.o0.W": [[0] * 11]})):
            value = response()
            mutate(value)
            with self.assertRaises(ValueError):
                M.validate_response(M.canonical(value))

    def test_signed_zero_literal_preserved_without_normalization(self):
        value = response()
        value["tensors"]["nca.h0.c0.b"] = [-0.0]
        parsed = M.validate_response(M.canonical(value))
        self.assertTrue(np.signbit(parsed["tensors"]["nca.h0.c0.b"][0]))

    def test_max_response_size_rejected(self):
        with self.assertRaisesRegex(ValueError, "byte limit"):
            M.validate_response(b" " * (M.MAX_RESPONSE_BYTES + 1))

    def test_packet_no_id_provenance_query_control_or_source_leaks(self):
        packet, mapping = M.build_founder_packet([task()])
        text = M.canonical(packet).decode()
        self.assertNotIn(task()["task_id"], text)
        self.assertEqual(mapping[0]["task_id"], task()["task_id"])
        self.assertEqual([case["case"] for case in packet["training_cases"]], ["case_01", "case_02"])
        for key in ("task_id", "query", "test", "provenance", "reference_tensors", "random_scores", "selected_baseline"):
            self.assertNotIn('"' + key + '"', text)
        self.assertEqual(set(packet), {"schema", "architecture", "tensor_schema", "training_cases", "response_contract"})
        self.assertIn("y-dy,x-dx", text)
        self.assertIn("(p[r,0]+p[r,1])+p[r,2]", text)
        self.assertIn("exactly 8", text)
        self.assertIn("No demonstration conditioning", text)

    def test_packet_rejects_query_fields(self):
        value = task()
        value["test"] = []
        with self.assertRaises(ValueError):
            M.build_founder_packet([value])
        value = task()
        value["train"][0]["query"] = []
        with self.assertRaises(ValueError):
            M.build_founder_packet([value])

    def test_unapproved_route_and_absence_of_declared_tools_fail_closed(self):
        for field, replacement in (("parent_approved", False), ("parent_identity", ""), ("packet_sha256", "wrong")):
            value = self.approval()
            value[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                M.reserve_attempt(self.run, self.expected, 1, value)
        value = self.approval()
        value["route"]["audit_mode"] = "no_declared_tools"
        with self.assertRaisesRegex(ValueError, "complete-event"):
            M.reserve_attempt(self.run, self.expected, 1, value)
        self.assertFalse((self.run / "attempts").exists())

    def test_reservation_then_durable_submission_charges_once(self):
        M.reserve_attempt(self.run, self.expected, 1, self.approval())
        self.assertEqual(M.accounting(self.run)["consumed_attempts"], 0)
        M.mark_submission_intent(self.run, self.expected, 1, "mock:submission")
        self.assertEqual(M.accounting(self.run), {"reservations": 1, "consumed_attempts": 1,
            "author_charged_positions": 1488, "paired_random_charged_positions": 1488})
        with self.assertRaises(FileExistsError):
            M.mark_submission_intent(self.run, self.expected, 1, "mock:retry")
        self.assertEqual(M.accounting(self.run)["consumed_attempts"], 1)

    def test_invalid_raw_preserved_and_charged_once_without_retry(self):
        raw = b"\xffnot json"
        receipt = self.complete(raw)
        self.assertEqual(receipt["status"], "invalid")
        self.assertEqual((M.attempt_dir(self.run, 1) / "RAW_RESPONSE.bin").read_bytes(), raw)
        self.assertEqual(M.accounting(self.run)["author_charged_positions"], 1488)
        with self.assertRaises(FileExistsError):
            M.capture_response(self.run, self.expected, 1, M.canonical(response()), self.audit(raw))
        with self.assertRaisesRegex(ValueError, "invalid/refused"):
            M.reserve_attempt(self.run, self.expected, 2, self.approval())

    def test_refusal_consumes_and_stops_without_repair(self):
        self.begin()
        raw = b"I cannot provide these arrays."
        receipt = M.capture_response(self.run, self.expected, 1, raw, self.audit(raw, "refused"))
        self.assertEqual(receipt["status"], "refused")
        self.assertEqual(M.accounting(self.run)["paired_random_charged_positions"], 1488)
        with self.assertRaises(ValueError):
            M.require_open_tranche(self.run)

    def test_unknown_submission_stops_all_future_work(self):
        self.begin()
        with self.assertRaisesRegex(ValueError, "unknown"):
            M.reserve_attempt(self.run, self.expected, 2, self.approval())
        M.mark_uncertain(self.run, self.expected, 1, "mock transport uncertainty")
        raw = M.canonical(response())
        with self.assertRaisesRegex(ValueError, "halted"):
            M.capture_response(self.run, self.expected, 1, raw, self.audit(raw))
        self.assertEqual(M.accounting(self.run)["consumed_attempts"], 1)

    def test_forbidden_tool_use_rejected_before_candidate_evaluation(self):
        self.begin()
        raw = M.canonical(response())
        audit = self.audit(raw)
        audit["forbidden_tool_uses"] = ["mock shell call"]
        receipt = M.capture_response(self.run, self.expected, 1, raw, audit)
        self.assertEqual(receipt["status"], "invalid")
        self.assertIn("forbidden tool", receipt["error"]["message"])
        self.assertEqual((M.attempt_dir(self.run, 1) / "RAW_RESPONSE.bin").read_bytes(), raw)

    def test_unverified_tools_even_empty_list_rejected(self):
        self.begin()
        raw = M.canonical(response())
        audit = self.audit(raw)
        audit["tools_audit_status"] = "no_tools_declared"
        self.assertEqual(M.capture_response(self.run, self.expected, 1, raw, audit)["status"], "invalid")
        self.assertTrue((self.run / "HALT.json").exists())

    def test_unknown_event_gap_truncation_timeout_and_uncertainty_rejected(self):
        reservation, submission = self.begin()
        raw = M.canonical(response())
        changes = [lambda audit: audit.update(event_stream_status="truncated"),
                   lambda audit: audit.update(timed_out=True), lambda audit: audit.update(completion_certain=False),
                   lambda audit: audit["event_metadata"][2].update(sequence=3),
                   lambda audit: audit["event_metadata"][2].update(event_type="unknown.exec"),
                   lambda audit: audit["event_metadata"][2].update(item_type="command_execution"),
                   lambda audit: audit.update(input_isolation_status="repo_context_present")]
        for mutate in changes:
            audit = self.audit(raw)
            mutate(audit)
            with self.assertRaises(ValueError):
                M.validate_host_audit(audit, reservation, submission, M.hashlib.sha256(raw).hexdigest())

    def test_reasoning_event_is_hash_metadata_only_and_not_exposed_text(self):
        reservation, submission = self.begin()
        raw = M.canonical(response())
        audit = self.audit(raw)
        audit["event_metadata"].insert(2, {"sequence": 2, "event_type": "item.completed", "item_type": "reasoning",
            "classification": "reasoning_summary", "event_sha256": "b" * 64, "final_raw_sha256": None})
        for index, event in enumerate(audit["event_metadata"]):
            event["sequence"] = index
        M.validate_host_audit(audit, reservation, submission, M.hashlib.sha256(raw).hexdigest())
        audit["event_metadata"][2]["text"] = "private text must never publish"
        with self.assertRaises(ValueError):
            M.validate_host_audit(audit, reservation, submission, M.hashlib.sha256(raw).hexdigest())
        receipt = M.capture_response(self.run, self.expected, 1, raw, audit)
        self.assertEqual(receipt["status"], "invalid")
        self.assertNotIn("private text must never publish", (M.attempt_dir(self.run, 1) / "HOST_AUDIT.json").read_text())

    def test_numeric_duplicates_ignore_integer_float_and_signed_zero_spellings(self):
        left = response()
        right = copy.deepcopy(left)
        for name, array in right["tensors"].items():
            def floats(value):
                return [floats(child) for child in value] if isinstance(value, list) else float(value)
            right["tensors"][name] = floats(array)
        left["tensors"]["nca.h0.c0.b"] = [0]
        right["tensors"]["nca.h0.c0.b"] = [-0.0]
        self.assertNotEqual(M.digest(left["tensors"]), M.digest(right["tensors"]))
        self.assertEqual(M.numeric_fingerprint(left["tensors"]), M.numeric_fingerprint(right["tensors"]))

    def test_host_audit_binds_packet_parent_raw_route_quota_and_submission(self):
        reservation, submission = self.begin()
        raw = M.canonical(response())
        for key in ("parent_identity", "packet_sha256", "raw_sha256", "submission_sha256", "route_attestation_reference", "quota_attestation_reference"):
            audit = self.audit(raw)
            audit[key] = "tamper"
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.validate_host_audit(audit, reservation, submission, M.hashlib.sha256(raw).hexdigest())

    def test_response_and_packet_tamper_rejected(self):
        receipt = self.complete()
        self.assertEqual(M.verified_intake(self.run, self.expected, 1), receipt)
        path = M.attempt_dir(self.run, 1) / "RAW_RESPONSE.bin"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "binding"):
            M.verified_intake(self.run, self.expected, 1)
        (self.run / "FOUNDER_PACKET.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "binding"):
            M.verify_reservation(self.run, self.expected, 1)

    def test_no_overwrite_and_two_attempt_cap(self):
        path = self.run / "immutable.bin"
        M.write_once(path, b"first")
        with self.assertRaises(FileExistsError):
            M.write_once(path, b"second")
        self.assertEqual(path.read_bytes(), b"first")
        with self.assertRaisesRegex(ValueError, "fresh"):
            M.freeze(self.run, self.run)
        for number in (0, 3, True, 1.0):
            with self.assertRaises(ValueError):
                M.attempt_dir(self.run, number)

    def test_second_random_unconstructed_before_second_consumption(self):
        self.complete()
        with self.assertRaises((ValueError, FileNotFoundError)):
            M.paired_second_literals(self.run, self.expected)
        self.assertFalse(any("paired_random_2" in path.name for path in self.run.rglob("*")))

    def test_full_mock_conditional_mutation_and_second_consumption_path(self):
        founder = self.complete()
        directory = M.attempt_dir(self.run, 1)
        packet, mapping = M.build_founder_packet([task()])
        M.write_json(self.run / "EVALUATOR_ONLY.json", {"case_map": mapping})
        evaluation = {"intake_sha256": M.file_hash(directory / "INTAKE.json"), "artifact_sha256": {}, "controls_valid": True,
            "candidate": {"summary": {"valid": True, "exact_demonstrations": 1},
                "demonstrations": [{"task_id": task()["task_id"], "demo_index": 0, "exact": True},
                    {"task_id": task()["task_id"], "demo_index": 1, "exact": False, "prediction": [[2]], "pixel_errors": 1, "normalized_pixel_error": 1.0}]}}
        M.write_json(directory / "EVALUATION.json", evaluation)
        M.write_once(directory / "EVALUATION.sha256", (M.file_hash(directory / "EVALUATION.json") + "\n").encode())
        packet_sha = M.prepare_mutation_packet(self.run, self.expected)
        packet = M.read_json(self.run / "MUTATION_PACKET.json")
        self.assertEqual(packet["parent_tensor_sha256"], founder["tensor_sha256"])
        self.assertEqual(packet["deterministic_training_failures"][0]["case"], "case_02")
        self.assertNotIn(task()["task_id"], M.canonical(packet).decode())
        self.assertEqual(packet_sha, M.file_hash(self.run / "MUTATION_PACKET.json"))
        M.reserve_attempt(self.run, self.expected, 2, self.approval(2))
        M.mark_submission_intent(self.run, self.expected, 2, "mock:second")
        raw = M.canonical(response())
        receipt = M.capture_response(self.run, self.expected, 2, raw, self.audit(raw, number=2))
        self.assertEqual(receipt["status"], "valid")
        self.assertEqual(receipt["parent_tensor_sha256"], founder["tensor_sha256"])
        self.assertEqual(M.accounting(self.run)["author_charged_positions"], 2976)
        self.assertEqual(M.validate_response(M.canonical(M.paired_second_literals(self.run, self.expected)))["schema"], "nca-author-tensors/1")
        with self.assertRaises(ValueError):
            M.reserve_attempt(self.run, self.expected, 3, self.approval())

    def test_comparison_requires_strict_gain_every_random_and_anchors(self):
        ref = arm("identity", 2, [("x", 0)])
        randoms = [arm("random1", 3), arm("random2", 4)]
        self.assertFalse(M.compare(arm("candidate", 4, [("x", 0)]), [ref], randoms, ref)["positive_feasibility"])
        self.assertFalse(M.compare(arm("candidate", 5, []), [ref], randoms, ref)["positive_feasibility"])
        self.assertTrue(M.compare(arm("candidate", 5, [("x", 0)]), [ref], randoms, ref)["positive_feasibility"])

    def test_invalid_control_scores_are_null_not_comparable(self):
        ref = arm("reference", 0)
        result = M.compare(arm("candidate", 17), [ref], [arm("random", 0, valid=False)], ref)
        self.assertFalse(result["comparable"])
        self.assertIsNone(result["positive_feasibility"])

    def test_mutation_must_improve_founder_and_preserve_founder_anchors(self):
        ref = arm("reference", 0)
        founder = arm("founder", 2, [("a", 0), ("b", 0)])
        self.assertFalse(M.compare(arm("candidate", 2, founder["summary"]["solved_demo_keys"]), [ref], [ref], ref, founder)["positive_mutation"])
        self.assertFalse(M.compare(arm("candidate", 3, [("a", 0)]), [ref], [ref], ref, founder)["positive_mutation"])
        self.assertTrue(M.compare(arm("candidate", 3, founder["summary"]["solved_demo_keys"]), [ref], [ref], ref, founder)["positive_mutation"])


class SourceAndInjectionTests(unittest.TestCase):
    def test_all_author_literals_injected_unchanged_into_saved_t8_descriptor(self):
        graph = BASE.control_graph("identity")
        descriptor = BASE.graph_structure(graph)
        manifest = {"graph": {"structure_sha256": M.digest(descriptor)}}
        value = response()
        value["tensors"]["nca.h0.c0.b"] = [-0.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            M.write_json(root / "GRAPH_STRUCTURE.json", descriptor)
            with patch.object(BASE, "control_graph", side_effect=AssertionError("no control substitution")):
                injected, hashes = M.inject_literals(BASE, root, manifest, value)
            self.assertEqual(len(hashes), 96)
            self.assertEqual(injected.meta["nca_recipe"]["iterations"], 8)
            self.assertEqual(BASE.graph_structure(injected), descriptor)
            for name, literals in value["tensors"].items():
                self.assertEqual(injected.params[name].tobytes(), np.asarray(literals, dtype=np.float64).tobytes())
            self.assertNotEqual(injected.hash(), graph.hash())

    def test_anti_substitution_detects_runtime_graph_changing_literals(self):
        graph = BASE.control_graph("identity")
        descriptor = BASE.graph_structure(graph)
        imports = BASE.runtime_imports()
        def dishonest_graph(defs, params, entries, meta):
            params["nca.h0.c0.b"] = np.array([1.0])
            return imports[2](defs, params, entries, meta)
        altered_imports = (*imports[:2], dishonest_graph, *imports[3:])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            M.write_json(root / "GRAPH_STRUCTURE.json", descriptor)
            with patch.object(BASE, "runtime_imports", return_value=altered_imports), self.assertRaisesRegex(ValueError, "substitution"):
                M.inject_literals(BASE, root, {"graph": {"structure_sha256": M.digest(descriptor)}}, response())

    def test_small_fixture_predicts_from_author_literals_not_identity(self):
        graph = BASE.control_graph("identity")
        descriptor = BASE.graph_structure(graph)
        value = {"schema": "nca-author-tensors/1", "tensors": {name: np.zeros_like(array).tolist() for name, array in graph.params.items()}}
        # Every update output is one in channel 7; all other positions stay literal zero.
        value["tensors"]["nca.o7.b"] = [1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            M.write_json(root / "GRAPH_STRUCTURE.json", descriptor)
            injected, _ = M.inject_literals(BASE, root, {"graph": {"structure_sha256": M.digest(descriptor)}}, value)
            *_, Predictor, _ = BASE.runtime_imports()
            result = BASE.evaluate_demo(Predictor(injected, backend="reference", cache_demo_state=False), task(), 0)
            self.assertEqual(result["prediction"], [[7, 7]])
            self.assertEqual(result["pixel_errors"], 2)

    def test_phase1_static_hash_validation_and_source_tamper(self):
        phase1 = ROOT / "runs/weights-nca-feasibility-20261008"
        manifest, result = M.verify_phase1(phase1)
        self.assertEqual(manifest["panel"]["all_tasks"], 11)
        self.assertEqual(manifest["panel"]["all_demonstrations"], 32)
        self.assertEqual(result["arms"]["paired_random_1"]["exact_demonstrations"], 0)
        actual = M.file_hash
        def tampered(path):
            return "tamper" if str(path).endswith("source/src/cellforge/runtime/ops.py") else actual(path)
        with patch.object(M, "file_hash", side_effect=tampered), self.assertRaisesRegex(ValueError, "source binding"):
            M.verify_phase1(phase1)

    def test_real_freeze_data_only_and_packet_source_contract_tamper(self):
        phase1 = ROOT / "runs/weights-nca-feasibility-20261008"
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "mock-data-only-freeze"
            expected = M.freeze(phase1, run)
            contract, _, _ = M.verify(run, expected)
            packet = M.read_json(run / "FOUNDER_PACKET.json")
            self.assertEqual(len(packet["training_cases"]), 17)
            self.assertFalse(contract["parent_author_approval"])
            self.assertEqual(M.accounting(run)["consumed_attempts"], 0)
            self.assertFalse((run / "attempts").exists())
            self.assertFalse((run / "MUTATION_PACKET.json").exists())
            text = (run / "FOUNDER_PACKET.json").read_text()
            for original_id in M.read_json(run / "EVALUATOR_ONLY.json")["coverage"]["all_task_ids"]:
                self.assertNotIn(original_id, text)
            source = run / "source" / M.SOURCE_NAME
            source.write_bytes(source.read_bytes() + b"\n# tampered")
            with self.assertRaisesRegex(ValueError, "artifact binding"):
                M.verify(run, expected)
            (run / "CONTRACT.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "contract hash"):
                M.verify(run, expected)

    def test_graph_descriptor_tamper_rejected(self):
        graph = BASE.control_graph("identity")
        descriptor = BASE.graph_structure(graph)
        manifest = {"graph": {"structure_sha256": M.digest(descriptor)}}
        descriptor["meta"]["nca_recipe"]["iterations"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            M.write_json(root / "GRAPH_STRUCTURE.json", descriptor)
            with self.assertRaisesRegex(ValueError, "descriptor"):
                M.inject_literals(BASE, root, manifest, response())

    def test_phase2_stop_including_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "STOP").symlink_to(run / "missing")
            with self.assertRaisesRegex(ValueError, "STOP"):
                M.verify(run, "irrelevant")


if __name__ == "__main__":
    unittest.main()
