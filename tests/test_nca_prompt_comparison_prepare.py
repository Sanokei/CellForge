"""Isolated preparation tests. No author, random tensor or scoring execution."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("nca_prompt_comparison_prepare", ROOT / "scripts/nca_prompt_comparison_prepare.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PromptComparisonPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads = M.build_payloads(ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.temp_root = Path(self.temp.name)
        self.run = self.temp_root / "preparation"

    def value(self, name):
        return M.strict_json(self.payloads[name])

    def freeze(self):
        return M.freeze(ROOT, self.run)

    def fixture_repo(self):
        repo = self.temp_root / "repo"
        names = {item["path"] for item in self.value("SOURCE_BINDINGS.json")["records"]}
        names.update([M.SOURCE_NAME, M.TEST_NAME])
        for name in names:
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        return repo

    def test_A_exact_original_bytes_sha_and_size(self):
        raw = self.payloads["author/A_PACKET.json"]
        self.assertEqual(raw, (ROOT / M.HISTORY / "FOUNDER_PACKET.json").read_bytes())
        self.assertEqual(len(raw), 16752)
        self.assertEqual(M.sha256(raw), M.FOUNDER_SHA)

    def test_schema_exact_original_bytes_sha_and_size(self):
        raw = self.payloads["author/RESPONSE_SCHEMA.json"]
        self.assertEqual(raw, (ROOT / M.HISTORY / "RESPONSE_SCHEMA.json").read_bytes())
        self.assertEqual(len(raw), 15494)
        self.assertEqual(M.sha256(raw), M.SCHEMA_SHA)

    def test_B_exact_two_path_diff_and_reversible(self):
        a = self.value("author/A_PACKET.json")
        b = self.value("author/B_PACKET.json")
        diff = self.value("PROMPT_DIFF.json")
        self.assertEqual(diff["permitted_paths"], ["/architecture/sharing", "/objective"])
        self.assertEqual(len(diff["patch"]), 2)
        self.assertEqual(diff["patch"][0], {"op": "replace", "path": "/architecture/sharing",
            "old_value": a["architecture"]["sharing"], "value": b["architecture"]["sharing"]})
        self.assertEqual(diff["patch"][1], {"op": "add", "path": "/objective", "value": M.OBJECTIVE})
        self.assertEqual(b["architecture"]["sharing"], a["architecture"]["sharing"].replace(M.OLD_PROHIBITION, M.NEW_PROHIBITION))
        b["architecture"]["sharing"] = a["architecture"]["sharing"]
        del b["objective"]
        self.assertEqual(M.json_bytes(b), self.payloads["author/A_PACKET.json"])
        for arm in ("A", "B"):
            raw = self.payloads["author/" + arm + "_PACKET.json"]
            self.assertEqual(diff[arm], {"bytes": len(raw), "sha256": M.sha256(raw)})

    def test_unchanged_architecture_examples_tensor_contract_subtree_bytes(self):
        a = self.value("author/A_PACKET.json")
        b = self.value("author/B_PACKET.json")
        for key in ("training_cases", "tensor_schema", "response_contract", "schema"):
            self.assertEqual(M.canonical(a[key]), M.canonical(b[key]))
        for key in a["architecture"]:
            if key != "sharing":
                self.assertEqual(M.canonical(a["architecture"][key]), M.canonical(b["architecture"][key]))
        del a["architecture"]["sharing"]
        del b["architecture"]["sharing"]
        del b["objective"]
        self.assertEqual(M.canonical(a), M.canonical(b))
        self.assertEqual(M.sha256(M.canonical(a)), self.value("PROMPT_DIFF.json")["unchanged_canonical_projection_sha256"])

    def test_manual_reasoning_allowed_without_runtime_tools_or_fitting(self):
        b = self.value("author/B_PACKET.json")
        self.assertIn("Manual numerical reasoning to choose literal weights is allowed", b["architecture"]["sharing"])
        for text in ("gradient or runtime fitting", "do not use tools, gradients, or code", "per-case weights", "executable structure changes"):
            self.assertIn(text, b["architecture"]["sharing"])
        for text in ("all 17 training demonstrations", "exact output-grid matches", "Break ties", "unweighted mean"):
            self.assertIn(text, b["objective"])

    def test_training_only_author_allowlist_no_weights_controls_ids_or_feedback(self):
        evaluator = self.value("EVALUATOR_ONLY.json")
        for arm in ("A", "B"):
            value = self.value("author/" + arm + "_PACKET.json")
            expected = {"schema", "architecture", "tensor_schema", "training_cases", "response_contract"}
            if arm == "B":
                expected.add("objective")
            self.assertEqual(set(value), expected)
            self.assertEqual(len(value["training_cases"]), 17)
            for index, case in enumerate(value["training_cases"], 1):
                self.assertEqual(set(case), {"case", "input", "output"})
                self.assertEqual(case["case"], "case_%02d" % index)
            raw = M.canonical(value).decode()
            for task_id in evaluator["coverage"]["all_task_ids"]:
                self.assertNotIn(task_id, raw)
            for forbidden in ("task_id", "query", "test", "tensors", "parent_tensors", "prior_weights", "feedback",
                              "random_scores", "selected_baseline", "reference_tensors", "control_results"):
                # The original response contract has a 'tensors' description,
                # but never an actual tensor object or values.
                if forbidden != "tensors":
                    self.assertNotIn('"' + forbidden + '":', raw)
            self.assertIsInstance(value["response_contract"]["tensors"], str)
            self.assertNotIn("tensors", value)

    def test_full_inventory_and_exact_restricted_scope_preserved(self):
        e = self.value("EVALUATOR_ONLY.json")
        original = json.loads((ROOT / M.HISTORY / "EVALUATOR_ONLY.json").read_text())
        self.assertEqual(e["coverage"], original["coverage"])
        self.assertEqual(e["case_map"], original["case_map"])
        coverage = e["coverage"]
        self.assertEqual([coverage[k] for k in ("all_tasks", "all_demonstrations", "compatible_tasks",
                                               "compatible_demonstrations", "excluded_tasks", "excluded_demonstrations")],
                         [11, 32, 6, 17, 5, 15])
        excluded = [entry for entry in coverage["coverage"] if entry["shape_mismatch_pairs"]]
        self.assertEqual(len(excluded), 5)
        self.assertEqual(sum(entry["demonstrations"] for entry in excluded), 15)
        self.assertIn("e7639916 remains excluded", coverage["cancelled_task"])
        self.assertEqual(len({case["task_id"] for case in e["case_map"]}), 6)

    def test_two_fresh_founders_order_no_mutation_or_historical_substitution(self):
        p = self.value("PROTOCOL.json")
        self.assertEqual(p["author_order"], ["A", "B"])
        self.assertEqual(p["mutations_allowed"], 0)
        self.assertFalse(p["cross_arm_feedback_allowed"])
        self.assertFalse(p["historical_founder_is_prospective_A"])
        for arm in p["author_order"]:
            self.assertEqual(p["authors"][arm]["kind"], "new_independent_founder")
            self.assertIsNone(p["authors"][arm]["parent_weights"])
        self.assertIn("even if A is perfect", p["continuation_gate_B"])
        self.assertIn("Do not invoke B", p["terminal_stop"]["A_timeout"])
        self.assertIn("inconclusive", p["terminal_stop"]["A_timeout"])
        self.assertFalse(p["accounting"]["retry_refund_replacement_allowed"])
        self.assertIn("timeout", p["terminal_stop"]["conditions"])
        self.assertIn("audit_failure", p["terminal_stop"]["conditions"])

    def test_same_architecture_model_route_and_limits(self):
        p = self.value("PROTOCOL.json")
        self.assertEqual(p["fixed_architecture"], {"iterations": 8, "arrays": 96, "positions": 1488,
                                                 "global_tensor_set": True, "vocabulary": [-1, -.5, -.25, 0, .25, .5, 1]})
        route = p["author_route"]
        for key, expected in (("model", "gpt-6.1-sol"), ("reasoning_effort", "low"), ("wall_seconds", 120),
                              ("memory_bytes", 2147483648), ("concurrency", 1), ("request_retries", 0), ("stream_retries", 0)):
            self.assertEqual(route[key], expected)
        self.assertTrue(route["same_limits_both_arms"])
        self.assertFalse(route["preventive_tool_isolation_claimed"])
        self.assertEqual(route["qualified_provenance"], "observed_zero_tool_use_only")

    def test_equal_full_position_budgets_and_distinct_future_random_seeds(self):
        p = self.value("PROTOCOL.json")
        a = p["accounting"]
        self.assertEqual(a["maximum_author_invocations_total"], 2)
        self.assertEqual(a["maximum_author_invocations_each_prompt_arm"], 1)
        self.assertEqual(a["maximum_author_positions_total"], 2976)
        self.assertEqual(a["maximum_matched_random_positions_total"], 2976)
        self.assertEqual(a["positions_each_consumed_author_attempt"], 1488)
        self.assertEqual(a["positions_each_matched_random_attempt"], 1488)
        self.assertTrue(a["charge_invalid_refused_neutral_duplicate_timeout_uncertain"])
        random = p["random_controls"]
        self.assertEqual(random["seeds"], {"A": 2026100803, "B": 2026100804})
        self.assertEqual(random["vocabulary"], p["fixed_architecture"]["vocabulary"])
        self.assertIn("uniform", random["distribution"])
        self.assertIn("actually consumed", random["generation_gate"])
        self.assertIn("forbids further construction/evaluation", random["terminal_gate"])
        self.assertFalse(random["constructed_at_freeze"])
        self.assertFalse(random["historical_random_reuse_allowed"])

    def test_fixed_control_outputs_are_only_original_bound_bytes(self):
        e = self.value("EVALUATOR_ONLY.json")
        self.assertEqual(set(e["fixed_references"]), {"identity", "neighbor_majority", "zero"})
        for name, reference in e["fixed_references"].items():
            self.assertEqual(len(reference["source_records"]), 3)
            for record in reference["source_records"]:
                raw = (ROOT / record["path"]).read_bytes()
                self.assertEqual(M.sha256(raw), record["sha256"])
                self.assertEqual(len(raw), record["bytes"])
                self.assertTrue(record["path"].startswith(M.PHASE1 + "/"))
        self.assertFalse(e["author_exposure_allowed"])
        self.assertIn("no redraw or reevaluation", self.value("PROTOCOL.json")["fixed_controls"]["policy"])

    def test_primary_secondary_and_descriptive_only(self):
        analysis = self.value("PROTOCOL.json")["analysis"]
        self.assertIn("B exact training demonstrations minus A", analysis["primary"])
        self.assertIn("denominator 17", analysis["primary"])
        joined = " ".join(analysis["secondary"])
        for text in ("Exact tasks", "pixel error", "Numerical identity equivalence", "prediction-vector identity equivalence"):
            self.assertIn(text, joined)
        for text in ("One prospective pair", "descriptive only", "no causal attribution", "fixed A-then-B order"):
            self.assertIn(text, analysis["interpretation"])

    def test_positive_feasibility_cannot_follow_from_B_greater_than_A_alone(self):
        gate = self.value("PROTOCOL.json")["positive_feasibility_label_gate"]
        self.assertTrue(gate["required_if_label_used"])
        for text in ("strictly more exact", "best fixed reference", "every consumed matched random", "without losing"):
            self.assertIn(text, gate["criterion"])
        self.assertIn("prohibit", gate["completeness"])
        self.assertIn("never sufficient for feasibility or capability gain", gate["B_greater_than_A_alone"])

    def test_current_authoritative_quota_and_period_gate(self):
        p = self.value("PROTOCOL.json")
        gate = p["quota_gate"]
        self.assertEqual(gate["shared_quota_stop_percent"], 85)
        self.assertIn("fresh authoritative true", gate["ordinaryUsageAllowed"])
        self.assertIn("before each", gate["ordinaryUsageAllowed"])
        self.assertIn("period/reset drift", gate["period_consistency"])
        for text in ("stale", "missing meter", "unmetered", "old tranche"):
            self.assertIn(text, gate["meter_policy"])
        self.assertIn("ordinaryUsageAllowed", " ".join(p["future_execution_gate"]))

    def test_preparation_has_no_approvals_attempts_results_or_calls(self):
        status = self.value("INITIAL_STATUS.json")
        for key in ("parent_author_approval", "route_approval", "execution_integration_implemented", "predictions_executed",
                    "scientific_acceptance", "publication_authorized"):
            self.assertIs(status[key], False)
        for key in ("reservations", "consumed_attempts", "author_invocations", "author_charged_positions",
                    "paired_random_charged_positions", "random_controls_constructed"):
            self.assertEqual(status[key], 0)
        self.assertIsNone(status["candidate_results"])
        self.assertFalse(self.value("CONTRACT.json")["parent_author_approval"])
        self.assertEqual(self.value("STANDARD.json")["budget_per_arm"], 0)
        self.assertEqual(self.value("STANDARD.json")["proposals"], [])
        self.assertFalse(any("attempts/" in name or "tensors/" in name for name in self.payloads))

    def test_source_contains_no_execution_intake_or_random_construction(self):
        tree = ast.parse((ROOT / M.SOURCE_NAME).read_text())
        imports = []
        functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
        self.assertFalse(set(imports) & {"random", "numpy", "subprocess", "socket", "requests", "importlib", "nca_author_intake"})
        self.assertFalse(functions & {"reserve_attempt", "mark_submission_intent", "evaluate", "dispatch", "random_tensors"})
        self.assertIn("NEVER use the old mutation-specific", " ".join(self.value("PROTOCOL.json")["future_execution_gate"]))

    def test_freeze_then_verify_all_artifacts(self):
        expected = self.freeze()
        result = M.verify(ROOT, self.run, expected)
        self.assertTrue(result["verified"])
        self.assertEqual(result["contract_sha256"], expected)
        self.assertEqual(result["status"], "preparation_only_unapproved")
        for name, raw in self.payloads.items():
            self.assertEqual((self.run / name).read_bytes(), raw)

    def test_existing_or_partial_directory_never_resumed(self):
        self.run.mkdir()
        (self.run / "partial").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.freeze()
        self.assertEqual((self.run / "partial").read_bytes(), b"partial")

    def test_corrupted_original_packet_rejected(self):
        raw = self.payloads["author/A_PACKET.json"]
        for modified in (raw + b"\n", raw.replace(b"1488", b"1489", 1)):
            with self.assertRaisesRegex(ValueError, "arm A"):
                M.packet_pair(modified)

    def test_tampered_packet_fail_closed_even_if_contract_rehashed_locally(self):
        expected = self.freeze()
        path = self.run / "author/B_PACKET.json"
        value = json.loads(path.read_text())
        value["training_cases"][0]["input"][0][0] = 9
        path.write_bytes(M.json_bytes(value))
        with self.assertRaisesRegex(ValueError, "artifact mismatch"):
            M.verify(ROOT, self.run, expected)
        contract = json.loads((self.run / "CONTRACT.json").read_text())
        contract["artifacts"]["author/B_PACKET.json"]["sha256"] = M.sha256(path.read_bytes())
        (self.run / "CONTRACT.json").write_bytes(M.json_bytes(contract))
        new_hash = M.sha256((self.run / "CONTRACT.json").read_bytes())
        (self.run / "CONTRACT.sha256").write_text(new_hash + "\n")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            M.verify(ROOT, self.run, expected)
        with self.assertRaisesRegex(ValueError, "exact protocol"):
            M.verify(ROOT, self.run, new_hash)

    def test_frozen_protocol_budget_seed_or_order_tampering_rejected(self):
        expected = self.freeze()
        path = self.run / "PROTOCOL.json"
        original = path.read_bytes()
        for mutate in (lambda p: p.update(author_order=["B", "A"]),
                       lambda p: p["accounting"].update(maximum_author_invocations_total=3),
                       lambda p: p["random_controls"]["seeds"].update(A=2026100801),
                       lambda p: p.update(parent_author_approval=True)):
            value = M.strict_json(original)
            mutate(value)
            path.write_bytes(M.json_bytes(value))
            with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                M.verify(ROOT, self.run, expected)
        path.write_bytes(original)

    def test_hash_bound_historical_source_change_prevents_freeze(self):
        repo = self.fixture_repo()
        path = repo / M.HISTORY / "source/scripts/nca_author_intake.py"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            M.freeze(repo, self.run)
        self.assertFalse(self.run.exists())

    def test_hash_bound_runtime_or_control_change_prevents_freeze(self):
        repo = self.fixture_repo()
        for name in ("source/src/cellforge/runtime/ops.py", "baseline/identity.json"):
            path = repo / M.PHASE1 / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "source hash mismatch"):
                M.freeze(repo, self.run)
            path.write_bytes(original)
            self.assertFalse(self.run.exists())

    def test_hash_bound_audit_schema_and_halt_change_prevents_freeze(self):
        repo = self.fixture_repo()
        for name in ("RESPONSE_SCHEMA.json", "AUDIT_ENCODING_SPEC.json", "NATIVE_EVENT_SCHEMA_SPEC.json", "HALT.json"):
            path = repo / M.HISTORY / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "source hash mismatch"):
                M.freeze(repo, self.run)
            path.write_bytes(original)
            self.assertFalse(self.run.exists())

    def test_historical_terminal_and_source_bindings_retained(self):
        sources = {record["path"]: record for record in self.value("SOURCE_BINDINGS.json")["records"]}
        for name, expected in M.ANCHORS.items():
            self.assertEqual(sources[name]["sha256"], expected)
        self.assertFalse(self.value("SOURCE_BINDINGS.json")["historical_approvals_transferable"])
        self.assertTrue(self.value("CONTRACT.json")["historical_tranche_closed"])

    def test_unknown_attempt_file_or_missing_file_blocks_verification(self):
        expected = self.freeze()
        (self.run / "RESERVATION.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "unexpected or missing"):
            M.verify(ROOT, self.run, expected)
        (self.run / "RESERVATION.json").unlink()
        (self.run / "INITIAL_STATUS.json").unlink()
        with self.assertRaises(FileNotFoundError):
            M.verify(ROOT, self.run, expected)

    def test_unexpected_empty_attempt_directory_blocks_verification(self):
        expected = self.freeze()
        (self.run / "attempts").mkdir()
        with self.assertRaisesRegex(ValueError, "unexpected or missing preparation directory"):
            M.verify(ROOT, self.run, expected)

    def test_external_hash_required_and_sidecar_cannot_authorize(self):
        expected = self.freeze()
        for value in (None, "", "wrong", "f" * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                M.verify(ROOT, self.run, value)
        (self.run / "CONTRACT.sha256").write_text("f" * 64 + "\n")
        with self.assertRaisesRegex(ValueError, "sidecar"):
            M.verify(ROOT, self.run, expected)

    def test_path_traversal_and_symlink_rejected(self):
        for name in ("../escape", "/absolute", "a/../../escape", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                M.safe_path(self.temp_root, name)
        (self.temp_root / "linked").symlink_to(ROOT, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            M.safe_path(self.temp_root, "linked/AGENTS.md")

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_invalid_utf8(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b"\xff"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                M.strict_json(raw)


if __name__ == "__main__":
    unittest.main()
