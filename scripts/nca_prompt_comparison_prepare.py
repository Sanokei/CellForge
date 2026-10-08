"""Prepare/verify a NEW, unapproved two-founder prompt comparison.

This module only reads existing evidence and freezes data. It never imports the
old intake/runtime, constructs tensors, reserves attempts, dispatches an author,
scores a candidate, or changes the terminal historical tranche.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

PHASE1 = "runs/weights-nca-feasibility-20261008"
HISTORY = "runs/weights-nca-author-tranche-20261008"
RUN_NAME = "runs/weights-nca-prompt-comparison-20261008"
SOURCE_NAME = "scripts/nca_prompt_comparison_prepare.py"
TEST_NAME = "tests/test_nca_prompt_comparison_prepare.py"
FOUNDER_SHA = "f26d3567f9d02903bbb525072803ed4713714108665a30458df99f2cd1532e37"
SCHEMA_SHA = "d7ebc66e50abf32d43db94d4984201725d8dc8c122f2795a2af57aef1d272f72"
VOCABULARY = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
SEEDS = {"A": 2026100803, "B": 2026100804}
POSITIONS = 1488
ANCHORS = {
    PHASE1 + "/MANIFEST.json": "11d371121214c1bd0caeaa2c78af0ebc92affab9f5f70c09728c6a6240373326",
    PHASE1 + "/BASELINE_RESULT.json": "c2bc1ca4841dca1d9f9999e6bfb7bab825eca362f3b805e7713c63bcdcdd2072",
    PHASE1 + "/REPLAY_RESULT.json": "2abe7a62ff7ac3a735a07d4f456fbc375733ca506c641d097c04ff5dd43cb4ff",
    HISTORY + "/CONTRACT.json": "18f8b7c1be20991b3ea021aa43d7bfd4a821aab67bc547989c2a9f222535d7ee",
    HISTORY + "/FOUNDER_PACKET.json": FOUNDER_SHA,
    HISTORY + "/RESPONSE_SCHEMA.json": SCHEMA_SHA,
    HISTORY + "/EVALUATOR_ONLY.json": "0ac34f4391cb00c37dd4fcef3fa041a7ecf502856440e89928e9816c02d90672",
    HISTORY + "/AUDIT_ENCODING_SPEC.json": "45e069f7f5487c7320fa665dd3776eec9276b70ed1512ec58507880b714c36a1",
    HISTORY + "/NATIVE_EVENT_SCHEMA_SPEC.json": "f9027f1aedcb4aa8cd0cb0cc93e85a1d8403432bcb07c7e1ce971522120edf8b",
    HISTORY + "/INVOCATION_ACCOUNTING_SPEC.json": "762fcd24ad82c30ac708e91ad5ac7e6391fe3474f30ca01d5b42508da78f6363",
    HISTORY + "/HALT.json": "e2bf8f7199d8e3dbddc88b0ed96508440628ae4d44cb5452135e9a37a549fdb9",
    HISTORY + "/TERMINAL.json": "052cf1719def76d3655dfb68591fb3aedcc6fb3f244e8b18c331875e9cf1ee9f",
    HISTORY + "/source/scripts/nca_author_intake.py": "39ce4aa1798549854299002e02b3a74f12c4c824851686e0bf6c14f001deb0f6",
    "scripts/nca_author_intake.py": "39ce4aa1798549854299002e02b3a74f12c4c824851686e0bf6c14f001deb0f6",
    "tests/test_nca_author_intake.py": "37fc5c144d652431e77c08896553c58dd15937c5453ed1c4b0fa3a07c59eaa3f",
    "tests/test_nca_native_event_envelopes.py": "4dd49650ee9230d7086bd4ae5583a0fc1d8962f853c77b45cd1d67f7288a1424",
    "reports/NCA_TRANCHE_RESULT_20261008.md": "27eec8761427a73dde252cc0abd4b2b2cef17afcb56069bab66cf92473ffc9da",
}
OLD_PROHIBITION = (
    "No demonstration conditioning, task routing, per-case weights, fitting, "
    "or executable structure changes."
)
NEW_PROHIBITION = (
    "No demonstration conditioning, task routing, per-case weights, gradient or "
    "runtime fitting, or executable structure changes. Manual numerical reasoning "
    "to choose literal weights is allowed; do not use tools, gradients, or code."
)
OBJECTIVE = (
    "Choose the one global tensor set to maximize the number of exact output-grid "
    "matches across all 17 training demonstrations. Break ties by lower mean "
    "per-demonstration normalized pixel error: for each demonstration, mismatched "
    "pixels divided by output pixels, then the unweighted mean across all 17 "
    "demonstrations."
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def json_bytes(value):
    return canonical(value) + b"\n"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=pairs,
                      parse_constant=lambda item: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + item)))


def safe_path(root, name):
    root = Path(root).absolute()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("relative contained artifact path required")
    result = root / relative
    if any(part.is_symlink() for part in (result, *result.parents)):
        raise ValueError("symlink artifact prohibited")
    return result


def checked_bytes(repo, name, expected):
    raw = safe_path(repo, name).read_bytes()
    if sha256(raw) != expected:
        raise ValueError("source hash mismatch: " + name)
    return raw


def source_evidence(repo):
    """Hash/read only: no import or execution of historical runtime/intake."""
    evidence = {}
    def bind(name, expected, role):
        raw = checked_bytes(repo, name, expected)
        record = {"path": name, "sha256": expected, "bytes": len(raw), "role": role}
        if name in evidence and evidence[name]["sha256"] != expected:
            raise ValueError("conflicting source binding: " + name)
        evidence.setdefault(name, record)
        return raw
    for name, expected in ANCHORS.items():
        bind(name, expected, "historical_read_only_anchor_not_execution_approval")
    manifest = strict_json(checked_bytes(repo, PHASE1 + "/MANIFEST.json", ANCHORS[PHASE1 + "/MANIFEST.json"]))
    baseline = strict_json(checked_bytes(repo, PHASE1 + "/BASELINE_RESULT.json", ANCHORS[PHASE1 + "/BASELINE_RESULT.json"]))
    replay = strict_json(checked_bytes(repo, PHASE1 + "/REPLAY_RESULT.json", ANCHORS[PHASE1 + "/REPLAY_RESULT.json"]))
    for relative, expected in manifest["source_sha256"].items():
        bind(PHASE1 + "/source/" + relative, expected, "frozen_phase1_runtime_source")
    for item in (manifest, baseline, replay):
        for relative, expected in item["artifact_sha256"].items():
            bind(PHASE1 + "/" + relative, expected, "frozen_phase1_artifact")
    historical = strict_json(checked_bytes(repo, HISTORY + "/CONTRACT.json", ANCHORS[HISTORY + "/CONTRACT.json"]))
    for relative, expected in historical["artifacts"].items():
        bind(HISTORY + "/" + relative, expected, "closed_tranche_artifact_not_prospective_arm")
    terminal = strict_json(checked_bytes(repo, HISTORY + "/TERMINAL.json", ANCHORS[HISTORY + "/TERMINAL.json"]))
    if terminal["additional_author_calls_allowed"] != 0 or terminal["author_invocations"] != 2:
        raise ValueError("historical tranche must remain closed")
    if baseline["stop_reasons"] or not replay["replay"]["passed"]:
        raise ValueError("saved phase1 baseline/replay invalid")
    if os.path.lexists(Path(repo) / PHASE1 / "STOP"):
        raise ValueError("phase1 STOP present")
    return {"schema": "nca-prompt-comparison-source-bindings/1",
            "historical_approvals_transferable": False,
            "records": [evidence[name] for name in sorted(evidence)]}, manifest, baseline


def packet_pair(original):
    if sha256(original) != FOUNDER_SHA:
        raise ValueError("arm A must be exact historical founder packet bytes")
    a = strict_json(original)
    if json_bytes(a) != original:
        raise ValueError("unexpected original encoding")
    if set(a) != {"schema", "architecture", "tensor_schema", "training_cases", "response_contract"}:
        raise ValueError("unexpected founder packet fields")
    sharing = a["architecture"]["sharing"]
    if sharing.count(OLD_PROHIBITION) != 1:
        raise ValueError("exact original sharing prohibition required")
    b = copy.deepcopy(a)
    b["architecture"]["sharing"] = sharing.replace(OLD_PROHIBITION, NEW_PROHIBITION, 1)
    b["objective"] = OBJECTIVE
    raw_b = json_bytes(b)
    patch = [{"op": "replace", "path": "/architecture/sharing", "old_value": sharing,
              "value": b["architecture"]["sharing"]},
             {"op": "add", "path": "/objective", "value": OBJECTIVE}]
    unchanged_a, unchanged_b = copy.deepcopy(a), copy.deepcopy(b)
    del unchanged_a["architecture"]["sharing"]
    del unchanged_b["architecture"]["sharing"]
    del unchanged_b["objective"]
    if canonical(unchanged_a) != canonical(unchanged_b):
        raise ValueError("nonpermitted packet change")
    diff = {"schema": "nca-prompt-comparison-exact-diff/1", "patch": patch,
            "permitted_paths": ["/architecture/sharing", "/objective"],
            "A": {"bytes": len(original), "sha256": sha256(original)},
            "B": {"bytes": len(raw_b), "sha256": sha256(raw_b)},
            "unchanged_canonical_projection_sha256": sha256(canonical(unchanged_a)),
            "response_schema_sha256_both_arms": SCHEMA_SHA,
            "encoding": "Same canonical UTF-8 JSON plus one LF; identical values/subtree encodings outside the two paths.",
            "sharing_exact_substring_change": {"old": OLD_PROHIBITION, "new": NEW_PROHIBITION},
            "design_limit": "Bundled objective clarification and fitting clarification; cannot isolate either component's effect."}
    return original, raw_b, diff


def protocol():
    return {
        "schema": "nca-prompt-contract-comparison/1",
        "status": "preparation_only_unapproved",
        "parent_author_approval": False,
        "execution_integration_implemented": False,
        "author_order": ["A", "B"],
        "authors": {arm: {"kind": "new_independent_founder", "parent_weights": None,
                          "packet": "author/" + arm + "_PACKET.json", "random_seed": SEEDS[arm]}
                    for arm in ("A", "B")},
        "historical_founder_is_prospective_A": False,
        "mutations_allowed": 0,
        "cross_arm_feedback_allowed": False,
        "author_inputs": {"scientific_inputs_only": ["respective_packet", "unchanged_response_schema"],
                          "forbidden": ["other_arm_packet", "other_arm_response", "other_arm_results", "prior_weights",
                                        "prior_results", "control_weights", "control_results", "task_ids", "query_inputs",
                                        "query_answers", "evaluation_split", "repository_context", "global_research_context"],
                          "generic_platform_context": "Ordinary platform/system context and built-in skill descriptions may remain; disclose actual route."},
        "fixed_architecture": {"iterations": 8, "arrays": 96, "positions": POSITIONS,
                               "global_tensor_set": True, "vocabulary": VOCABULARY},
        "author_route": {"model": "gpt-6.1-sol", "reasoning_effort": "low", "concurrency": 1,
                         "wall_seconds": 120, "memory_bytes": 2147483648,
                         "same_limits_both_arms": True, "official_endpoint_and_existing_authentication": True,
                         "request_retries": 0, "stream_retries": 0, "websocket": False,
                         "audit": "complete_event_stream_conservative_allowlist",
                         "qualified_provenance": "observed_zero_tool_use_only",
                         "preventive_tool_isolation_claimed": False,
                         "new_route_attestation_required": True},
        "accounting": {"maximum_author_invocations_total": 2, "maximum_author_invocations_each_prompt_arm": 1,
                       "maximum_candidates_each_invocation": 1, "positions_each_consumed_author_attempt": POSITIONS,
                       "positions_each_matched_random_attempt": POSITIONS,
                       "maximum_author_positions_total": 2976, "maximum_matched_random_positions_total": 2976,
                       "charge_invalid_refused_neutral_duplicate_timeout_uncertain": True,
                       "submission_intent_consumes_once": True, "retry_refund_replacement_allowed": False,
                       "backend_internal_requests": "unknown", "credits": "unknown", "monetary_equality": "not established",
                       "diagnostic_invocations": "No new diagnostics authorized; any future approved overhead must be separately disclosed."},
        "random_controls": {"seeds": SEEDS, "vocabulary": VOCABULARY,
                            "distribution": "Independent uniform choice over the same seven values for every position.",
                            "order": "Lexicographically sorted tensor names, C/row-major array positions.",
                            "generation_gate": "Only after the corresponding new author invocation is actually consumed; never before dispatch or merely upon reservation.",
                            "terminal_gate": "A terminal STOP forbids further construction/evaluation, even if positions were already charged; report incomplete controls honestly.",
                            "historical_random_reuse_allowed": False, "constructed_at_freeze": False,
                            "rng_implementation": "Future execution integration must freeze and test exact RNG implementation before dispatch; seeds alone are not an implemented generator."},
        "fixed_controls": {"reuse": ["identity", "neighbor_majority", "zero"],
                           "policy": "Reuse original phase1 literal tensors, saved predictions and result bytes unchanged; no redraw or reevaluation.",
                           "author_exposure_allowed": False},
        "continuation_gate_B": "A must have a complete valid response, passed observed-zero-tool audit, valid frozen-runtime evaluation and complete valid matched random control. B remains a fresh founder without A feedback, even if A is perfect. No A failure may be replaced by the historical founder.",
        "terminal_stop": {"conditions": ["invalid_response", "refusal", "audit_failure", "tool_attempt", "unknown_event",
                                         "stream_gap", "truncation", "timeout", "uncertain_completion", "route_mismatch",
                                         "source_hash_mismatch", "runtime_failure", "invalid_control"],
                          "action": "Terminal STOP, no repair, replacement, retry, mutation or continued scoring/authoring.",
                          "A_timeout": "Do not invoke B; prospective A-versus-B comparison is inconclusive.",
                          "B_failure": "No replacement; report incomplete/inconclusive comparison, not a synthetic zero score."},
        "response_handling": "Save exact complete original final-message UTF-8 bytes durably once before parsing; no trimming, added newline, correction, repair, formulas or executable fields. Apply unchanged schema/intake constraints.",
        "audit_handling": "Validate every full raw native event before projection; exact allowlisted keys/types only; hash each exact JSONL line including terminators and the complete trace. Require complete lifecycle and one final message. Preserve all five native usage counters, bound to raw event/trace. Private reasoning bodies remain restricted. Historical audit specs are references, not transferable approvals.",
        "evaluation": "Future execution only: frozen phase1 float64 reference runtime, graph, full tensor injection, leave-one-out context, limits and ordering. No current candidate or control execution.",
        "analysis": {"primary": "B exact training demonstrations minus A exact training demonstrations, denominator 17.",
                     "secondary": ["Exact tasks (denominator 6)", "Unweighted mean per-demonstration normalized pixel error",
                                   "Numerical identity equivalence (signed-zero-normalized comparison, retain original bytes)",
                                   "Complete 17-demo prediction-vector identity equivalence (panel-local only)",
                                   "Descriptive comparisons to fixed controls and each newly matched random control"],
                     "tie_break": "If exact-demonstration counts tie, compare lower mean per-demonstration normalized pixel error; report as secondary, not an exact-match gain.",
                     "interpretation": "One prospective pair, fixed A-then-B order, bundled prompt changes: descriptive only; no causal attribution, significance, replication, architecture incapability, or independent generalization claim.",
                     "denominators": {"restricted_tasks": 6, "restricted_demonstrations": 17,
                                      "full_inventory_tasks": 11, "full_inventory_demonstrations": 32,
                                      "size_excluded_tasks": 5, "size_excluded_demonstrations": 15},
                     "qualification": "Previously exposed public-training development only; not held-out, whole-panel feasibility, ARC benchmark, admission, or ONE_STANDARD qualification."},
        "positive_feasibility_label_gate": {
            "required_if_label_used": True,
            "criterion": "The candidate must score strictly more exact training demonstrations than the best fixed reference and every consumed matched random control, without losing any selected-reference solved demonstration.",
            "completeness": "Every required candidate/control must be valid and complete; missing, invalid or unexecuted charged controls prohibit a positive-feasibility label.",
            "B_greater_than_A_alone": "Descriptive prompt-comparison difference only; never sufficient for feasibility or capability gain.",
            "claim_scope": "Even passing this criterion is exposed-training development evidence only, not generalization or a causal prompt effect."},
        "quota_gate": {"ordinaryUsageAllowed": "Require a fresh authoritative true value before each prospective dispatch; unavailable, stale or false blocks dispatch.",
                       "shared_quota_stop_percent": 85,
                       "period_consistency": "Bind the current shared quota period and reset timestamp; period/reset drift requires a fresh attestation and approval before continuing.",
                       "meter_policy": "Current authoritative shared usage must be below the 85% STOP. No stale meter, missing meter, unmetered permission, or old tranche quota approval may authorize execution.",
                       "credit_accounting": "Do not infer exact credits or monetary cost from coarse subscription percentages or token counts."},
        "future_execution_gate": ["Explicit new approval bound to this contract and both exact packet/schema hashes.",
                                  "New reviewed execution integration for two independent founders; NEVER use the old mutation-specific reserve/dispatch path for B.",
                                  "Separate current route, resource, input-isolation, complete-event-audit and quota attestations.",
                                  "Fresh authoritative ordinaryUsageAllowed and current shared usage below the 85% STOP, with no unapproved period/reset drift, stale meter or unmetered permission.",
                                  "Freeze exact RNG/invocation capture implementation and verify independent replay before result claims.",
                                  "No dispatch, reservations, model/API calls, random generation, evaluation, source publication, commit, push, deployment or installation authorized by preparation."],
    }


def build_payloads(repo):
    evidence, phase1, baseline = source_evidence(repo)
    raw_a, raw_b, diff = packet_pair(checked_bytes(repo, HISTORY + "/FOUNDER_PACKET.json", FOUNDER_SHA))
    schema = checked_bytes(repo, HISTORY + "/RESPONSE_SCHEMA.json", SCHEMA_SHA)
    evaluator = strict_json(checked_bytes(repo, HISTORY + "/EVALUATOR_ONLY.json", ANCHORS[HISTORY + "/EVALUATOR_ONLY.json"]))
    coverage = phase1["panel"]
    expected_counts = {"all_tasks": 11, "all_demonstrations": 32, "compatible_tasks": 6,
                       "compatible_demonstrations": 17, "excluded_tasks": 5, "excluded_demonstrations": 15}
    if any(coverage[name] != expected for name, expected in expected_counts.items()) or evaluator["coverage"] != coverage:
        raise ValueError("coverage inventory mismatch")
    panel = strict_json(safe_path(repo, PHASE1 + "/PANEL.json").read_bytes())
    compatible = [task for task in panel["tasks"] if task["task_id"] in coverage["compatible_task_ids"]]
    cases, mapping = [], []
    for task in compatible:
        if set(task) != {"task_id", "train"}:
            raise ValueError("training-only panel required")
        for index, pair in enumerate(task["train"]):
            if set(pair) != {"input", "output"}:
                raise ValueError("training-pair fields changed")
            label = "case_%02d" % (len(cases) + 1)
            cases.append({"case": label, **pair})
            mapping.append({"case": label, "task_id": task["task_id"], "demo_index": index})
    a = strict_json(raw_a)
    if cases != a["training_cases"] or mapping != evaluator["case_map"]:
        raise ValueError("frozen author training cases mismatch")
    if phase1["graph"]["tensor_schema"] != a["tensor_schema"]:
        raise ValueError("tensor schema mismatch")
    fixed = {}
    for name in ("identity", "neighbor_majority", "zero"):
        names = ["tensors/" + name + ".json", "baseline/" + name + ".json", "replay/" + name + ".json"]
        fixed[name] = {"source_records": [item for item in evidence["records"] if item["path"] in {PHASE1 + "/" + n for n in names}],
                       "summary": baseline["arms"][name]}
        if len(fixed[name]["source_records"]) != 3:
            raise ValueError("incomplete fixed reference bindings")
    status = {"schema": "nca-prompt-comparison-preparation-status/1", "parent_author_approval": False,
              "route_approval": False, "quota_attestation": "pending", "execution_integration_implemented": False,
              "reservations": 0, "consumed_attempts": 0, "author_invocations": 0,
              "author_charged_positions": 0, "paired_random_charged_positions": 0,
              "random_controls_constructed": 0, "predictions_executed": False,
              "candidate_results": None, "scientific_acceptance": False, "publication_authorized": False}
    standard = {"arms": {"llm": {}, "random": {}}, "budget_per_arm": 0, "founders": [], "genomes": [],
                "niches": [], "proposals": [], "seeds": list(SEEDS.values()), "seeds_declared_before_run": True,
                "scientific_acceptance": False,
                "limitations": ["Unapproved preparation only; no new authors, reservations, tensors or evaluations. Broad qualification remains red/incomplete."]}
    payloads = {"author/A_PACKET.json": raw_a, "author/B_PACKET.json": raw_b,
                "author/RESPONSE_SCHEMA.json": schema,
                "PROMPT_DIFF.json": json_bytes(diff), "PROTOCOL.json": json_bytes(protocol()),
                "SOURCE_BINDINGS.json": json_bytes(evidence), "INITIAL_STATUS.json": json_bytes(status),
                "STANDARD.json": json_bytes(standard),
                "EVALUATOR_ONLY.json": json_bytes({"schema": "nca-prompt-comparison-evaluator-only/1",
                    "author_exposure_allowed": False, "coverage": coverage, "case_map": mapping,
                    "phase1_path": PHASE1, "fixed_references": fixed,
                    "runtime": phase1["runtime"], "resource_limits": phase1["resource_limits"],
                    "graph": phase1["graph"], "historical_founder_is_not_arm_A": True}),
                "source/" + SOURCE_NAME: safe_path(repo, SOURCE_NAME).read_bytes(),
                "source/" + TEST_NAME: safe_path(repo, TEST_NAME).read_bytes()}
    contract = {"schema": "nca-prompt-comparison-preparation-contract/1", "parent_author_approval": False,
                "new_independent_run": RUN_NAME, "historical_tranche": HISTORY, "historical_tranche_closed": True,
                "execution_integration_implemented": False, "frozen_before_any_new_author": True,
                "artifacts": {name: {"sha256": sha256(raw), "bytes": len(raw)} for name, raw in sorted(payloads.items())}}
    payloads["CONTRACT.json"] = json_bytes(contract)
    return payloads


def write_once(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def freeze(repo, run):
    run = Path(run).absolute()
    if os.path.lexists(run) or any(part.is_symlink() for part in run.parents):
        raise ValueError("fresh nonsymlink preparation directory required")
    # Complete all validation before creating the directory. A partial write is
    # deliberately not resumable: exclusive files and fresh-directory check stop.
    payloads = build_payloads(Path(repo))
    run.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        write_once(safe_path(run, name), raw)
    expected = sha256(payloads["CONTRACT.json"])
    write_once(run / "CONTRACT.sha256", (expected + "\n").encode())
    return expected


def verify(repo, run, expected):
    """Require an independently supplied contract hash, not just a local sidecar."""
    run = Path(run)
    if type(expected) is not str or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("exact external contract SHA256 required")
    checked_bytes(run, "CONTRACT.json", expected)
    if safe_path(run, "CONTRACT.sha256").read_bytes() != (expected + "\n").encode():
        raise ValueError("contract sidecar mismatch")
    payloads = build_payloads(Path(repo))
    if sha256(payloads["CONTRACT.json"]) != expected:
        raise ValueError("preparation contract does not match verified sources and exact protocol")
    for name, raw in payloads.items():
        if safe_path(run, name).read_bytes() != raw:
            raise ValueError("frozen artifact mismatch: " + name)
    actual, actual_dirs = set(), set()
    for path in run.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink artifact prohibited")
        if path.is_file():
            actual.add(path.relative_to(run).as_posix())
        elif path.is_dir():
            actual_dirs.add(path.relative_to(run).as_posix())
        else:
            raise ValueError("unexpected special artifact")
    if actual != set(payloads) | {"CONTRACT.sha256"}:
        raise ValueError("unexpected or missing preparation artifact; execution state is not allowed")
    expected_dirs = {parent.as_posix() for name in payloads for parent in Path(name).parents if parent != Path(".")}
    if actual_dirs != expected_dirs:
        raise ValueError("unexpected or missing preparation directory; execution state is not allowed")
    return {"verified": True, "contract_sha256": expected, "status": "preparation_only_unapproved",
            "author_invocations": 0, "reservations": 0, "random_controls_constructed": 0,
            "predictions_executed": False, "artifacts": strict_json(payloads["CONTRACT.json"])["artifacts"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify"])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path)
    parser.add_argument("--contract-sha256")
    args = parser.parse_args(argv)
    run = args.run if args.run is not None else args.repo / RUN_NAME
    if args.action == "prepare":
        if args.contract_sha256 is not None:
            parser.error("prepare does not accept an approval or expected contract")
        print(json.dumps({"contract_sha256": freeze(args.repo, run), "author_approval": False}))
    else:
        if args.contract_sha256 is None:
            parser.error("verify requires an externally retained --contract-sha256")
        print(json.dumps(verify(args.repo, run, args.contract_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
