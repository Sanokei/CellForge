"""Finite, data-only T8 author intake. No network, model calls, or dispatch code.

The real freeze starts unapproved. Only prepare/verify are public CLI operations.
Future host operations require exact external approval and host audit receipts;
their presence is checked, not fabricated. Tests use separate temporary fixtures.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import struct
import subprocess
import sys
import time

PHASE1_SHA = "11d371121214c1bd0caeaa2c78af0ebc92affab9f5f70c09728c6a6240373326"
RUNNER_SHA = "c5c3ea59e389f100693365293edadfe5f5fd1d527f8247917fc2173b15e704c7"
BASELINE_SHA = "c2bc1ca4841dca1d9f9999e6bfb7bab825eca362f3b805e7713c63bcdcdd2072"
REPLAY_SHA = "2abe7a62ff7ac3a735a07d4f456fbc375733ca506c641d097c04ff5dd43cb4ff"
VOCABULARY = (-1, -.5, -.25, 0, .25, .5, 1)
POSITIONS = 1488
MAX_RESPONSE_BYTES = 262144
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
SOURCE_NAME = "scripts/nca_author_intake.py"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def numeric_fingerprint(tensors):
    """Numerical duplicate key, separate from literal/byte provenance hashes.

    Integer/float spellings and either signed zero compare numerically equal;
    actual injection and per-array hashes still preserve float64 sign bits.
    """
    hashed = hashlib.sha256()
    def flatten(value):
        if isinstance(value, list):
            for child in value:
                yield from flatten(child)
        else:
            yield float(value)
    for item in tensor_schema():
        hashed.update(item["name"].encode() + b"\0")
        for value in flatten(tensors[item["name"]]):
            hashed.update(struct.pack("<d", 0.0 if value == 0 else value))
    return hashed.hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))


def read_json(path):
    return strict_json(Path(path).read_bytes())


def write_once(path, raw):
    """Exclusive durable creation. An interrupted/partial file blocks resumption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    write_once(path, canonical(value) + b"\n")


def bound_path(root, name):
    path = root / name
    if path.is_absolute() and not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes run")
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("symlink artifacts prohibited")
    return path


def tensor_schema():
    shapes = {}
    for row in range(12):
        for chunk in range(3):
            shapes[f"nca.h{row}.c{chunk}.W"] = [1, 36]
            shapes[f"nca.h{row}.c{chunk}.b"] = [1]
        shapes[f"nca.o{row}.W"] = [1, 12]
        shapes[f"nca.o{row}.b"] = [1]
    return [{"name": name, "shape": shape, "positions": math.prod(shape)} for name, shape in sorted(shapes.items())]


def architecture():
    return {
        "state": "A 30x30 canvas has 12 float64 channels. At t=0, channels 0..9 are the input color one-hot and channels 10..11 are zero. All channels outside the input rectangle are zero.",
        "feature_order": "For dy in (-1,0,1), then dx in (-1,0,1), concatenate channels c=0..11. Feature index j=12*(3*(dy+1)+(dx+1))+c samples S_t[y-dy,x-dx,c]. Shifts fill outside the 30x30 canvas with zero; there is no wraparound.",
        "hidden": "For r=0..11 and k=0..2: p[r,k] = sum(j=0..35, nca.hr.ck.W[0][j] * F[36*k+j]) + nca.hr.ck.b[0]. Here hr and ck substitute the decimal row r and chunk k. H[r] = max(0, (p[r,0]+p[r,1])+p[r,2]). Three affine partials, including all three biases, are summed before ReLU.",
        "output": "For r=0..11: S_(t+1)[y,x,r] = input_mask[y,x] * (sum(j=0..11, nca.or.W[0][j] * H[j]) + nca.or.b[0]); or substitutes decimal row r. No output ReLU, residual addition, stochastic update, or clipping. Outside the original rectangle the mask is zero.",
        "iteration_and_decode": "Apply exactly 8 shared updates t=0..7 with identical tensors at every location and time. Output color is argmax of S_8 channels 0..9; ties choose lowest color. Height and width equal the input height and width.",
        "sharing": "One global set of all 96 arrays, totaling 1488 literal positions, applies to every case. No demonstration conditioning, task routing, per-case weights, fitting, or executable structure changes. There are 48 active affine microcells.",
        "ordering": "Parameter names are lexicographically ordered; array positions are C/row-major. Numeric channel order 0..11 is used inside concatenations, even though tensor names use lexicographic order.",
    }


def build_founder_packet(tasks):
    cases, mapping = [], []
    for task in tasks:
        if set(task) != {"task_id", "train"}:
            raise ValueError("only sanitized training tasks permitted")
        for index, pair in enumerate(task["train"]):
            if set(pair) != {"input", "output"}:
                raise ValueError("unexpected training pair field")
            label = f"case_{len(cases) + 1:02d}"
            cases.append({"case": label, "input": pair["input"], "output": pair["output"]})
            mapping.append({"case": label, "task_id": task["task_id"], "demo_index": index})
    packet = {
        "schema": "nca-data-only-founder-packet/1", "architecture": architecture(),
        "tensor_schema": tensor_schema(), "training_cases": cases,
        "response_contract": {"schema": "nca-author-tensors/1", "exact_top_level_keys": ["schema", "tensors"],
            "tensors": "Object with exactly every named tensor in tensor_schema, each with its full nested shape. Emit all 1488 numeric literals. No omitted arrays, defaults, ellipses, formulas, source/program fields, commentary, Markdown, or extra keys.",
            "vocabulary": list(VOCABULARY), "stored_positions": POSITIONS,
            "format": "One UTF-8 JSON object only. Duplicate keys, nonfinite numbers, booleans, wrong shapes and values outside the vocabulary are rejected. Use no tools. Produce literal data only."},
    }
    return packet, mapping


def validate_response(raw, schema=None):
    if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("response byte limit/type")
    value = strict_json(raw)
    if type(value) is not dict or set(value) != {"schema", "tensors"} or value["schema"] != "nca-author-tensors/1":
        raise ValueError("exact data-only envelope required; executable/extra fields forbidden")
    tensors = value["tensors"]
    schema = tensor_schema() if schema is None else schema
    if type(tensors) is not dict or set(tensors) != {item["name"] for item in schema}:
        raise ValueError("exact tensor name set required")
    def check(item, shape):
        if shape:
            if type(item) is not list or len(item) != shape[0]:
                raise ValueError("wrong tensor shape")
            return sum(check(child, shape[1:]) for child in item)
        if type(item) not in (int, float) or not math.isfinite(item) or item not in VOCABULARY:
            raise ValueError("literal finite non-boolean vocabulary member required")
        return 1
    count = sum(check(tensors[item["name"]], item["shape"]) for item in schema)
    if count != POSITIONS:
        raise ValueError("full 1488 positions required")
    return value


def verify_phase1(run):
    for name, expected in (("MANIFEST.json", PHASE1_SHA), ("BASELINE_RESULT.json", BASELINE_SHA), ("REPLAY_RESULT.json", REPLAY_SHA)):
        if file_hash(bound_path(run, name)) != expected:
            raise ValueError("phase1 binding failure: " + name)
    if os.path.lexists(run / "STOP"):
        raise ValueError("phase1 STOP present")
    manifest = read_json(run / "MANIFEST.json")
    for name, expected in manifest["source_sha256"].items():
        if file_hash(bound_path(run, "source/" + name)) != expected:
            raise ValueError("phase1 saved source binding failure: " + name)
    if manifest["source_sha256"]["scripts/nca_feasibility_preflight.py"] != RUNNER_SHA:
        raise ValueError("phase1 runner mismatch")
    for collection in (manifest["artifact_sha256"], read_json(run / "BASELINE_RESULT.json")["artifact_sha256"],
                       read_json(run / "REPLAY_RESULT.json")["artifact_sha256"]):
        for name, expected in collection.items():
            if file_hash(bound_path(run, name)) != expected:
                raise ValueError("phase1 artifact binding failure: " + name)
    result = read_json(run / "BASELINE_RESULT.json")
    replay = read_json(run / "REPLAY_RESULT.json")
    if result["stop_reasons"] or not replay["replay"]["passed"]:
        raise ValueError("baseline/replay does not permit an author decision")
    if manifest["graph"]["tensor_schema"] != tensor_schema():
        raise ValueError("frozen tensor schema mismatch")
    return manifest, result


def protocol():
    return {
        "maximum_attempts": 2, "attempts": ["founder", "one_conditional_mutation"],
        "mutation_condition": "Valid founder, verified no-tool host audit, completed valid controls and at least one failed training demonstration. No mutation after invalid/refusal, unknown submission, perfect founder, source mismatch or STOP.",
        "charge": {"author_positions_per_consumed_attempt": POSITIONS, "paired_random_positions_per_consumed_attempt": POSITIONS,
                   "maximum_each_arm": 2976, "charged_even_if": ["invalid", "refused", "neutral", "numeric_duplicate", "functional_duplicate", "submitted_unknown"],
                   "unknown_policy": "Reserve full budgets before dispatch; submission intent conservatively consumes once. Unknown/uncertain outcomes stop the entire tranche without retry or refund."},
        "random": {"first": "Reuse phase1 paired_random_1 tensors and complete saved result exactly; never redraw or reevaluate.",
                   "second_seed": 2026100802, "second": "Construct and evaluate only if second author attempt is actually consumed. Uniform independent choice of the common seven values in sorted-name, C/row-major order."},
        "raw_response": "Write complete original bytes once durably before parsing or receipt validation; never repair or overwrite invalid bytes. No recapture, retry or resampling.",
        "author_route": "Pending. Approval must bind parent identity, exact packet and contract, packet-only input isolation excluding repository/global context, complete-event-stream audit route and quota/credit attestation references before reservation. Only observed-zero-tool provenance is claimed; preventive tool isolation is not claimed.",
        "audit": "Every response requires host-captured packet/raw/submission-bound metadata for a complete CLI event stream, under a conservative lifecycle/reasoning-summary/final-message allowlist. Tools, unknown events, gaps, truncation, timeout or uncertain completion consume the attempt and terminally halt without retry or claim. Publish classified event hashes and usage only; the full raw trace remains restricted to parent route audit. Absent declared tools does not prove observed zero tool use; parent attestation is not a cryptographic platform receipt.",
        "evaluation": "Only frozen phase1 runtime and graph descriptor; reference float64, original leave-one-out context, full literal injection, unchanged limits, 17-demo/6-task scores plus all-11/32 coverage. Invalid candidate/control scores are null; invalid controls block claims and further authoring.",
        "claim": "Strictly more exact demos than best fixed reference and every charged random; preserve selected baseline solved demos. Mutation additionally strictly improves founder and preserves every founder-solved demo. No crossover, admission, held-out, benchmark, generality or ONE_STANDARD qualification.",
        "publication": "Paused. No source publication, commit, push, deployment or author dispatch is implemented or authorized by this preparation.",
    }


def freeze(phase1, run):
    if run.exists():
        raise ValueError("fresh phase2 directory required")
    manifest, result = verify_phase1(phase1)
    tasks = [task for task in read_json(phase1 / "PANEL.json")["tasks"] if task["task_id"] in manifest["panel"]["compatible_task_ids"]]
    packet, mapping = build_founder_packet(tasks)
    if len(packet["training_cases"]) != 17 or len(tasks) != 6:
        raise ValueError("17-demo/6-task panel required")
    run.mkdir(parents=True)
    write_json(run / "FOUNDER_PACKET.json", packet)
    write_json(run / "EVALUATOR_ONLY.json", {"case_map": mapping, "coverage": manifest["panel"],
               "phase1": str(phase1.resolve()), "baseline_result_sha256": BASELINE_SHA,
               "replay_result_sha256": REPLAY_SHA, "selected_baseline": result["selected_baseline"]})
    write_once(run / "source" / SOURCE_NAME, Path(__file__).read_bytes())
    write_json(run / "INITIAL_STATUS.json", {"parent_author_approval": False, "route": "pending_complete_event_audit_and_input_isolation",
        "quota_attestation": "pending", "reservations": 0, "consumed_attempts": 0, "author_charged_positions": 0,
        "paired_random_charged_positions": 0, "second_random_constructed": False, "predictions_executed": False})
    write_json(run / "STANDARD.json", {"arms": {"llm": {}, "random": {}}, "budget_per_arm": 0,
        "founders": [], "genomes": [], "niches": [], "proposals": [], "seeds": [2026100801, 2026100802],
        "seeds_declared_before_run": True, "scientific_acceptance": False,
        "limitations": ["Unapproved data-only packet/intake preparation, zero authors. Broader qualification remains red/incomplete."]})
    contract = {"schema": "nca-author-intake-contract/1", "parent_author_approval": False,
        "phase1_manifest_sha256": PHASE1_SHA, "phase1_runner_sha256": RUNNER_SHA,
        "phase1_path": str(phase1.resolve()), "protocol": protocol(), "limits": manifest["resource_limits"],
        "artifacts": {name: file_hash(run / name) for name in ("FOUNDER_PACKET.json", "EVALUATOR_ONLY.json", "INITIAL_STATUS.json", "STANDARD.json", "source/" + SOURCE_NAME)},
        "frozen_before_any_author": True}
    write_json(run / "CONTRACT.json", contract)
    expected = file_hash(run / "CONTRACT.json")
    write_once(run / "CONTRACT.sha256", (expected + "\n").encode())
    return expected


def verify(run, expected):
    if os.path.lexists(run / "STOP"):
        raise ValueError("phase2 STOP present")
    if file_hash(run / "CONTRACT.json") != expected or (run / "CONTRACT.sha256").read_text().strip() != expected:
        raise ValueError("contract hash mismatch")
    contract = read_json(run / "CONTRACT.json")
    if contract["schema"] != "nca-author-intake-contract/1" or contract["parent_author_approval"] is not False or contract["protocol"] != protocol():
        raise ValueError("frozen protocol drift")
    for name, sha in contract["artifacts"].items():
        if file_hash(bound_path(run, name)) != sha:
            raise ValueError("phase2 artifact binding failure: " + name)
    if file_hash(__file__) != contract["artifacts"]["source/" + SOURCE_NAME]:
        raise ValueError("intake source binding failure")
    manifest, baseline = verify_phase1(Path(contract["phase1_path"]))
    if contract["limits"] != manifest["resource_limits"]:
        raise ValueError("resource limit drift")
    return contract, manifest, baseline


def attempt_dir(run, number):
    if type(number) is not int or number not in (1, 2):
        raise ValueError("only two finite attempts permitted")
    return run / "attempts" / f"{number:02d}"


def existing_attempts(run):
    root = run / "attempts"
    if not root.exists():
        return []
    names = sorted(path.name for path in root.iterdir())
    if names not in ([], ["01"], ["01", "02"]):
        raise ValueError("unexpected attempt roster")
    return [root / name for name in names]


def require_open_tranche(run):
    if os.path.lexists(run / "HALT.json"):
        raise ValueError("tranche halted; no retry")
    for directory in existing_attempts(run):
        if not (directory / "INTAKE.json").exists():
            raise ValueError("unresolved/reserved/submitted-unknown attempt stops all future work")
        if read_json(directory / "INTAKE.json")["status"] != "valid":
            raise ValueError("invalid/refused attempt stops authoring")


def verify_authorization(approval, expected, packet_sha):
    keys = {"schema", "parent_approved", "parent_identity", "contract_sha256", "packet_sha256", "approval_reference", "route", "quota"}
    if type(approval) is not dict or set(approval) != keys or approval["schema"] != "nca-parent-approval/1" or approval["parent_approved"] is not True:
        raise ValueError("exact external parent approval required")
    if approval["contract_sha256"] != expected or approval["packet_sha256"] != packet_sha:
        raise ValueError("approval packet/contract mismatch")
    for key in ("parent_identity", "approval_reference"):
        if not isinstance(approval[key], str) or not approval[key].strip():
            raise ValueError("parent identity/reference missing")
    route = approval["route"]
    if (set(route) != {"identity", "attestation_reference", "audit_mode", "input_isolation_status", "preventive_isolation_status"} or
            route["audit_mode"] != "complete_event_stream_conservative_allowlist" or
            route["input_isolation_status"] != "verified_packet_only_no_repo_global_context" or
            route["preventive_isolation_status"] != "not_claimed" or
            not all(isinstance(v, str) and v.strip() for v in route.values())):
        raise ValueError("verified complete-event audit and packet-only route required, not declared tools")
    quota = approval["quota"]
    if (set(quota) != {"attestation_reference", "approved_max_attempts", "approved_positions_each_arm", "credit_accounting"} or
            type(quota["approved_max_attempts"]) is not int or quota["approved_max_attempts"] != 2 or
            type(quota["approved_positions_each_arm"]) is not int or quota["approved_positions_each_arm"] != 2976 or
            not all(isinstance(quota[k], str) and quota[k].strip() for k in ("attestation_reference", "credit_accounting"))):
        raise ValueError("bounded quota/credit attestation required")


def packet_path(run, number):
    return run / ("FOUNDER_PACKET.json" if number == 1 else "MUTATION_PACKET.json")


def reserve_attempt(run, expected, number, approval):
    verify(run, expected)
    require_open_tranche(run)
    if len(existing_attempts(run)) != number - 1:
        raise ValueError("attempt order/retry prohibited")
    if number == 2:
        verify_mutation_packet(run, expected)
    packet_sha = file_hash(packet_path(run, number))
    verify_authorization(approval, expected, packet_sha)
    directory = attempt_dir(run, number)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "APPROVAL.json", approval)
    reservation = {"schema": "nca-host-reservation/1", "attempt": number, "role": "founder" if number == 1 else "mutation",
        "contract_sha256": expected, "packet_sha256": packet_sha, "parent_identity": approval["parent_identity"],
        "approval_sha256": file_hash(directory / "APPROVAL.json"), "approval_reference": approval["approval_reference"],
        "route_attestation_reference": approval["route"]["attestation_reference"], "quota_attestation_reference": approval["quota"]["attestation_reference"],
        "reserved_author_positions": POSITIONS, "reserved_paired_random_positions": POSITIONS,
        "parent_tensor_sha256": None if number == 1 else verified_intake(run, expected, 1)["tensor_sha256"],
        "submitted": False, "dispatch_implemented": False}
    write_json(directory / "RESERVATION.json", reservation)
    return reservation


def verify_reservation(run, expected, number):
    verify(run, expected)
    directory = attempt_dir(run, number)
    reservation = read_json(directory / "RESERVATION.json")
    if (reservation["contract_sha256"] != expected or reservation["attempt"] != number or
            reservation["packet_sha256"] != file_hash(packet_path(run, number)) or
            reservation["approval_sha256"] != file_hash(directory / "APPROVAL.json")):
        raise ValueError("reservation/packet/approval binding failure")
    verify_authorization(read_json(directory / "APPROVAL.json"), expected, reservation["packet_sha256"])
    if number == 2:
        verify_mutation_packet(run, expected)
    return reservation


def mark_submission_intent(run, expected, number, host_reference):
    """Host must call BEFORE possible external dispatch; no dispatch happens here."""
    if os.path.lexists(run / "HALT.json"):
        raise ValueError("tranche halted")
    reservation = verify_reservation(run, expected, number)
    if not isinstance(host_reference, str) or not host_reference.strip():
        raise ValueError("durable host dispatch reference required")
    receipt = {"schema": "nca-submission-intent/1", "reservation_sha256": digest(reservation),
        "packet_sha256": reservation["packet_sha256"], "host_reference": host_reference,
        "status": "submitted_unknown_until_full_host_receipt", "consumed_author_positions": POSITIONS,
        "consumed_paired_random_positions": POSITIONS}
    write_json(attempt_dir(run, number) / "SUBMISSION.json", receipt)
    return receipt


def mark_uncertain(run, expected, number, reason):
    verify_reservation(run, expected, number)
    directory = attempt_dir(run, number)
    submission = read_json(directory / "SUBMISSION.json")
    write_json(run / "HALT.json", {"reason": "submission_unknown_or_uncertain", "attempt": number,
        "detail": str(reason), "submission_sha256": digest(submission), "no_retry": True,
        "consumed_author_positions": POSITIONS, "consumed_paired_random_positions": POSITIONS})


def validate_host_audit(audit, reservation, submission, raw_sha):
    keys = {"schema", "status", "host_capture", "parent_identity", "packet_sha256", "raw_sha256", "submission_sha256", "route_attestation_reference", "quota_attestation_reference", "tools_audit_status", "forbidden_tool_uses", "audit_reference", "event_stream_status", "event_metadata", "restricted_trace_sha256", "restricted_trace_reference", "usage", "input_isolation_status", "preventive_isolation_status", "timed_out", "completion_certain"}
    if type(audit) is not dict or set(audit) != keys or audit["schema"] != "nca-host-response-audit/1":
        raise ValueError("complete host response audit required")
    if audit["status"] not in ("completed", "refused") or audit["host_capture"] is not True:
        raise ValueError("uncertain or non-host response capture")
    for key in ("parent_identity", "packet_sha256", "route_attestation_reference", "quota_attestation_reference"):
        if audit[key] != reservation[key]:
            raise ValueError("host audit identity/binding mismatch")
    if audit["raw_sha256"] != raw_sha or audit["submission_sha256"] != digest(submission):
        raise ValueError("host audit response/submission binding mismatch")
    if (audit["tools_audit_status"] != "observed_zero_tool_use" or type(audit["forbidden_tool_uses"]) is not list or
            audit["forbidden_tool_uses"] or not isinstance(audit["audit_reference"], str) or not audit["audit_reference"].strip()):
        raise ValueError("forbidden tool use or unverified tool audit")
    if (audit["event_stream_status"] != "complete" or audit["timed_out"] is not False or audit["completion_certain"] is not True or
            audit["input_isolation_status"] != "verified_packet_only_no_repo_global_context" or audit["preventive_isolation_status"] != "not_claimed"):
        raise ValueError("incomplete/uncertain event capture or unverified input isolation")
    def sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if not sha(audit["restricted_trace_sha256"]) or not isinstance(audit["restricted_trace_reference"], str) or not audit["restricted_trace_reference"].strip():
        raise ValueError("restricted parent event-trace hash/reference required")
    metadata = audit["event_metadata"]
    if type(metadata) is not list or len(metadata) < 4:
        raise ValueError("complete classified event metadata required")
    lifecycle = {"thread.started", "turn.started", "turn.completed"}
    final_count = 0
    for index, event in enumerate(metadata):
        if (type(event) is not dict or set(event) != {"sequence", "event_type", "item_type", "classification", "event_sha256", "final_raw_sha256"} or
                type(event["sequence"]) is not int or event["sequence"] != index or not sha(event["event_sha256"])):
            raise ValueError("event gap/truncation or malformed metadata")
        if event["event_type"] in lifecycle:
            valid = event["item_type"] is None and event["classification"] == "lifecycle" and event["final_raw_sha256"] is None
        elif event["event_type"] in {"item.started", "item.updated", "item.completed"} and event["item_type"] == "reasoning":
            valid = event["classification"] == "reasoning_summary" and event["final_raw_sha256"] is None
        elif event["event_type"] == "item.completed" and event["item_type"] == "agent_message":
            valid = event["classification"] == "final_message" and event["final_raw_sha256"] == raw_sha
            final_count += 1
        else:
            valid = False
        if not valid:
            raise ValueError("tool attempt or unknown execution-bearing event")
    if ([event["event_type"] for event in metadata].count("thread.started") != 1 or metadata[0]["event_type"] != "thread.started" or
            [event["event_type"] for event in metadata].count("turn.started") != 1 or metadata[1]["event_type"] != "turn.started" or
            [event["event_type"] for event in metadata].count("turn.completed") != 1 or metadata[-1]["event_type"] != "turn.completed" or final_count != 1):
        raise ValueError("incomplete or repeated invocation/final lifecycle")
    usage = audit["usage"]
    if type(usage) is not dict or set(usage) != {"input_tokens", "cached_input_tokens", "output_tokens", "credits"}:
        raise ValueError("usage accounting required")
    if any(type(usage[key]) is not int or usage[key] < 0 for key in ("input_tokens", "cached_input_tokens", "output_tokens")) or usage["credits"] != "unknown":
        raise ValueError("invalid token accounting; credits remain unknown")


def capture_response(run, expected, number, raw, host_audit):
    if os.path.lexists(run / "HALT.json"):
        raise ValueError("tranche halted")
    reservation = verify_reservation(run, expected, number)
    directory = attempt_dir(run, number)
    submission = read_json(directory / "SUBMISSION.json")
    if submission["reservation_sha256"] != digest(reservation):
        raise ValueError("submission/reservation binding failure")
    # Preserve complete raw bytes even when invalid/oversize/non-UTF8/refused.
    write_once(directory / "RAW_RESPONSE.bin", raw)
    raw_sha = file_hash(directory / "RAW_RESPONSE.bin")
    error, value, status, audit_valid = None, None, "invalid", False
    try:
        validate_host_audit(host_audit, reservation, submission, raw_sha)
        audit_valid = True
        if host_audit["status"] == "refused":
            status = "refused"
        else:
            value = validate_response(raw)
            status = "valid"
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    # Only an allowlisted audit may enter the publishable evidence directory.
    # In particular, never persist unexpected trace/reasoning text from a bad audit.
    if audit_valid:
        published_audit = host_audit
    else:
        try:
            supplied_hash = digest(host_audit)
        except (ValueError, TypeError, OverflowError, RecursionError):
            supplied_hash = None
        published_audit = {"schema": "nca-rejected-host-audit/1", "supplied_audit_sha256": supplied_hash,
                           "status": "rejected", "error": error, "raw_trace_retained_only_by_parent": True}
    write_json(directory / "HOST_AUDIT.json", published_audit)
    receipt = {"schema": "nca-strict-intake/1", "attempt": number, "status": status, "error": error,
        "contract_sha256": expected, "packet_sha256": reservation["packet_sha256"], "raw_sha256": raw_sha,
        "reservation_sha256": digest(reservation), "submission_sha256": digest(submission),
        "host_audit_sha256": file_hash(directory / "HOST_AUDIT.json"), "parent_identity": reservation["parent_identity"],
        "route_attestation_reference": reservation["route_attestation_reference"], "quota_attestation_reference": reservation["quota_attestation_reference"],
        "tools_audit_status": "observed_zero_tool_use" if audit_valid else "rejected_or_missing",
        "tensor_sha256": digest(value["tensors"]) if value else None,
        "numeric_fingerprint": numeric_fingerprint(value["tensors"]) if value else None,
        "parent_tensor_sha256": reservation["parent_tensor_sha256"],
        "author_positions": POSITIONS, "paired_random_positions": POSITIONS}
    write_json(directory / "INTAKE.json", receipt)
    write_once(directory / "INTAKE.sha256", (file_hash(directory / "INTAKE.json") + "\n").encode())
    if not audit_valid:
        write_json(run / "HALT.json", {"reason": "route_event_audit_failed", "attempt": number,
            "intake_sha256": file_hash(directory / "INTAKE.json"), "no_retry_or_claim": True,
            "consumed_author_positions": POSITIONS, "consumed_paired_random_positions": POSITIONS})
    return receipt


def verified_intake(run, expected, number):
    reservation = verify_reservation(run, expected, number)
    directory = attempt_dir(run, number)
    receipt = read_json(directory / "INTAKE.json")
    if file_hash(directory / "INTAKE.json") != (directory / "INTAKE.sha256").read_text().strip():
        raise ValueError("intake receipt tampered")
    submission = read_json(directory / "SUBMISSION.json")
    if (receipt["raw_sha256"] != file_hash(directory / "RAW_RESPONSE.bin") or
            receipt["host_audit_sha256"] != file_hash(directory / "HOST_AUDIT.json") or
            receipt["reservation_sha256"] != digest(reservation) or receipt["submission_sha256"] != digest(submission) or
            submission["reservation_sha256"] != digest(reservation)):
        raise ValueError("response/audit/submission binding failure")
    if receipt["status"] == "valid":
        validate_host_audit(read_json(directory / "HOST_AUDIT.json"), reservation, submission, receipt["raw_sha256"])
        value = validate_response((directory / "RAW_RESPONSE.bin").read_bytes())
        if digest(value["tensors"]) != receipt["tensor_sha256"] or numeric_fingerprint(value["tensors"]) != receipt["numeric_fingerprint"]:
            raise ValueError("response tensor substitution")
    return receipt


def accounting(run):
    directories = existing_attempts(run)
    consumed = sum((directory / "SUBMISSION.json").exists() for directory in directories)
    return {"reservations": len(directories), "consumed_attempts": consumed,
            "author_charged_positions": consumed * POSITIONS, "paired_random_charged_positions": consumed * POSITIONS}


def frozen_runtime(phase1):
    verify_phase1(phase1)
    path = phase1 / "source/scripts/nca_feasibility_preflight.py"
    spec = importlib.util.spec_from_file_location("_frozen_nca_phase1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest, tasks = module.verify_freeze(phase1, PHASE1_SHA)
    return module, manifest, tasks


def inject_literals(module, phase1, manifest, response):
    """Instantiate exact saved descriptor, never mutable builder/T2/evolution caller."""
    np, _, Graph, Limits, array_hash, _, _ = module.runtime_imports()
    descriptor = read_json(phase1 / "GRAPH_STRUCTURE.json")
    if digest(descriptor) != manifest["graph"]["structure_sha256"]:
        raise ValueError("graph descriptor binding failure")
    validated = validate_response(canonical(response))
    params = {name: np.asarray(values, dtype=np.float64) for name, values in validated["tensors"].items()}
    graph = Graph(copy.deepcopy(descriptor["defs"]), params, copy.deepcopy(descriptor["entries"]), copy.deepcopy(descriptor["meta"]))
    if module.graph_structure(graph) != descriptor:
        raise ValueError("author structure/schema substitution")
    # Every literal is checked after construction, including signed zero bits.
    for name, values in validated["tensors"].items():
        expected = np.asarray(values, dtype=np.float64)
        if graph.params[name].tobytes() != expected.tobytes():
            raise ValueError("author literal substitution")
    graph.validate(Limits(strict_microcells=True))
    return graph, {name: array_hash(array) for name, array in sorted(graph.params.items())}


def paired_second_literals(run, expected):
    verified_intake(run, expected, 2)
    # A consumed second attempt is necessary even when its response is invalid.
    rng = random.Random(2026100802)
    tensors = {}
    for item in tensor_schema():
        values = [rng.choice(VOCABULARY) for _ in range(item["positions"])]
        tensors[item["name"]] = [values] if len(item["shape"]) == 2 else values
    return {"schema": "nca-author-tensors/1", "tensors": tensors}


def worker(run, expected, number, kind, out):
    contract, _, _ = verify(run, expected)
    if os.path.lexists(run / "HALT.json"):
        raise ValueError("unknown/uncertain submission stops evaluation")
    receipt = verified_intake(run, expected, number)
    if out.resolve() != (attempt_dir(run, number) / "evaluation" / (kind + ".progress.jsonl")).resolve():
        raise ValueError("worker output must be in its phase2 attempt evaluation directory")
    phase1 = Path(contract["phase1_path"])
    module, manifest, tasks = frozen_runtime(phase1)
    module.apply_limits()
    if kind == "candidate":
        if receipt["status"] != "valid":
            raise ValueError("invalid author cannot execute")
        response = validate_response((attempt_dir(run, number) / "RAW_RESPONSE.bin").read_bytes())
    elif kind == "paired_random_2" and number == 2:
        response = paired_second_literals(run, expected)
        write_json(out.parent / "paired_random_2.tensors.json", response)
    else:
        raise ValueError("only candidate or consumed second paired random worker allowed")
    graph, hashes = inject_literals(module, phase1, manifest, response)
    write_json(out.parent / (kind + ".injection.json"), {"graph_sha256": graph.hash(), "tensor_sha256": digest(response["tensors"]), "per_tensor_sha256": hashes,
        "phase1_structure_sha256": manifest["graph"]["structure_sha256"], "all_1488_literals_injected_unchanged": True})
    *_, Predictor, _ = module.runtime_imports()
    predictor = Predictor(graph, backend="reference", cache_demo_state=False)
    def timed_out(signum, frame):
        raise TimeoutError("per-demo wall limit exceeded")
    signal.signal(signal.SIGALRM, timed_out)
    with out.open("x") as handle:
        for task in tasks:
            for index in range(len(task["train"])):
                if os.path.lexists(run / "STOP") or os.path.lexists(phase1 / "STOP") or os.path.lexists(run / "HALT.json"):
                    raise ValueError("STOP/HALT present")
                signal.setitimer(signal.ITIMER_REAL, contract["limits"]["demo_wall_seconds"])
                try:
                    row = module.evaluate_demo(predictor, task, index)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                handle.write(canonical(row).decode() + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def score_worker(run, expected, number, kind, output_dir, contract, module, tasks):
    progress = output_dir / (kind + ".progress.jsonl")
    phase1 = Path(contract["phase1_path"])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    env.update({key: "1" for key in THREAD_KEYS})
    env["PYTHONPATH"] = str(phase1 / "source/src") + os.pathsep + str(phase1 / "source/scripts")
    cmd = [sys.executable, str(run / "source" / SOURCE_NAME), "_worker", "--run", str(run), "--contract-sha256", expected,
           "--attempt", str(number), "--kind", kind, "--out", str(progress)]
    error, code, stdout, stderr = None, None, b"", b""
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=run, env=env, timeout=contract["limits"]["arm_wall_seconds"], capture_output=True, check=False)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
        if code:
            error = {"type": "SubprocessError", "message": "worker exit " + str(code)}
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
        error = {"type": "ArmTimeout", "message": "arm wall limit exceeded"}
    write_once(output_dir / (kind + ".stdout.txt"), stdout)
    write_once(output_dir / (kind + ".stderr.txt"), stderr)
    rows = []
    if progress.exists():
        for line in progress.read_bytes().splitlines():
            try:
                rows.append(strict_json(line))
            except (ValueError, UnicodeError):
                error = {"type": "IncompleteProgress", "message": "partial worker row"}
                break
    keys = [(task["task_id"], index) for task in tasks for index in range(len(task["train"]))]
    if [(row.get("task_id"), row.get("demo_index")) for row in rows] != keys[:len(rows)]:
        raise ValueError("worker row coverage/order mismatch")
    if len(rows) != len(keys) and error is None:
        error = {"type": "IncompleteArm", "message": "missing worker rows"}
    rows.extend(module.error_row(task_id, index, error) for task_id, index in keys[len(rows):])
    summary = module.summarize(rows, tasks)
    if error:
        summary.update(valid=False, exact_demonstrations=None, exact_tasks=None, mean_normalized_pixel_error=None)
    result = {"arm": kind, "summary": summary, "demonstrations": rows, "process_error": error, "returncode": code,
              "process_elapsed_seconds": time.perf_counter() - start}
    write_json(output_dir / (kind + ".json"), result)
    return result


def compare(candidate, references, randoms, selected, founder=None):
    arms = [candidate, *references, *randoms] + ([founder] if founder is not None else [])
    if not all(arm["summary"]["valid"] for arm in arms):
        return {"comparable": False, "positive_feasibility": None, "positive_mutation": None, "reason": "invalid_arm_null_outcome"}
    summary = candidate["summary"]
    solved = set(map(tuple, summary["solved_demo_keys"]))
    anchors = set(map(tuple, selected["summary"]["solved_demo_keys"]))
    positive = summary["exact_demonstrations"] > max(arm["summary"]["exact_demonstrations"] for arm in [*references, *randoms]) and anchors <= solved
    mutation = None if founder is None else (positive and summary["exact_demonstrations"] > founder["summary"]["exact_demonstrations"] and set(map(tuple, founder["summary"]["solved_demo_keys"])) <= solved)
    return {"comparable": True, "positive_feasibility": positive, "positive_mutation": mutation,
            "selected_baseline_anchors_preserved": anchors <= solved}


def evaluate_attempt(run, expected, number):
    """Future authorized intake only; never invoked on real phase2 during preparation."""
    contract, manifest, baseline = verify(run, expected)
    if os.path.lexists(run / "HALT.json"):
        raise ValueError("unknown submission stops all future work")
    receipt = verified_intake(run, expected, number)
    phase1 = Path(contract["phase1_path"])
    module, _, tasks = frozen_runtime(phase1)
    output_dir = attempt_dir(run, number) / "evaluation"
    output_dir.mkdir(exist_ok=False)
    references = [read_json(phase1 / "baseline" / (name + ".json")) for name in module.REFERENCES]
    randoms = [read_json(phase1 / "baseline/paired_random_1.json")]
    if number == 2:
        randoms.append(score_worker(run, expected, number, "paired_random_2", output_dir, contract, module, tasks))
    if receipt["status"] == "valid":
        candidate = score_worker(run, expected, number, "candidate", output_dir, contract, module, tasks)
    else:
        rows = [module.error_row(task["task_id"], index, {"type": "InvalidSubmission", "message": receipt["status"]}) for task in tasks for index in range(len(task["train"]))]
        candidate = {"arm": "candidate", "summary": module.summarize(rows, tasks), "demonstrations": rows, "process_error": None, "returncode": None}
        write_json(output_dir / "candidate.json", candidate)
    selected = next(arm for arm in references if arm["arm"] == baseline["selected_baseline"])
    founder = dict(verified_evaluation(run, expected, 1)["candidate"], arm="founder") if number == 2 else None
    controls_valid = all(arm["summary"]["valid"] for arm in [*references, *randoms])
    numeric_comparators = {spec["arm"]: numeric_fingerprint(read_json(phase1 / spec["tensor_file"])["tensors"]) for spec in manifest["arms"]}
    if founder:
        numeric_comparators["founder"] = verified_intake(run, expected, 1)["numeric_fingerprint"]
        if (output_dir / "paired_random_2.tensors.json").exists():
            numeric_comparators["paired_random_2"] = numeric_fingerprint(read_json(output_dir / "paired_random_2.tensors.json")["tensors"])
    result = {"schema": "nca-author-evaluation/1", "attempt": number, "intake_sha256": file_hash(attempt_dir(run, number) / "INTAKE.json"),
        "candidate": candidate, "comparison": compare(candidate, references, randoms, selected, founder), "controls_valid": controls_valid,
        "phase1_random_reused_tensor_sha256": next(spec["tensor_sha256"] for spec in manifest["arms"] if spec["arm"] == "paired_random_1"),
        "phase1_random_reused_result_sha256": file_hash(phase1 / "baseline/paired_random_1.json"),
        "coverage": manifest["panel"], "accounting": accounting(run),
        "numeric_duplicates": [name for name, sha in numeric_comparators.items() if receipt["numeric_fingerprint"] is not None and sha == receipt["numeric_fingerprint"]],
        "functional_duplicates": [arm["arm"] for arm in [*references, *randoms] + ([founder] if founder else []) if candidate["summary"]["valid"] and arm["summary"]["valid"] and [r["prediction"] for r in candidate["demonstrations"]] == [r["prediction"] for r in arm["demonstrations"]]],
        "artifact_sha256": {path.name: file_hash(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
        "qualification": "Exposed training feasibility only. No admission or ONE_STANDARD qualification."}
    write_json(attempt_dir(run, number) / "EVALUATION.json", result)
    write_once(attempt_dir(run, number) / "EVALUATION.sha256", (file_hash(attempt_dir(run, number) / "EVALUATION.json") + "\n").encode())
    return result


def verified_evaluation(run, expected, number):
    verified_intake(run, expected, number)
    directory = attempt_dir(run, number)
    if file_hash(directory / "EVALUATION.json") != (directory / "EVALUATION.sha256").read_text().strip():
        raise ValueError("evaluation receipt tampered")
    result = read_json(directory / "EVALUATION.json")
    if result["intake_sha256"] != file_hash(directory / "INTAKE.json"):
        raise ValueError("evaluation/intake binding failure")
    for name, sha in result["artifact_sha256"].items():
        if file_hash(bound_path(directory / "evaluation", name)) != sha:
            raise ValueError("evaluation artifact tampered")
    return result


def mutation_material(run, expected):
    if os.path.lexists(run / "HALT.json"):
        raise ValueError("tranche halted")
    founder = verified_intake(run, expected, 1)
    result = verified_evaluation(run, expected, 1)
    candidate = result["candidate"]
    if founder["status"] != "valid" or not result["controls_valid"] or not candidate["summary"]["valid"] or candidate["summary"]["exact_demonstrations"] >= 17:
        raise ValueError("founder not eligible for one conditional mutation")
    mapping = {(row["task_id"], row["demo_index"]): row["case"] for row in read_json(run / "EVALUATOR_ONLY.json")["case_map"]}
    failures = [{"case": mapping[(row["task_id"], row["demo_index"])], "prediction": row["prediction"], "pixel_errors": row["pixel_errors"], "normalized_pixel_error": row["normalized_pixel_error"]}
                for row in candidate["demonstrations"] if row["exact"] is False]
    return {"schema": "nca-data-only-mutation-packet/1", "frozen_founder_context": read_json(run / "FOUNDER_PACKET.json"),
            "parent_tensor_sha256": founder["tensor_sha256"], "parent_tensors": validate_response((attempt_dir(run, 1) / "RAW_RESPONSE.bin").read_bytes())["tensors"],
            "deterministic_training_failures": failures}


def prepare_mutation_packet(run, expected):
    verify(run, expected)
    if len(existing_attempts(run)) != 1:
        raise ValueError("one completed founder required")
    value = mutation_material(run, expected)
    write_json(run / "MUTATION_PACKET.json", value)
    write_once(run / "MUTATION_PACKET.sha256", (file_hash(run / "MUTATION_PACKET.json") + "\n").encode())
    return file_hash(run / "MUTATION_PACKET.json")


def verify_mutation_packet(run, expected):
    if file_hash(run / "MUTATION_PACKET.json") != (run / "MUTATION_PACKET.sha256").read_text().strip():
        raise ValueError("mutation packet tampered")
    # Compare to founder data deterministically; no scores from controls exported.
    if read_json(run / "MUTATION_PACKET.json") != mutation_material(run, expected):
        raise ValueError("mutation parent/failures/context binding failure")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "_worker"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase1", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--attempt", type=int, choices=(1, 2))
    parser.add_argument("--kind", choices=("candidate", "paired_random_2"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if args.command == "freeze":
        if args.phase1 is None:
            parser.error("phase1 required")
        result = {"contract_sha256": freeze(args.phase1.resolve(), run), "author_dispatch_authorized": False}
    else:
        if not args.contract_sha256:
            parser.error("explicit reviewed contract SHA256 required")
        if args.command == "verify":
            verify(run, args.contract_sha256)
            result = {"verified": True, "accounting": accounting(run), "author_dispatch_authorized": False}
        else:
            if args.attempt is None or args.kind is None or args.out is None:
                parser.error("worker needs attempt/kind/out")
            worker(run, args.contract_sha256, args.attempt, args.kind, args.out)
            result = {"worker_complete": True}
    print(canonical(result).decode())


if __name__ == "__main__":
    main()
