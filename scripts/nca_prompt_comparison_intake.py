"""Separate finite A/B founder adapter. No model dispatch is implemented.

Production operations require new exact parent approvals and a reviewed adapter
supplement. Preparation/tests never initialize a real execution ledger. The old
mutation reservation path is not used. The frozen preparation remains unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

PREPARATION_SHA = "f4f89d4eaeec4adbd50f4f8bd42c5e35ad56cf535396032f36f516a5b93a09f4"
PREPARER_SHA = "928851e2a30594a12b1f9abb28f744ed8f347005fffac4f5faa92c15b9092f92"
INTAKE_SHA = "39ce4aa1798549854299002e02b3a74f12c4c824851686e0bf6c14f001deb0f6"
SOURCE = "scripts/nca_prompt_comparison_intake.py"
TESTS = "tests/test_nca_prompt_comparison_intake.py"
FRESH_SECONDS = 300
NATIVE_COUNTERS = {"input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "reasoning_output_tokens"}
CAPTURE_SHA = "5d822fff874cbd84856474ed275912a43936da3af881dff63c33e1d89539c43b"
CONFIG_SHAS = {"A": "720dd9f88aa0af7022ea9235a83886bbadaf69d0db7a04dc2d1a9d390c83ceaf",
               "B": "77c49e86ebb18fa7437ef50aaf5e3289c97082ccc753c0b7d15ecd6c49f065c8"}


def raw_hash(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return raw_hash(canonical(value))


def load_module(path, expected, name):
    if raw_hash(path.read_bytes()) != expected:
        raise ValueError("bound module hash mismatch: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preparer(repo):
    return load_module(Path(repo) / "scripts/nca_prompt_comparison_prepare.py", PREPARER_SHA, "_comparison_preparer")


def supplement_payload(repo):
    repo = Path(repo).resolve()
    p = preparer(repo)
    p.verify(repo, repo / p.RUN_NAME, PREPARATION_SHA)
    route_bindings = {}
    for name, expected in [("prospective_native_capture_v5.py", CAPTURE_SHA)] + [("PROSPECTIVE_V5_COMPARISON_" + arm + ".json", CONFIG_SHAS[arm]) for arm in ("A", "B")]:
        path = repo.parent / "cellforge_route_review" / name
        raw = path.read_bytes()
        if raw_hash(raw) != expected:
            raise ValueError("actual reviewed V5 source/config hash mismatch")
        route_bindings[name] = {"sha256": expected, "bytes": len(raw)}
    return {"schema": "nca-prompt-comparison-adapter-supplement/1", "parent_execution_approval": False,
            "real_reservations_at_freeze": 0, "model_dispatch_implemented": False,
            "preparation_contract_sha256": PREPARATION_SHA, "preparation_path": p.RUN_NAME,
            "source_sha256": {name: raw_hash((repo / name).read_bytes()) for name in (SOURCE, TESTS)},
            "reused_intake_sha256": INTAKE_SHA, "preparer_sha256": PREPARER_SHA,
            "actual_V5_route_bindings": route_bindings,
            "native_audit": "Reuse exact reviewed V5 full-native validator and existing HOST_AUDIT contract. Adapter validates original V5 result/native-usage receipt bytes and line-hash correspondence, not a duplicate native trace parser. Parent retains full trace and verifies packet/schema/command/receipt identity.",
            "fresh_attestation_max_age_seconds": FRESH_SECONDS,
            "random_implementation": "Python random.Random(seed).choice over frozen seven-value vocabulary, sorted tensor names, C/row-major; full output bytes retained. Python runtime version is ledger-bound.",
            "execution_directory": "A NEW directory outside every historical/preparation run; initialize only with future parent execution approval.",
            "crash_policy": "Partial ledger or artifact transactions fail closed. Consume once before possible external dispatch. Never relaunch, retry, refund or replace. Recover original host receipts only; uncertain completion terminally stops.",
            "bound_helpers": ["validate_response", "validate_host_audit", "numeric_fingerprint", "frozen_runtime", "inject_literals", "compare"],
            "limits": "Author: gpt-6.1-sol/low, 120 s, 2 GiB, concurrency 1. Evaluation: unchanged frozen phase1 limits.",
            "required_review": "Independent source/test review and exact new packet-specific parent approval before every real reservation. This supplement is not approval."}


def freeze_supplement(repo, path):
    p = preparer(repo)
    value = supplement_payload(repo)
    p.write_once(Path(path), p.json_bytes(value))
    return raw_hash(Path(path).read_bytes())


def now():
    return time.time()


def timestamp(value):
    if type(value) is not str:
        raise ValueError("timezone-bearing timestamp required")
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("timezone-bearing timestamp required")
    return date.timestamp()


def nonempty(value):
    return type(value) is str and bool(value.strip())


def sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def quota_check(value, clock, previous=None):
    keys = {"authoritative", "ordinaryUsageAllowed", "usage_percent", "period_id", "reset_at", "observed_at", "attestation_reference"}
    if type(value) is not dict or set(value) != keys or value["authoritative"] is not True or value["ordinaryUsageAllowed"] is not True:
        raise ValueError("fresh authoritative ordinaryUsageAllowed=true required")
    percent = value["usage_percent"]
    if type(percent) not in (int, float) or not math.isfinite(percent) or not 0 <= percent < 85:
        raise ValueError("current shared quota must be below 85% STOP")
    if not nonempty(value["period_id"]) or not nonempty(value["attestation_reference"]):
        raise ValueError("quota period/reference required")
    if not 0 <= clock - timestamp(value["observed_at"]) <= FRESH_SECONDS or timestamp(value["reset_at"]) <= clock:
        raise ValueError("stale/future quota snapshot or reset")
    if previous and (value["period_id"], value["reset_at"]) != (previous["period_id"], previous["reset_at"]):
        raise ValueError("quota period/reset drift blocks this frozen comparison")


def route_check(value, packet_sha, schema_sha, clock):
    constants = {"model": "gpt-6.1-sol", "reasoning_effort": "low", "wall_seconds": 120,
                 "memory_bytes": 2147483648, "concurrency": 1, "request_retries": 0, "stream_retries": 0,
                 "websocket": False, "audit_mode": "complete_event_stream_conservative_allowlist",
                 "input_isolation_status": "verified_packet_only_no_repo_global_context",
                 "preventive_isolation_status": "not_claimed", "new_independent_session": True,
                 "prior_weights_exposed": False, "controls_exposed": False, "cross_arm_feedback_exposed": False,
                 "scientific_input_sha256": [packet_sha, schema_sha]}
    extra = {"identity", "attestation_reference", "isolated_role_directory", "observed_at", "capture_code_sha256", "capture_config_sha256"}
    if type(value) is not dict or set(value) != set(constants) | extra:
        raise ValueError("exact packet-only route snapshot required")
    for key, expected in constants.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError("route/isolation/limit mismatch: " + key)
    if not all(nonempty(value[k]) for k in ("identity", "attestation_reference", "isolated_role_directory")):
        raise ValueError("route identity/reference/directory required")
    if not sha(value["capture_code_sha256"]) or not sha(value["capture_config_sha256"]):
        raise ValueError("reviewed capture implementation/configuration hashes required")
    if not 0 <= clock - timestamp(value["observed_at"]) <= FRESH_SECONDS:
        raise ValueError("stale/future route snapshot")


class Adapter:
    def __init__(self, repo, supplement, supplement_sha, state):
        proposed_repo = Path(repo).absolute()
        if any(part.is_symlink() for part in (proposed_repo, *proposed_repo.parents)):
            raise ValueError("nonsymlink repository path required")
        self.repo = proposed_repo.resolve()
        self.supplement_path = Path(supplement).absolute()
        self.supplement_sha = supplement_sha
        proposed = Path(state).absolute()
        if any(part.is_symlink() for part in (proposed, *proposed.parents)):
            raise ValueError("nonsymlink execution path required")
        self.root = proposed.resolve()
        self.p = preparer(self.repo)
        self.preparation = self.repo / self.p.RUN_NAME
        self.verify_bindings()
        self.intake = load_module(self.repo / "scripts/nca_author_intake.py", INTAKE_SHA, "_comparison_numeric_intake")

    def verify_bindings(self):
        raw = self.supplement_path.read_bytes()
        if not sha(self.supplement_sha) or raw_hash(raw) != self.supplement_sha:
            raise ValueError("external adapter supplement hash mismatch")
        value = self.p.strict_json(raw)
        if raw != self.p.json_bytes(supplement_payload(self.repo)):
            raise ValueError("adapter supplement/source drift")
        if raw_hash(Path(__file__).read_bytes()) != value["source_sha256"][SOURCE]:
            raise ValueError("executing adapter source mismatch")
        if self.root == self.repo or self.root.is_relative_to(self.preparation) or any(part.is_symlink() for part in (self.root, *self.root.parents)):
            raise ValueError("separate nonsymlink execution directory required")
        for old in (self.repo / self.p.HISTORY, self.repo / self.p.PHASE1):
            if self.root == old or self.root.is_relative_to(old):
                raise ValueError("historical run is immutable")
        return value

    def packet_sha(self, arm):
        if arm not in ("A", "B"):
            raise ValueError("only A and B founders allowed")
        return raw_hash((self.preparation / "author" / (arm + "_PACKET.json")).read_bytes())

    def approval_check(self, approval, arm, previous=None):
        keys = {"schema", "parent_approved", "parent_identity", "approval_reference", "arm", "preparation_contract_sha256",
                "adapter_supplement_sha256", "packet_sha256", "response_schema_sha256", "approved_max_invocations",
                "approved_positions_each_arm", "route", "quota"}
        if type(approval) is not dict or set(approval) != keys or approval["schema"] != "nca-comparison-parent-approval/1" or approval["parent_approved"] is not True:
            raise ValueError("exact new packet-specific parent approval required")
        constants = {"arm": arm, "preparation_contract_sha256": PREPARATION_SHA, "adapter_supplement_sha256": self.supplement_sha,
                     "packet_sha256": self.packet_sha(arm), "response_schema_sha256": self.p.SCHEMA_SHA,
                     "approved_max_invocations": 2, "approved_positions_each_arm": 2976}
        if any(type(approval[k]) is not type(v) or approval[k] != v for k, v in constants.items()):
            raise ValueError("approval contract/packet/schema/budget mismatch")
        if not all(nonempty(approval[k]) for k in ("parent_identity", "approval_reference")):
            raise ValueError("parent identity/reference required")
        clock = now()
        route_check(approval["route"], self.packet_sha(arm), self.p.SCHEMA_SHA, clock)
        self.exact_route(approval["route"], arm)
        quota_check(approval["quota"], clock, previous)

    def exact_route(self, route, arm):
        config_path = self.repo.parent / "cellforge_route_review" / ("PROSPECTIVE_V5_COMPARISON_" + arm + ".json")
        config = self.p.strict_json(config_path.read_bytes())
        if route["capture_code_sha256"] != CAPTURE_SHA or route["capture_config_sha256"] != CONFIG_SHAS[arm] or route["isolated_role_directory"] != config["cwd"]:
            raise ValueError("actual reviewed V5 code/config/cwd binding required")

    @contextmanager
    def locked(self):
        self.verify_bindings()
        if not self.root.is_dir():
            raise ValueError("uninitialized execution directory")
        path = self.p.safe_path(self.root, "LOCK")
        with path.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self):
        raw = self.p.safe_path(self.root, "LEDGER.jsonl").read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise ValueError("partial/missing ledger: terminal fail-closed, no retry")
        events, previous, stages = [], "0" * 64, {}
        stopped = False
        for number, line in enumerate(raw.splitlines(keepends=True)):
            event = self.p.strict_json(line)
            if type(event) is not dict or set(event) != {"index", "previous_sha256", "kind", "arm", "payload", "sha256"}:
                raise ValueError("ledger envelope mismatch")
            base = {k: v for k, v in event.items() if k != "sha256"}
            if event["index"] != number or event["previous_sha256"] != previous or event["sha256"] != digest(base) or line != self.p.json_bytes(event):
                raise ValueError("ledger hash chain mismatch")
            kind, arm, payload = event["kind"], event["arm"], event["payload"]
            if stopped:
                raise ValueError("event after terminal STOP")
            if number == 0:
                if kind != "initialize" or arm is not None or payload != self.genesis():
                    raise ValueError("ledger genesis/runtime mismatch")
            elif kind == "reserve":
                if arm not in ("A", "B") or arm in stages or (arm == "B" and stages.get("A") != "evaluated") or (arm == "A" and stages):
                    raise ValueError("duplicate/out-of-order reservation")
                stages[arm] = "reserved"
            elif kind == "submit":
                if stages.get(arm) != "reserved":
                    raise ValueError("duplicate/out-of-order consumption")
                stages[arm] = "submitted"
            elif kind == "capture":
                if stages.get(arm) != "submitted":
                    raise ValueError("duplicate/out-of-order capture")
                stages[arm] = "captured" if payload["status"] == "valid" else "invalid"
            elif kind == "evaluate_begin":
                if stages.get(arm) != "captured":
                    raise ValueError("duplicate/out-of-order evaluation")
                stages[arm] = "evaluating"
            elif kind == "evaluate_end":
                if stages.get(arm) != "evaluating" or payload["valid"] is not True:
                    raise ValueError("invalid evaluation transition")
                stages[arm] = "evaluated"
            elif kind == "stop":
                stopped = True
            else:
                raise ValueError("unknown ledger transition")
            for collection in (payload.get("artifacts", {}), payload.get("restricted_artifacts", {})):
                for name, expected in collection.items():
                    self.p.checked_bytes(self.root, name, expected)
            events.append(event)
            previous = event["sha256"]
        return events, stages, stopped

    def genesis(self):
        return {"schema": "nca-comparison-execution-ledger/1", "preparation_contract_sha256": PREPARATION_SHA,
                "adapter_supplement_sha256": self.supplement_sha, "python_version": sys.version,
                "author_order": ["A", "B"], "positions_per_consumption_each_accounting_arm": 1488}

    def _append(self, events, kind, arm, payload):
        base = {"index": len(events), "previous_sha256": events[-1]["sha256"] if events else "0" * 64,
                "kind": kind, "arm": arm, "payload": payload}
        event = {**base, "sha256": digest(base)}
        with (self.root / "LEDGER.jsonl").open("ab") as handle:
            handle.write(self.p.json_bytes(event))
            handle.flush()
            os.fsync(handle.fileno())
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return event

    def initialize(self, approval):
        self.verify_bindings()
        self.require_open(False)
        self.approval_check(approval, "A")
        if os.path.lexists(self.root):
            raise ValueError("new execution directory required; no initialization retry")
        self.root.mkdir(parents=True, exist_ok=False)
        self.p.write_once(self.root / "INITIAL_APPROVAL.json", self.p.json_bytes(approval))
        self.p.write_once(self.root / "STANDARD.json", self.p.json_bytes({"arms": {"llm": {}, "random": {}},
            "seeds": [2026100803, 2026100804], "seeds_declared_before_run": True, "budget_per_arm": 0,
            "founders": [], "genomes": [], "niches": [], "proposals": [], "scientific_acceptance": False,
            "limitations": ["Initial status only; authoritative consumed charges are in LEDGER.jsonl. Not ONE_STANDARD-qualified."]}))
        self._append([], "initialize", None, self.genesis())
        return self.status()

    def status(self):
        with self.locked():
            events, stages, stopped = self._read()
            consumed = sum(e["kind"] == "submit" for e in events)
            unresolved = [arm for arm, stage in stages.items() if stage in ("reserved", "submitted", "evaluating", "invalid")]
            return {"stages": stages, "terminal_stop": stopped or self.stop_present(), "unresolved_no_retry": unresolved,
                    "reservations": sum(e["kind"] == "reserve" for e in events), "consumed_attempts": consumed,
                    "author_charged_positions": 1488 * consumed, "paired_random_charged_positions": 1488 * consumed,
                    "ledger_head_sha256": events[-1]["sha256"], "model_dispatch_implemented": False}

    @staticmethod
    def event(events, kind, arm):
        matches = [e for e in events if e["kind"] == kind and e["arm"] == arm]
        if len(matches) != 1:
            raise ValueError("exactly one prior " + kind + " required")
        return matches[0]["payload"]

    def stop_present(self):
        return os.path.lexists(self.root / "STOP") or os.path.lexists(self.repo / self.p.PHASE1 / "STOP")

    def require_open(self, stopped):
        if stopped or self.stop_present():
            raise ValueError("terminal STOP; no further action")

    def reserve(self, arm, approval):
        with self.locked():
            events, stages, stopped = self._read()
            self.require_open(stopped)
            if arm in stages or arm not in ("A", "B") or (arm == "B" and stages != {"A": "evaluated"}) or (arm == "A" and stages):
                raise ValueError("A-then-B fresh founder order; no retry/replacement")
            previous = self.event(events, "submit", "A")["quota"] if arm == "B" else None
            self.approval_check(approval, arm, previous)
            payload = {"approval": approval, "packet_sha256": self.packet_sha(arm), "parent_identity": approval["parent_identity"],
                       "reserved_author_positions": 1488, "reserved_paired_random_positions": 1488,
                       "parent_tensor_sha256": None, "role": "new_independent_founder"}
            return self._append(events, "reserve", arm, payload)

    def submit(self, arm, host_reference, route, quota):
        with self.locked():
            events, stages, stopped = self._read()
            self.require_open(stopped)
            if stages.get(arm) != "reserved" or not nonempty(host_reference):
                raise ValueError("one reserved unconsumed attempt and host reference required")
            reservation = self.event(events, "reserve", arm)
            approved = reservation["approval"]
            clock = now()
            route_check(route, self.packet_sha(arm), self.p.SCHEMA_SHA, clock)
            self.exact_route(route, arm)
            quota_check(quota, clock, approved["quota"])
            ignore = {"observed_at", "attestation_reference"}
            if {k: v for k, v in route.items() if k not in ignore} != {k: v for k, v in approved["route"].items() if k not in ignore}:
                raise ValueError("route changed since packet-specific approval")
            return self._append(events, "submit", arm, {"reservation_sha256": digest(reservation),
                "packet_sha256": self.packet_sha(arm), "host_reference": host_reference, "route": route, "quota": quota,
                "status": "consumed_unknown_until_original_complete_receipts", "consumed_author_positions": 1488,
                "consumed_paired_random_positions": 1488})

    def stop(self, arm, reason, receipt_reference):
        if not nonempty(reason) or not nonempty(receipt_reference):
            raise ValueError("specific terminal reason and original receipt reference required")
        with self.locked():
            events, stages, stopped = self._read()
            if stopped:
                raise ValueError("terminal STOP already recorded")
            if arm not in stages:
                raise ValueError("reserved attempt required")
            return self._append(events, "stop", arm, {"reason": reason, "receipt_reference": receipt_reference,
                                "no_retry": True, "no_refund": True, "comparison": "inconclusive"})

    def capture(self, arm, raw, audit_raw, native_usage_raw, host_result_raw):
        with self.locked():
            events, stages, stopped = self._read()
            self.require_open(stopped)
            if stages.get(arm) != "submitted":
                raise ValueError("consumed attempt required, no capture retry")
            if any(type(item) is not bytes for item in (raw, audit_raw, native_usage_raw, host_result_raw)):
                raise ValueError("exact original response/audit/native/host bytes required")
            directory = self.root / "capture" / arm
            restricted = self.root / "restricted_raw_evidence" / arm
            if os.path.lexists(directory) or os.path.lexists(restricted):
                raise ValueError("partial/duplicate capture transaction: terminal no retry")
            directory.mkdir(parents=True, exist_ok=False)
            # Raw candidate bytes are durable before any parser or audit check.
            self.p.write_once(directory / "RAW_RESPONSE.bin", raw)
            restricted_hashes = {}
            for name, content in (("SUBMITTED_AUDIT.bin", audit_raw), ("NATIVE_USAGE.bin", native_usage_raw), ("HOST_RESULT.bin", host_result_raw)):
                self.p.write_once(restricted / name, content)
                restricted_hashes["restricted_raw_evidence/" + arm + "/" + name] = raw_hash(content)
            status, value, usage, published_audit = "invalid", None, None, None
            try:
                audit = self.p.strict_json(audit_raw)
                reservation = self.event(events, "reserve", arm)
                submission = self.event(events, "submit", arm)
                effective = {"parent_identity": reservation["parent_identity"], "packet_sha256": reservation["packet_sha256"],
                             "route_attestation_reference": submission["route"]["attestation_reference"],
                             "quota_attestation_reference": submission["quota"]["attestation_reference"]}
                self.intake.validate_host_audit(audit, effective, submission, raw_hash(raw))
                host_result = self.p.strict_json(host_result_raw)
                usage = self.p.strict_json(native_usage_raw)
                self.host_check(host_result, audit, usage, native_usage_raw)
                published_audit = audit
                if audit["status"] == "refused":
                    raise ValueError("refused candidate")
                value = self.intake.validate_response(raw)
                status = "valid"
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                # Never publish attacker-controlled error strings or private trace bodies.
                status = "invalid"
            payload = {"status": status, "raw_sha256": raw_hash(raw), "tensor_sha256": digest(value["tensors"]) if value else None,
                       "numeric_fingerprint": self.intake.numeric_fingerprint(value["tensors"]) if value else None,
                       "artifacts": {"capture/" + arm + "/RAW_RESPONSE.bin": raw_hash(raw)},
                       "restricted_artifacts": restricted_hashes, "restricted_evidence_not_publishable": True}
            if status == "valid":
                for name, content in (("HOST_AUDIT.json", self.p.json_bytes(published_audit)), ("NATIVE_USAGE.json", native_usage_raw), ("HOST_RESULT.json", host_result_raw)):
                    self.p.write_once(directory / name, content)
                    payload["artifacts"]["capture/" + arm + "/" + name] = raw_hash((directory / name).read_bytes())
            event = self._append(events, "capture", arm, payload)
            if status != "valid":
                self._append(events + [event], "stop", arm, {"reason": "invalid_response_or_host_audit", "no_retry": True,
                                                            "no_refund": True, "comparison": "inconclusive"})
            return event

    @staticmethod
    def host_check(value, audit, usage, usage_raw):
        keys = {"accepted", "rejection", "exit_code", "stdout_EOF", "stderr_EOF", "wall_seconds", "peak_owned_RSS_KiB",
                "raw_stdout_sha256", "raw_stderr_sha256", "raw_line_audit", "unterminated_tail_bytes", "validated_prefix_metadata",
                "charged_attempts", "no_automatic_wrapper_retry", "native_usage_receipt_sha256", "stderr_classification"}
        if type(value) is not dict or set(value) != keys:
            raise ValueError("exact original V5 result required")
        expected = {"accepted": True, "rejection": None, "exit_code": 0, "stdout_EOF": True, "stderr_EOF": True,
                    "raw_stdout_sha256": audit["restricted_trace_sha256"], "raw_stderr_sha256": raw_hash(b""),
                    "unterminated_tail_bytes": 0, "charged_attempts": 1, "no_automatic_wrapper_retry": True,
                    "native_usage_receipt_sha256": raw_hash(usage_raw),
                    "stderr_classification": "empty required; no observed benign byte sequence approved"}
        if any(type(value[k]) is not type(v) or value[k] != v for k, v in expected.items()):
            raise ValueError("host timeout/incomplete resource or capture binding")
        if type(value["wall_seconds"]) not in (int, float) or not math.isfinite(value["wall_seconds"]) or not 0 <= value["wall_seconds"] <= 120:
            raise ValueError("author wall limit exceeded")
        if type(value["peak_owned_RSS_KiB"]) is not int or not 0 <= value["peak_owned_RSS_KiB"] <= 2097152:
            raise ValueError("author memory limit exceeded")
        metadata = audit["event_metadata"]
        lines, validated = value["raw_line_audit"], value["validated_prefix_metadata"]
        if type(lines) is not list or type(validated) is not list or len(lines) != len(metadata) or len(validated) != len(metadata):
            raise ValueError("incomplete original V5 line audit")
        for index, (line, passed, event) in enumerate(zip(lines, validated, metadata)):
            if type(line) is not dict or set(line) != {"raw_line_index", "raw_line_bytes", "raw_line_sha256"} or type(line["raw_line_bytes"]) is not int or line["raw_line_bytes"] <= 0:
                raise ValueError("invalid V5 line audit")
            if line["raw_line_index"] != index or line["raw_line_sha256"] != event["event_sha256"] or passed != {**line, "type": event["event_type"]}:
                raise ValueError("V5 raw/validated/host event correspondence mismatch")
        if type(usage) is not dict or set(usage) != {"native_usage", "raw_turn_completed_line_sha256", "full_stdout_trace_sha256", "raw_turn_completed_line_index"}:
            raise ValueError("exact original native usage receipt required")
        native = usage["native_usage"]
        if type(native) is not dict or set(native) != NATIVE_COUNTERS or any(type(v) is not int or v < 0 for v in native.values()):
            raise ValueError("exact five native counters required")
        if usage["raw_turn_completed_line_sha256"] != metadata[-1]["event_sha256"] or usage["full_stdout_trace_sha256"] != audit["restricted_trace_sha256"] or usage["raw_turn_completed_line_index"] != len(metadata) - 1:
            raise ValueError("native usage/trace/event binding mismatch")
        if any(native[k] != audit["usage"][k] for k in ("input_tokens", "cached_input_tokens", "output_tokens")):
            raise ValueError("native/legacy usage projection mismatch")

    def random_response(self, arm):
        # Caller is the bounded worker during an already-consumed valid attempt.
        events, stages, stopped = self._read()
        self.require_open(stopped)
        self.event(events, "submit", arm)
        if stages.get(arm) != "evaluating":
            raise ValueError("random construction only during consumed evaluation")
        rng, tensors = random.Random(self.p.SEEDS[arm]), {}
        for item in self.intake.tensor_schema():
            values = [rng.choice(self.intake.VOCABULARY) for _ in range(item["positions"])]
            tensors[item["name"]] = [values] if len(item["shape"]) == 2 else values
        return {"schema": "nca-author-tensors/1", "tensors": tensors}

    def evaluate(self, arm):
        with self.locked():
            events, stages, stopped = self._read()
            self.require_open(stopped)
            if stages.get(arm) != "captured" or (self.root / "evaluation" / arm).exists():
                raise ValueError("valid captured attempt required; no evaluation retry")
            self._append(events, "evaluate_begin", arm, {"no_retry": True})
            output = self.root / "evaluation" / arm
            output.mkdir(parents=True, exist_ok=False)
            try:
                module, _, tasks = self.intake.frozen_runtime(self.repo / self.p.PHASE1)
                candidate = self.run_worker(arm, "candidate", output, module, tasks)
                if not candidate["summary"]["valid"]:
                    raise ValueError("invalid candidate runtime")
                matched = self.run_worker(arm, "random", output, module, tasks)
                if not matched["summary"]["valid"]:
                    raise ValueError("invalid matched random runtime")
                result = self.comparison(arm, candidate, matched)
                self.p.write_once(output / "COMPARISON.json", self.p.json_bytes(result))
                artifacts = {path.relative_to(self.root).as_posix(): raw_hash(path.read_bytes()) for path in sorted(output.rglob("*")) if path.is_file()}
                events, _, _ = self._read()
                return self._append(events, "evaluate_end", arm, {"valid": True, "artifacts": artifacts})
            except Exception:
                events, _, _ = self._read()
                self._append(events, "stop", arm, {"reason": "invalid_or_interrupted_evaluation", "no_retry": True,
                                                  "no_refund": True, "comparison": "inconclusive"})
                raise

    def comparison(self, arm, candidate, matched):
        phase1 = self.repo / self.p.PHASE1
        references = [self.intake.read_json(phase1 / "baseline" / (name + ".json")) for name in ("identity", "neighbor_majority", "zero")]
        randoms = [matched]
        a = None
        if arm == "B":
            randoms.insert(0, self.intake.read_json(self.root / "evaluation/A/random.json"))
            a = self.intake.read_json(self.root / "evaluation/A/candidate.json")
        feasibility = self.intake.compare(candidate, references, randoms, references[0])
        identity_predictions = [row["prediction"] for row in references[0]["demonstrations"]]
        prediction_identity = [row["prediction"] for row in candidate["demonstrations"]] == identity_predictions
        raw = self.intake.validate_response((self.root / "capture" / arm / "RAW_RESPONSE.bin").read_bytes())
        identity_tensors = self.intake.read_json(phase1 / "tensors/identity.json")
        # Saved phase1 tensor containers use 'tensors'; no reference is shown to authors.
        tensors = identity_tensors.get("tensors", identity_tensors)
        result = {"arm": arm, "positive_feasibility": feasibility["positive_feasibility"],
                  "selected_reference_anchors_preserved": feasibility["selected_baseline_anchors_preserved"],
                  "identity_numeric_equivalent": self.intake.numeric_fingerprint(raw["tensors"]) == self.intake.numeric_fingerprint(tensors),
                  "identity_prediction_equivalent_on_17_demos": prediction_identity,
                  "descriptive_B_minus_A_exact_demonstrations": None if a is None else candidate["summary"]["exact_demonstrations"] - a["summary"]["exact_demonstrations"],
                  "B_vs_A_secondary": None if a is None else {"A": a["summary"], "B": candidate["summary"]},
                  "comparison_complete": arm == "B", "causal_or_generalization_claim": False,
                  "qualification": "One A-then-B pair, exposed training only; B>A alone is not feasibility or capability gain."}
        if arm == "B":
            # Reassess A against BOTH consumed random controls; no provisional
            # A-only positive label may survive a stronger B-matched random.
            result["A_positive_feasibility_over_all_consumed_randoms"] = self.intake.compare(a, references, randoms, references[0])["positive_feasibility"]
        return result

    def run_worker(self, arm, kind, output, module, tasks):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        env.update({key: "1" for key in self.intake.THREAD_KEYS})
        phase1 = self.repo / self.p.PHASE1
        env["PYTHONPATH"] = str(phase1 / "source/src") + os.pathsep + str(phase1 / "source/scripts")
        command = [sys.executable, str(self.repo / SOURCE), "worker", "--repo", str(self.repo), "--supplement", str(self.supplement_path),
                   "--supplement-sha256", self.supplement_sha, "--state", str(self.root), "--arm", arm, "--kind", kind]
        limits = self.p.strict_json((self.preparation / "EVALUATOR_ONLY.json").read_bytes())["resource_limits"]
        error, stdout, stderr, code = None, b"", b"", None
        started = time.perf_counter()
        try:
            done = subprocess.run(command, cwd=self.repo, env=env, capture_output=True, timeout=limits["arm_wall_seconds"], check=False)
            stdout, stderr, code = done.stdout, done.stderr, done.returncode
            if code:
                error = {"type": "SubprocessError", "message": "bounded worker failed"}
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
            error = {"type": "ArmTimeout", "message": "bounded worker exceeded wall limit"}
        for suffix, raw in (("stdout.txt", stdout), ("stderr.txt", stderr)):
            self.p.write_once(output / (kind + "." + suffix), raw)
        progress = output / (kind + ".progress.jsonl")
        rows = []
        if progress.exists():
            for line in progress.read_bytes().splitlines():
                try:
                    rows.append(self.p.strict_json(line))
                except ValueError:
                    error = {"type": "IncompleteProgress", "message": "partial worker row"}
                    break
        expected_keys = [(task["task_id"], index) for task in tasks for index in range(len(task["train"]))]
        if [(row.get("task_id"), row.get("demo_index")) for row in rows] != expected_keys[:len(rows)]:
            raise ValueError("bounded worker row order/coverage mismatch")
        if len(rows) != len(expected_keys):
            error = {"type": "IncompleteArm", "message": "missing demonstrations"}
        rows.extend(module.error_row(task_id, index, error) for task_id, index in expected_keys[len(rows):])
        summary = module.summarize(rows, tasks)
        if error:
            summary.update(valid=False, exact_demonstrations=None, exact_tasks=None, mean_normalized_pixel_error=None)
        result = {"arm": kind, "summary": summary, "demonstrations": rows, "process_error": error,
                  "returncode": code, "process_elapsed_seconds": time.perf_counter() - started}
        self.p.write_once(output / (kind + ".json"), self.p.json_bytes(result))
        return result

    def worker(self, arm, kind):
        # Parent holds the ledger lock throughout evaluate; worker only reads it.
        self.verify_bindings()
        events, stages, stopped = self._read()
        self.require_open(stopped)
        if stages.get(arm) != "evaluating" or kind not in ("candidate", "random"):
            raise ValueError("worker requires active consumed evaluation")
        self.event(events, "submit", arm)
        module, manifest, tasks = self.intake.frozen_runtime(self.repo / self.p.PHASE1)
        module.apply_limits()
        output = self.root / "evaluation" / arm
        if kind == "candidate":
            response = self.intake.validate_response((self.root / "capture" / arm / "RAW_RESPONSE.bin").read_bytes())
        else:
            candidate = self.intake.read_json(output / "candidate.json")
            if not candidate["summary"]["valid"]:
                raise ValueError("invalid candidate forbids further control evaluation")
            response = self.random_response(arm)
            self.p.write_once(output / "random.tensors.json", self.p.json_bytes(response))
        graph, hashes = self.intake.inject_literals(module, self.repo / self.p.PHASE1, manifest, response)
        self.p.write_once(output / (kind + ".injection.json"), self.p.json_bytes({"graph_sha256": graph.hash(),
            "tensor_sha256": digest(response["tensors"]), "per_tensor_sha256": hashes,
            "phase1_structure_sha256": manifest["graph"]["structure_sha256"], "all_1488_literals_injected_unchanged": True}))
        *_, Predictor, _ = module.runtime_imports()
        predictor = Predictor(graph, backend="reference", cache_demo_state=False)
        limits = self.p.strict_json((self.preparation / "EVALUATOR_ONLY.json").read_bytes())["resource_limits"]
        def timeout(signum, frame):
            raise TimeoutError("per-demo wall limit exceeded")
        signal.signal(signal.SIGALRM, timeout)
        with (output / (kind + ".progress.jsonl")).open("xb") as handle:
            for task in tasks:
                for index in range(len(task["train"])):
                    if self.stop_present():
                        raise ValueError("STOP present")
                    signal.setitimer(signal.ITIMER_REAL, limits["demo_wall_seconds"])
                    try:
                        row = module.evaluate_demo(predictor, task, index)
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    handle.write(self.p.json_bytes(row))
                    handle.flush()
                    os.fsync(handle.fileno())
        return {"worker_completed": True, "arm": arm, "kind": kind}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze-supplement", "verify", "initialize", "status", "reserve", "submit", "capture", "stop", "evaluate", "worker"])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--supplement-sha256")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--arm", choices=["A", "B"])
    parser.add_argument("--kind", choices=["candidate", "random"])
    for name in ("approval", "route", "quota", "raw", "audit", "native-usage", "host-result"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--host-reference")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    if args.action == "freeze-supplement":
        result = {"supplement_sha256": freeze_supplement(args.repo, args.supplement), "parent_execution_approval": False}
    else:
        if args.supplement_sha256 is None or args.state is None:
            parser.error("exact supplement hash and separate state path required")
        adapter = Adapter(args.repo, args.supplement, args.supplement_sha256, args.state)
        def read(name):
            path = getattr(args, name)
            if path is None:
                parser.error("--" + name.replace("_", "-") + " required")
            return adapter.p.strict_json(path.read_bytes())
        if args.action == "verify":
            result = {"bindings_verified": True, "parent_execution_approval": False, "state_created": False}
        elif args.action == "initialize":
            result = adapter.initialize(read("approval"))
        elif args.action == "status":
            result = adapter.status()
        elif args.action == "reserve":
            result = adapter.reserve(args.arm, read("approval"))
        elif args.action == "submit":
            result = adapter.submit(args.arm, args.host_reference, read("route"), read("quota"))
        elif args.action == "capture":
            if args.raw is None or args.native_usage is None or args.host_result is None:
                parser.error("--raw, original --native-usage and --host-result required")
            if args.audit is None:
                parser.error("--audit original receipt bytes required")
            result = adapter.capture(args.arm, args.raw.read_bytes(), args.audit.read_bytes(), args.native_usage.read_bytes(), args.host_result.read_bytes())
        elif args.action == "stop":
            result = adapter.stop(args.arm, args.reason, args.host_reference)
        elif args.action == "evaluate":
            result = adapter.evaluate(args.arm)
        else:
            result = adapter.worker(args.arm, args.kind)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
