"""Freeze one search-feedback refinement packet; never author/fit new weights."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "weights-belief-native-packet-20261006"
PARENT_FREEZE = "79ebfc7d0fdddcddcadad2d3a416f5f990d6244f94a2e35c30451d7541d80f7f"
PARENT_RESULT = "9a81f6dfa6f8dec4d782bc89512a40be195765e5ad460f13b475f9f838208dce"
PACKET_ID = "belief-native-20261006-refine-v2"
RANDOM_SEED = 2026100632


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write(name, raw):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def forward_trace(b, cell, inputs):
    x, p = inputs.vector(), cell.parameters
    preactivation = tuple(b._affine(p[row * 12:(row + 1) * 12], x, p[192 + row]) for row in range(16))
    hidden = tuple(math.tanh(value) for value in preactivation)
    logits = tuple(b._affine(p[208 + row * 16:208 + (row + 1) * 16], hidden, p[288 + row]) for row in range(5))
    exponentials = tuple(math.exp(value - max(logits)) for value in logits)
    output = tuple(value / math.fsum(exponentials) for value in exponentials)
    if output != cell.forward(inputs).values:
        raise ValueError("trace differs from frozen forward")
    # These groups describe the existing V1 coefficients only. No new weights.
    groups = {label: [math.fsum(p[208 + row * 16 + j] * hidden[j] for j in nodes)
                      for row in range(5)] for label, nodes in
              (("hidden_0_through_4", range(5)), ("hidden_5_through_9", range(5, 10)),
               ("hidden_10_through_15", range(10, 16)))}
    return {"input": list(x), "hidden_preactivations": list(preactivation),
            "hidden_tanh_outputs": list(hidden), "output_logits": list(logits),
            "output_probabilities": list(output), "output_contributions": groups,
            "output_bias": list(p[288:293])}


def main():
    if (ROOT / "FREEZE.json").exists() or (ROOT / "author").exists():
        raise SystemExit("V2 preparation already started/frozen; never overwrite.")
    parent_raw = (PARENT / "FREEZE.json").read_bytes()
    if digest(parent_raw) != PARENT_FREEZE:
        raise ValueError("V1 freeze changed")
    parent_manifest = json.loads(parent_raw)
    for name, expected in parent_manifest["files"].items():
        if digest((PARENT / name).read_bytes()) != expected:
            raise ValueError("V1 frozen artifact changed: " + name)
    result_raw = (PARENT / "host/attempt/RESULT.json").read_bytes()
    if digest(result_raw) != PARENT_RESULT:
        raise ValueError("V1 result changed")
    result = json.loads(result_raw)
    attempt_raw = (PARENT / "host/attempt/ATTEMPT.json").read_bytes()
    attempt = json.loads(attempt_raw)
    if digest(attempt_raw) != result["attempt_sha256"]:
        raise ValueError("V1 attempt changed")
    response_raw = (PARENT / "host/attempt/RESPONSE_CAPTURE.bin").read_bytes()
    if digest(response_raw) != attempt["response_sha256"] or attempt["status"] != "valid":
        raise ValueError("V1 response changed/invalid")
    b = load("v2_parent_belief", PARENT / "host/runtime/belief_cell.py")
    evaluator = load("v2_parent_evaluator", PARENT / "host/evaluate.py")
    old_protocol = json.loads((PARENT / "host/PROTOCOL.json").read_bytes())
    cell, _ = evaluator.parse_response(b, old_protocol, response_raw)
    if cell != b.BeliefCell.from_dict(json.loads((PARENT / "host/attempt/IMPORTED_CELL.json").read_bytes())):
        raise ValueError("V1 imported coefficients differ from captured response")
    packet = json.loads((PARENT / "author/AUTHORING_PACKET.json").read_bytes())
    search = json.loads((PARENT / "host/SEARCH.json").read_bytes())
    record = search[0]
    scenario = evaluator.unpack(b, record)
    saved = result["arms"]["native"]["splits"]["search"]["scenarios"][0]
    if saved["scenario_id"] != record["scenario_id"]:
        raise ValueError("search scenario mismatch")
    reference_state = model_state = scenario.initial_beliefs
    propagation, primary = [], None
    for index in range(3):
        event, query = scenario.events[index], record["queries"][index]
        teacher_input = evaluator.slot_input(b, reference_state, event, query)
        model_input = evaluator.slot_input(b, model_state, event, query)
        teacher_trace = forward_trace(b, cell, teacher_input)
        rollout_trace = forward_trace(b, cell, model_input)
        target_state = b.reference_step(reference_state, event)
        expected = list(target_state.beliefs[query["actor_id"]][query["object_id"]].values)
        next_model = b.apply_event(cell, model_state, event)
        actual = list(next_model.beliefs[query["actor_id"]][query["object_id"]].values)
        if (teacher_trace["output_probabilities"] != saved["teacher_forced"]["outcomes"][index]["predicted"]
                or actual != saved["rollout"]["outcomes"][index]["predicted"]
                or rollout_trace["output_probabilities"] != actual or expected != query["expected"]):
            raise ValueError("search trace differs from frozen evidence")
        if index == 0:
            if query["category"] != "seen" or teacher_trace["input"] != packet["search_examples"][0]["input"]:
                raise ValueError("primary failure is not the declared observed search example")
            old_top = max(range(5), key=teacher_input.old.values.__getitem__)
            expected_top = max(range(5), key=expected.__getitem__)
            actual_top = max(range(5), key=actual.__getitem__)
            if actual_top == expected_top:
                raise ValueError("selected case is not a failure")
            old_group = teacher_trace["output_contributions"]["hidden_0_through_4"]
            new_group = teacher_trace["output_contributions"]["hidden_5_through_9"]
            primary = {"scenario_id": record["scenario_id"], "event_index": 0,
                       "actor_id": query["actor_id"], "object_id": query["object_id"],
                       "category": "seen", "mode": "teacher_forced", "expected": expected,
                       "expected_top1": expected_top, "actual_top1": actual_top,
                       **teacher_trace,
                       "measured_mechanism": {
                           "scope": "directly recomputed from this search input and the supplied V1 coefficients",
                           "observation": "Both event bits are 1. Hidden nodes 5..9 receive preactivations 8 or 12 and saturate near +1; their output contributions are nearly common across locations. Hidden nodes 0..4 carry the old-location signal. The equal output biases cancel in comparisons. The resulting largest logit selects old location 0 rather than the observed location 1.",
                           "old_path_advantage_old_over_expected": old_group[old_top] - old_group[expected_top],
                           "new_path_advantage_expected_over_old": new_group[expected_top] - new_group[old_top],
                           "total_logit_advantage_old_over_expected": teacher_trace["output_logits"][old_top] - teacher_trace["output_logits"][expected_top],
                           "replacement_weights_provided": False,
                           "untested_hypotheses_claimed_proven": False}}
        if index and list(model_input.old.values) != propagation[-1]["actual"]:
            raise ValueError("rollout old state was repaired/replaced")
        propagation.append({"scenario_id": record["scenario_id"], "event_index": index,
                            "actor_id": query["actor_id"], "object_id": query["object_id"],
                            "category": query["category"], "model_input": list(model_input.vector()),
                            "expected": expected, "actual": actual,
                            "teacher_forced_input": list(teacher_input.vector()),
                            "teacher_forced_actual": teacher_trace["output_probabilities"],
                            "model_old_is_previous_model_output": index > 0})
        reference_state, model_state = target_state, next_model
    packet["packet_id"] = PACKET_ID
    packet["task"] = "Refine the supplied failed V1 literal weights in ONE complete 293-number response using only the original search examples and verified search failure trace."
    packet["instructions"].extend([
        "This is one reviewed refinement, not a request to sample unrelated weights blindly. Inspect the supplied failure and preserve behavior that the search evidence supports.",
        "The prior weights are the author's actual failed proposal, not host-designed replacement weights. All 293 response positions will be charged regardless of how many change.",
        "Use the exact trace as search-only feedback. Do not infer that teacher-forced preservation repairs recursive rollout errors.",
        "No development statistics, labels, examples, trace or outcome are provided or authorized. Do not seek them.",
        "No third author attempt is authorized without a new review."])
    packet["prior_candidate"] = {"parameter_hash": cell.parameter_hash, "parameters": list(cell.parameters),
                                  "provenance": "previous native author's exact captured numeric proposal; failed search behavior below"}
    packet["search_failure_trace"] = {"primary_teacher_forced_failure": primary,
                                       "recursive_error_propagation": propagation,
                                       "verification": "Host independently recomputed these three original search events from captured V1 weights with frozen forward semantics; outputs exactly match saved V1 search evidence. Reference targets never replace model state."}
    schema = json.loads((PARENT / "author/RESPONSE_SCHEMA.json").read_bytes())
    schema["properties"]["packet_id"]["const"] = PACKET_ID
    write("author/AUTHORING_PACKET.json", encoded(packet))
    write("author/RESPONSE_SCHEMA.json", encoded(schema))
    for name in ("host/SEARCH.json", "host/DEVELOPMENT.json", "host/runtime/belief_cell.py", "host/evaluate.py"):
        write(name, (PARENT / name).read_bytes())
    protocol = dict(old_protocol)
    protocol.update(packet_id=PACKET_ID, random_seed=RANDOM_SEED,
                    family_author_attempt_ordinal=2, family_author_attempt_limit=2,
                    third_author_requires_review=True,
                    access_boundary="fresh author context, explicit two-file read restrictions, tool access may exist; prompt-restricted, not tool-disabled or an enforced filesystem sandbox",
                    source_status="V2 author not invoked; source IDs and exact credit usage unknown",
                    parent_parameter_hash=cell.parameter_hash,
                    parent_freeze_sha256=PARENT_FREEZE, parent_result_sha256=PARENT_RESULT,
                    parent_response_sha256=attempt["response_sha256"],
                    panel_policy="byte-identical V1 search/development data and unchanged evaluation semantics; one new predeclared random control seed for this separately charged attempt",
                    feedback_policy="original search examples plus one exact observed-update failure and its next two search rollout outcomes only")
    write("host/PROTOCOL.json", encoded(protocol))
    standard = {"arms": {"llm": {}, "random": {}}, "seeds": [RANDOM_SEED],
                "seeds_declared_before_run": True, "budget_per_arm": 0,
                "niches": ["structured-belief-update"], "founders": [], "genomes": [], "proposals": [],
                "scientific_acceptance": False, "status": "V2 refinement packet only; no new author attempt consumed",
                "parent_parameter_hash": cell.parameter_hash,
                "limitations": ["single control seed below qualification", "author credits unknown", "no V2 author response yet", "reused development panel is not fresh held-out evidence"]}
    write("STANDARD.json", encoded(standard))
    verification = {"schema": "cellforge-belief-v2-preparation/1", "parent_freeze_sha256": PARENT_FREEZE,
                    "parent_result_sha256": PARENT_RESULT, "parent_parameter_hash": cell.parameter_hash,
                    "search_scenario_id": record["scenario_id"], "search_event_indices": [0, 1, 2],
                    "forward_trace_equal": True, "saved_search_outcomes_equal": True,
                    "rollout_uses_own_predictions": True, "host_designed_replacement_weights": False,
                    "author_calls_by_preparation": 0, "new_author_credit_usage": None,
                    "source_and_data_parent_hashes": {name: digest((PARENT / name).read_bytes()) for name in
                        ("host/SEARCH.json", "host/DEVELOPMENT.json", "host/runtime/belief_cell.py", "host/evaluate.py")},
                    "author_visible_feedback": "one primary teacher-forced observed-update failure, its exact forward intermediates, and first three search rollout outcomes; no development evidence",
                    "primary_failure": {"expected_top1": primary["expected_top1"], "actual_top1": primary["actual_top1"],
                                        **primary["measured_mechanism"]}}
    write("host/PREPARATION_VERIFICATION.json", encoded(verification))
    paths = sorted(path for path in ROOT.rglob("*") if path.is_file())
    if any("__pycache__" in path.parts for path in paths):
        raise ValueError("Unexpected bytecode; use python -B")
    manifest = {"schema": "cellforge-belief-freeze/1", "packet_id": PACKET_ID,
                "source_path_at_freeze": "V1 frozen host/runtime/belief_cell.py",
                "source_sha256": parent_manifest["source_sha256"],
                "parent_freeze_sha256": PARENT_FREEZE, "parent_result_sha256": PARENT_RESULT,
                "files": {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in paths}}
    raw = encoded(manifest)
    write("FREEZE.json", raw)
    write("FREEZE.sha256", (digest(raw) + "\n").encode("ascii"))
    print(json.dumps({"freeze_sha256": digest(raw), "packet": str(ROOT / "author/AUTHORING_PACKET.json"),
                      "schema": str(ROOT / "author/RESPONSE_SCHEMA.json"),
                      "search_trace_verified": True, "panel_bytes_unchanged": True, "author_calls": 0}))


if __name__ == "__main__":
    main()
