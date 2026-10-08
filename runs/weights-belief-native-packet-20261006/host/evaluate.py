"""Frozen host-only one-attempt importer/evaluator. Never send this to author."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_RESPONSE_BYTES = 65536


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_new(path, value):
    with path.open("xb") as stream:
        stream.write(encode(value))


def pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def no_constant(value):
    raise ValueError("nonfinite JSON literal: " + value)


def parse(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=no_constant)


def verify(freeze_hash):
    raw = (ROOT / "FREEZE.json").read_bytes()
    if len(freeze_hash) != 64 or digest(raw) != freeze_hash:
        raise ValueError("FREEZE hash differs from externally supplied pre-author anchor")
    manifest = parse(raw)
    for name, expected in manifest["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path.read_bytes()) != expected:
            raise ValueError("frozen file differs: " + name)
    if manifest["files"]["host/runtime/belief_cell.py"] != manifest["source_sha256"]:
        raise ValueError("runtime snapshot differs from declared source")
    spec = importlib.util.spec_from_file_location("frozen_belief", ROOT / "host/runtime/belief_cell.py")
    b = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = b
    spec.loader.exec_module(b)
    protocol = parse((ROOT / "host/PROTOCOL.json").read_bytes())
    return b, protocol


def distribution(b, values):
    return b.Distribution(tuple(values))


def unpack(b, record):
    initial = b.BeliefState(record["actor_count"], record["object_count"], tuple(
        tuple(distribution(b, values) for values in row) for row in record["initial_beliefs"]))
    events = tuple(b.MoveEvent(event["object_id"], event["destination"], tuple(event["observed_by"]))
                   for event in record["events"])
    return b.Scenario(b.WorldState(tuple(record["initial_world"])), initial, events)


def slot_input(b, state, event, query):
    observed = query["actor_id"] in event.observed_by
    return b.KernelInput(state.beliefs[query["actor_id"]][query["object_id"]],
                         b.Distribution.at(event.destination if observed else b.UNKNOWN),
                         query["object_id"] == event.object_id, observed)


def category(inputs):
    if not inputs.object_match:
        return "unrelated"
    if not inputs.observation:
        return "unseen"
    return "unknown" if inputs.observed_new.values[4] == 1 else "seen"


def validate_data(b, protocol):
    datasets, visible = {}, []
    identities = set()
    for split in ("search", "development"):
        rows = parse((ROOT / f"host/{split.upper()}.json").read_bytes())
        if [row["seed"] for row in rows] != protocol[split + "_seeds"]:
            raise ValueError("seed declaration mismatch")
        count = {key: 0 for key in protocol["query_categories"]}
        for record in rows:
            if record["scenario_id"] in identities or record["family"] != protocol[split + "_family"]:
                raise ValueError("scenario identity/family mismatch")
            identities.add(record["scenario_id"])
            scenario = unpack(b, record)
            state = scenario.initial_beliefs
            if len(record["queries"]) != len(scenario.events):
                raise ValueError("one scored query per event required")
            for index, (event, query) in enumerate(zip(scenario.events, record["queries"])):
                if query["event_index"] != index:
                    raise ValueError("query index mismatch")
                if not (type(query["actor_id"]) is int and 0 <= query["actor_id"] < state.actor_count
                        and type(query["object_id"]) is int and 0 <= query["object_id"] < state.object_count):
                    raise ValueError("query slot out of bounds")
                inputs = slot_input(b, state, event, query)
                expected = b.reference_update(inputs)
                if category(inputs) != query["category"] or distribution(b, query["expected"]) != expected:
                    raise ValueError("category/reference target mismatch")
                count[query["category"]] += 1
                if split == "search":
                    visible.append({"scenario_id": record["scenario_id"], "event_index": index,
                                    "category": query["category"], "input": list(inputs.vector()),
                                    "expected": list(expected.values)})
                state = b.reference_step(state, event)
        if len(set(count.values())) != 1 or sum(count.values()) != protocol[split + "_queries"]:
            raise ValueError("unbalanced or wrong query count")
        datasets[split] = rows
    if set(protocol["search_seeds"]) & set(protocol["development_seeds"]):
        raise ValueError("search/development seeds overlap")
    packet = parse((ROOT / "author/AUTHORING_PACKET.json").read_bytes())
    if packet["search_examples"] != visible or packet["parameter_schema"] != b.parameter_schema():
        raise ValueError("author examples/schema differ from frozen search/runtime")
    return datasets


def parse_response(b, protocol, raw):
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("response exceeds 65536-byte contract")
    response = parse(raw)
    if type(response) is not dict or set(response) != {"schema", "packet_id", "parameters", "rationale"}:
        raise ValueError("exact response wrapper required")
    if response["schema"] != "cellforge-belief-native-response/1" or response["packet_id"] != protocol["packet_id"]:
        raise ValueError("response schema/packet mismatch")
    if type(response["rationale"]) is not str or not 1 <= len(response["rationale"]) <= 4000:
        raise ValueError("bounded rationale required")
    if type(response["parameters"]) is not list or len(response["parameters"]) != 293:
        raise ValueError("293 literal parameters required")
    return b.BeliefCell(tuple(response["parameters"])), response["rationale"]


def metric(predicted, expected, tolerance):
    actual_top = max(range(5), key=predicted.__getitem__)
    expected_top = max(range(5), key=expected.__getitem__)
    differences = [abs(a - e) for a, e in zip(predicted, expected)]
    return {"predicted": list(predicted), "expected": list(expected),
            "correct": actual_top == expected_top,
            "probability_error": max(differences) > tolerance,
            "l1": math.fsum(differences)}


def summarize(outcomes):
    complete = [outcome for outcome in outcomes if outcome["status"] == "predicted"]
    correct = sum(outcome["correct"] for outcome in complete)
    return {"queries": len(outcomes), "predictions": len(complete), "correct": correct,
            "categorical_errors": len(complete) - correct,
            "unresolved": len(outcomes) - len(complete),
            "runtime_errors": sum(outcome["status"] == "runtime_error" for outcome in outcomes),
            "probability_errors": sum(outcome["probability_error"] for outcome in complete),
            "mean_l1": math.fsum(outcome["l1"] for outcome in complete) / len(complete) if complete else None,
            "accuracy": correct / len(outcomes), "solved": correct == len(outcomes)}


def score(b, protocol, records, cell, oracle=False):
    results = []
    for record in records:
        scenario = unpack(b, record)
        model_state, reference_state = scenario.initial_beliefs, scenario.initial_beliefs
        modes = {"rollout": [], "teacher_forced": []}
        blocked = False
        for event, query in zip(scenario.events, record["queries"]):
            target = b.reference_step(reference_state, event)
            expected = target.beliefs[query["actor_id"]][query["object_id"]].values
            base = {key: query[key] for key in ("event_index", "actor_id", "object_id", "category")}
            for mode in modes:
                if mode == "rollout" and blocked:
                    modes[mode].append(dict(base, status="not_executed_after_error"))
                    continue
                try:
                    if mode == "rollout":
                        model_state = b.reference_step(model_state, event) if oracle else b.apply_event(cell, model_state, event)
                        prediction = model_state.beliefs[query["actor_id"]][query["object_id"]]
                    else:
                        inputs = slot_input(b, reference_state, event, query)
                        prediction = b.reference_update(inputs) if oracle else cell.forward(inputs)
                    modes[mode].append(dict(base, status="predicted", **metric(prediction.values, expected, protocol["probability_tolerance"])))
                except (b.BeliefError, ArithmeticError) as error:
                    modes[mode].append(dict(base, status="runtime_error", error_code=getattr(error, "code", type(error).__name__),
                                            detail=str(error)[:300]))
                    if mode == "rollout":
                        blocked = True
            reference_state = target
        results.append({"scenario_id": record["scenario_id"], "family": record["family"],
                        **{mode: {"summary": summarize(outcomes), "outcomes": outcomes} for mode, outcomes in modes.items()}})
    summary = {}
    for mode in ("rollout", "teacher_forced"):
        outcomes = [outcome for result in results for outcome in result[mode]["outcomes"]]
        summary[mode] = dict(summarize(outcomes),
            scenarios_solved=sum(result[mode]["summary"]["solved"] for result in results), scenario_count=len(results),
            by_category={key: summarize([outcome for outcome in outcomes if outcome["category"] == key]) for key in protocol["query_categories"]})
    return {"summary": summary, "scenarios": results}


def evaluate(b, protocol, datasets, cell=None):
    arms = {"zero": b.zero_cell(), "random": b.random_cell(protocol["random_seed"]), "reference": None}
    if cell is not None:
        arms["native"] = cell
    return {name: {"parameter_hash": kernel.parameter_hash if kernel is not None else None,
                   "kind": "authored reference rule; not weights" if name == "reference" else "literal MLP",
                   "splits": {split: score(b, protocol, records, kernel, oracle=name == "reference")
                              for split, records in datasets.items()}} for name, kernel in arms.items()}


def baseline_check(b, protocol, datasets, freeze_hash):
    checks = []
    template = {"schema": "cellforge-belief-native-response/1", "packet_id": protocol["packet_id"],
                "parameters": [0.0] * 293, "rationale": "Parser-only zero fixture; not a founder or author response."}
    cell, _ = parse_response(b, protocol, encode(template))
    if cell != b.zero_cell():
        raise ValueError("parser fixture differs")
    checks.append("literal_parser_zero_fixture")
    malformed = [b'{"schema":1,"schema":2}', b'{}', b'NaN']
    for kind in ("count", "bool", "infinity", "packet"):
        value = dict(template, parameters=list(template["parameters"]))
        if kind == "count":
            value["parameters"].pop()
        elif kind == "bool":
            value["parameters"][0] = True
        elif kind == "infinity":
            value["parameters"][0] = "infinity"
        else:
            value["packet_id"] = "wrong"
        malformed.append(encode(value))
    for index, raw in enumerate(malformed):
        try:
            parse_response(b, protocol, raw)
        except (ValueError, OverflowError, RecursionError):
            checks.append(f"reject_malformed_{index}")
        else:
            raise ValueError("malformed parser fixture accepted")
    try:
        verify("0" * 64)
    except ValueError:
        checks.append("reject_wrong_freeze_anchor")
    else:
        raise ValueError("wrong freeze anchor accepted")
    arms = evaluate(b, protocol, datasets)
    for split in datasets:
        for mode in ("rollout", "teacher_forced"):
            if not arms["reference"]["splits"][split]["summary"][mode]["solved"]:
                raise ValueError("reference control failed its own event targets")
    checks += ["frozen_data_balanced_and_search_only_packet", "reference_matches_all_frozen_targets"]
    return {"schema": "cellforge-belief-controls/1", "freeze_sha256": freeze_hash,
            "checks_passed": len(checks), "checks": checks, "arms": arms,
            "author_calls": 0, "author_credit_usage": None, "attempt_consumed": False,
            "scientific_acceptance": False}


def import_attempt(b, protocol, datasets, args):
    folder = ROOT / "host/attempt"
    folder.mkdir()  # Exactly one submission, consumed before reading/parsing.
    attempt = {"schema": "cellforge-belief-attempt/1", "freeze_sha256": args.expected_freeze,
               "attempt_index": 1, "charged": True, "native_positions": 293, "random_positions": 293,
               "author_credit_usage": None, "credit_status": "unknown, not free",
               "source_thread_id": None, "source_turn_id": None,
               "source_status": "parent must attach source receipt", "response_sha256": None}
    cell, reason = None, None
    try:
        with Path(args.response).open("rb") as stream:
            raw = stream.read(MAX_RESPONSE_BYTES + 1)
        attempt["capture_complete"] = len(raw) <= MAX_RESPONSE_BYTES
        attempt["captured_bytes"] = len(raw)
        attempt["response_sha256"] = digest(raw) if attempt["capture_complete"] else None
        attempt["capture_prefix_sha256"] = digest(raw)
        with (folder / "RESPONSE_CAPTURE.bin").open("xb") as stream:
            stream.write(raw)
        cell, reason = parse_response(b, protocol, raw)
        attempt["status"] = "valid"
    except (OSError, ValueError, OverflowError, RecursionError) as error:
        attempt["status"] = "invalid"
        attempt["error"] = str(error)[:500]
    write_new(folder / "ATTEMPT.json", attempt)
    if cell is not None:
        write_new(folder / "IMPORTED_CELL.json", cell.to_dict())
    result = {"schema": "cellforge-belief-result/1", "freeze_sha256": args.expected_freeze,
              "attempt_sha256": digest((folder / "ATTEMPT.json").read_bytes()),
              "native_status": attempt["status"], "native_rationale": reason,
              "arms": evaluate(b, protocol, datasets, cell), "scientific_acceptance": False,
              "admitted": False, "development_feedback_to_author": False}
    write_new(folder / "RESULT.json", result)
    return {"status": attempt["status"], "result": str(folder / "RESULT.json"),
            "result_sha256": digest((folder / "RESULT.json").read_bytes()), "author_credit_usage": None}


def replay(b, protocol, datasets, freeze_hash):
    folder = ROOT / "host/attempt"
    attempt = parse((folder / "ATTEMPT.json").read_bytes())
    saved = parse((folder / "RESULT.json").read_bytes())
    if saved["freeze_sha256"] != freeze_hash or attempt["freeze_sha256"] != freeze_hash:
        raise ValueError("attempt/result freeze mismatch")
    if saved["attempt_sha256"] != digest((folder / "ATTEMPT.json").read_bytes()):
        raise ValueError("attempt was altered")
    cell = None
    if attempt["status"] == "valid":
        raw = (folder / "RESPONSE_CAPTURE.bin").read_bytes()
        if digest(raw) != attempt["response_sha256"]:
            raise ValueError("response capture was altered")
        cell, _ = parse_response(b, protocol, raw)
        if b.BeliefCell.from_dict(parse((folder / "IMPORTED_CELL.json").read_bytes())) != cell:
            raise ValueError("imported cell differs from captured literals")
    recomputed = evaluate(b, protocol, datasets, cell)
    if saved["arms"] != recomputed:
        raise ValueError("recomputed arms differ from saved outcomes")
    return {"replay_equal": True, "result_sha256": digest((folder / "RESULT.json").read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-freeze", required=True, help="Pre-author hash retained externally by parent")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    submit = commands.add_parser("import")
    submit.add_argument("--response", required=True)
    commands.add_parser("replay")
    source = commands.add_parser("record-source")
    for name in ("source-thread-id", "source-turn-id", "model", "reasoning"):
        source.add_argument("--" + name, required=True)
    args = parser.parse_args()
    b, protocol = verify(args.expected_freeze)
    datasets = validate_data(b, protocol)
    if args.command == "check":
        result = baseline_check(b, protocol, datasets, args.expected_freeze)
        path = ROOT / "host/CONTROLS_CHECK.json"
        write_new(path, result)
        output = {"checks_passed": result["checks_passed"], "file": str(path), "sha256": digest(path.read_bytes()),
                  "rollout": {arm: {split: values["summary"]["rollout"] for split, values in data["splits"].items()}
                              for arm, data in result["arms"].items()}}
    elif args.command == "import":
        output = import_attempt(b, protocol, datasets, args)
    elif args.command == "replay":
        output = replay(b, protocol, datasets, args.expected_freeze)
    else:
        folder = ROOT / "host/attempt"
        attempt = parse((folder / "ATTEMPT.json").read_bytes())
        fields = {key: getattr(args, key) for key in ("source_thread_id", "source_turn_id", "model", "reasoning")}
        if any(not 1 <= len(value) <= 256 for value in fields.values()):
            raise ValueError("bounded source fields required")
        receipt = {"schema": "cellforge-belief-source-receipt/1", **fields,
                   "attempt_sha256": digest((folder / "ATTEMPT.json").read_bytes()),
                   "response_sha256": attempt["response_sha256"], "author_credit_usage": None,
                   "provenance_status": "parent-reported file-supervised native source; not cryptographically authenticated"}
        write_new(folder / "SOURCE_RECEIPT.json", receipt)
        output = {"source_receipt": str(folder / "SOURCE_RECEIPT.json"), "author_credit_usage": None}
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
