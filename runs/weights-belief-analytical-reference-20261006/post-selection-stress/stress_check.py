"""One declared post-selection stress pass over fixed analytical weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
PARENT_FREEZE = "078c1c23f60097769be127dc06115618b00e7609b58b8e337641ebbc825b5e88"
PARENT_RESULT = "1cb5a4b2efc629146b7db2926d4c6ada189b30b9f027df0affe99bd52b824a66"
WEIGHT_HASH = "9ca57f326afc539f87020a679e445842aec1eb2e5ad04c80615097f3d969e8d3"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write(name, raw):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def at(location):
    return [float(index == location) for index in range(5)]


def event(obj, location, observers):
    return {"object_id": obj, "destination": location, "observed_by": list(observers)}


def record(name, cohort, beliefs, events, world=None, **extra):
    return {"scenario_id": name, "cohort": cohort, "actor_count": 4, "object_count": 4,
            "initial_world": [4] * 4 if world is None else world,
            "initial_beliefs": beliefs, "events": events, **extra}


def rename(base, suffix, actors, locations):
    def distribution(old):
        result = [0.0] * 5
        for previous, new in enumerate(locations):
            result[new] = old[previous]
        return result
    rows = [None] * 4
    for actor in range(4):
        rows[actors[actor]] = [distribution(value) for value in base["initial_beliefs"][actor]]
    events = [event(e["object_id"], locations[e["destination"]],
                    sorted(actors[actor] for actor in e["observed_by"])) for e in base["events"]]
    return record(base["scenario_id"] + suffix, "categorical", rows, events,
                  [locations[value] for value in base["initial_world"]],
                  renamed_from=base["scenario_id"], actor_permutation=actors,
                  location_permutation=locations)


def prepare():
    if (ROOT / "FREEZE.json").exists() or (ROOT / "SCENARIOS.json").exists():
        raise ValueError("Stress preparation already started/frozen; never regenerate after results.")
    parent_raw = (PARENT / "FREEZE.json").read_bytes()
    if digest(parent_raw) != PARENT_FREEZE:
        raise ValueError("parent freeze changed")
    parent_manifest = json.loads(parent_raw)
    for name, expected in parent_manifest["files"].items():
        if digest((PARENT / name).read_bytes()) != expected:
            raise ValueError("parent frozen artifact changed: " + name)
    if digest((PARENT / "host/attempt/RESULT.json").read_bytes()) != PARENT_RESULT:
        raise ValueError("parent result changed")
    source = (PARENT / "host/runtime/belief_cell.py").read_bytes()
    weights = (PARENT / "host/attempt/IMPORTED_CELL.json").read_bytes()
    if json.loads(weights)["parameter_hash"] != WEIGHT_HASH:
        raise ValueError("unexpected successful weights")
    write("runtime/belief_cell.py", source)
    write("WEIGHTS.json", weights)
    # No random search or seed selection: these exact schedules precede scoring.
    coverage_events = [event(obj, (actor + obj) % 4, (actor,))
                       for actor in range(4) for obj in range(4)]
    coverage_events += [event(index % 4, 4 if index % 4 == 0 else (index // 4 + index) % 4, ())
                        for index in range(16)]
    coverage = record("all-pairs-then-unseen-32", "categorical",
                      [[at(4) for _ in range(4)] for _ in range(4)], coverage_events)
    distractors = []
    for index in range(32):
        mode, actor = index % 4, (index // 4) % 4
        observers = () if mode == 0 else ((actor,) if mode == 1 else
                     (tuple(sorted((actor, (actor + 2) % 4))) if mode == 2 else (0, 1, 2, 3)))
        distractors.append(event(index % 4, (index // 4 + index) % 5, observers))
    interference = record("distractor-partial-observation-32", "categorical",
                          [[at((actor + 2 * obj) % 5) for obj in range(4)] for actor in range(4)],
                          distractors, [0, 1, 2, 3])
    scenarios = []
    for base in (coverage, interference):
        scenarios += [base,
                      rename(base, "-rename-a", [1, 3, 0, 2], [2, 0, 3, 1, 4]),
                      rename(base, "-rename-b", [3, 2, 1, 0], [3, 2, 1, 0, 4])]
    mixtures = [[0.4, 0.3, 0.2, 0.1, 0.0], [0.05, 0.1, 0.4, 0.2, 0.25],
                [0.15, 0.15, 0.1, 0.15, 0.45], [0.2] * 5]
    rows = [[mixtures[(actor + obj) % 4][:] for obj in range(4)] for actor in range(4)]
    scenarios.append(record("mixed-unseen-preserve-32", "distributed", rows,
                            [event(index % 4, (index // 4) % 5, ()) for index in range(32)]))
    scenarios.append(record("mixed-unrelated-preserve-32", "distributed", rows,
                            [event(3, index % 5, (0, 1, 2, 3)) for index in range(32)]))
    near_tie = [0.21, 0.2, 0.2, 0.2, 0.19]
    near_rows = [[near_tie[(actor + obj) % 5:] + near_tie[:(actor + obj) % 5]
                  for obj in range(4)] for actor in range(4)]
    scenarios.append(record("near-tie-repeated-preserve-32", "distributed", near_rows,
                            [event(index % 4, index % 5, ()) for index in range(32)]))
    protocol = {"schema": "cellforge-belief-post-selection-stress/1",
                "classification": "post-selection stress, not untouched final test",
                "parameter_hash": WEIGHT_HASH, "parent_result_sha256": PARENT_RESULT,
                "max_events": 32, "actors": 4, "objects": 4, "scenario_count": 9,
                "categorical_scenarios": 6, "distributed_scenarios": 3,
                "total_every_slot_predictions": 4608, "random_search": False,
                "scenario_construction": "deterministic declared schedules; no generated-candidate search or stress-feedback tuning",
                "novel_pair_scope": "All 16 actor/object pairs explicitly observed in the coverage family; 12 pairs use actor or object indices outside original 2x2 search domain. They are not claimed unseen in the earlier 4x4 development panel.",
                "metrics": "Every one of 16 slots after every event; whole-scenario top-1 and distribution-tolerance passes reported separately.",
                "tie_policy": "lowest index", "probability_tolerance": 0.05,
                "identity_probe_tolerance": 1e-12, "renaming_tolerance": 1e-12,
                "unknown_renaming": "unknown index 4 remains fixed; locations 0..3 and actors permute consistently",
                "runtime_cap_seconds": 120, "internal_deadline_seconds": 105,
                "event_yield_seconds": 0.01, "worker_processes": 1,
                "error_policy": "No repair/fallback; runtime failure ends that scenario with unexecuted slots unresolved.",
                "continuous_limit": "Sharp categorical memory may preserve top-1 yet distort mixed probability mass; score against unchanged full reference distributions, not rounded one-hot targets.",
                "author_calls": 0, "optimization_steps": 0, "auto_admission": False,
                "original_panels_changed": False}
    write("SCENARIOS.json", encoded(scenarios))
    write("PROTOCOL.json", encoded(protocol))
    files = sorted(path for path in ROOT.rglob("*") if path.is_file())
    if any("__pycache__" in path.parts for path in files):
        raise ValueError("use python -B; unexpected bytecode")
    manifest = {"schema": "cellforge-belief-stress-freeze/1", "pre_stress_evaluation": True,
                "parent_freeze_sha256": PARENT_FREEZE, "parameter_hash": WEIGHT_HASH,
                "files": {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in files}}
    raw = encoded(manifest)
    write("FREEZE.json", raw)
    write("FREEZE.sha256", (digest(raw) + "\n").encode())
    print(json.dumps({"freeze_sha256": digest(raw), "scenarios": len(scenarios),
                      "events_each": 32, "every_slot_predictions": 4608,
                      "model_evaluated_during_preparation": False}))


def run(anchor):
    started = time.monotonic()
    raw = (ROOT / "FREEZE.json").read_bytes()
    if digest(raw) != anchor:
        raise ValueError("stress freeze differs from pre-evaluation anchor")
    manifest = json.loads(raw)
    for name, expected in manifest["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path.read_bytes()) != expected:
            raise ValueError("frozen stress artifact changed: " + name)
    spec = importlib.util.spec_from_file_location("stress_belief", ROOT / "runtime/belief_cell.py")
    b = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = b
    spec.loader.exec_module(b)
    cell = b.BeliefCell.from_dict(json.loads((ROOT / "WEIGHTS.json").read_bytes()))
    if cell.parameter_hash != WEIGHT_HASH:
        raise ValueError("successful candidate coefficients changed")
    protocol = json.loads((ROOT / "PROTOCOL.json").read_bytes())
    records = json.loads((ROOT / "SCENARIOS.json").read_bytes())
    if len(records) != 9 or any(len(record["events"]) != 32 for record in records):
        raise ValueError("declared scenario/event bounds differ")
    results, traces = [], {}
    for record in records:
        if time.monotonic() - started > protocol["internal_deadline_seconds"]:
            write("STOPPED.json", encoded({"status": "internal_time_cap", "completed_scenarios": len(results)}))
            raise TimeoutError("bounded stress deadline reached; no further scoring")
        initial = b.BeliefState(4, 4, tuple(tuple(b.Distribution(tuple(value)) for value in row)
                                           for row in record["initial_beliefs"]))
        events = tuple(b.MoveEvent(e["object_id"], e["destination"], tuple(e["observed_by"]))
                       for e in record["events"])
        scenario = b.Scenario(b.WorldState(tuple(record["initial_world"])), initial, events)
        predicted, reference = initial, initial
        counts = {key: 0 for key in ("predictions", "categorical_errors", "probability_errors", "identity_errors", "runtime_errors")}
        l1_values, errors, per_event, full_trace = [], [], [], []
        max_error, first_distribution_failure = 0.0, None
        for index, event in enumerate(scenario.events):
            if time.monotonic() - started > protocol["internal_deadline_seconds"]:
                write("STOPPED.json", encoded({"status": "internal_time_cap", "completed_scenarios": len(results), "event": index}))
                raise TimeoutError("bounded stress deadline reached")
            try:
                predicted = b.apply_event(cell, predicted, event)
                reference = b.reference_step(reference, event)
            except (b.BeliefError, ArithmeticError) as error:
                counts["runtime_errors"] += 1
                errors.append({"event": index, "runtime_error": str(error)})
                break
            event_errors = {"event": index, "categorical_errors": 0, "probability_errors": 0,
                            "identity_errors": 0, "max_abs_error": 0.0}
            event_trace = []
            for actor in range(4):
                row = []
                for obj in range(4):
                    actual, expected = predicted.beliefs[actor][obj].values, reference.beliefs[actor][obj].values
                    row.append(list(actual))
                    maximum = max(abs(a - e) for a, e in zip(actual, expected))
                    categorical = max(range(5), key=actual.__getitem__) != max(range(5), key=expected.__getitem__)
                    distribution_error = maximum > protocol["probability_tolerance"]
                    identity_error = maximum > protocol["identity_probe_tolerance"]
                    counts["predictions"] += 1
                    for name, value in (("categorical_errors", categorical), ("probability_errors", distribution_error), ("identity_errors", identity_error)):
                        counts[name] += int(value)
                        event_errors[name] += int(value)
                    l1_values.append(math.fsum(abs(a - e) for a, e in zip(actual, expected)))
                    max_error = max(max_error, maximum)
                    event_errors["max_abs_error"] = max(event_errors["max_abs_error"], maximum)
                    if categorical or distribution_error:
                        failure = {"event": index, "actor": actor, "object": obj,
                                   "categorical_error": categorical, "probability_error": distribution_error,
                                   "max_abs_error": maximum, "expected": list(expected), "actual": list(actual)}
                        errors.append(failure)
                        if distribution_error and first_distribution_failure is None:
                            first_distribution_failure = failure
                event_trace.append(row)
            full_trace.append(event_trace)
            per_event.append(event_errors)
            time.sleep(protocol["event_yield_seconds"])
        unresolved = 512 - counts["predictions"]
        traces[record["scenario_id"]] = full_trace
        results.append({"scenario_id": record["scenario_id"], "cohort": record["cohort"],
                        "slot_count": 512, **counts, "unresolved": unresolved,
                        "categorical_scenario_pass": counts["categorical_errors"] == 0 and unresolved == 0,
                        "distribution_scenario_pass": counts["probability_errors"] == 0 and unresolved == 0,
                        "mean_l1": math.fsum(l1_values) / len(l1_values) if l1_values else None,
                        "max_abs_error": max_error, "first_distribution_failure": first_distribution_failure,
                        "events": per_event, "failures": errors})
    renamings = []
    for record in records:
        if "renamed_from" not in record:
            continue
        original, transformed = traces[record["renamed_from"]], traces[record["scenario_id"]]
        differences = [abs(original[t][actor][obj][location] - transformed[t][record["actor_permutation"][actor]][obj][record["location_permutation"][location]])
                       for t in range(min(len(original), len(transformed)))
                       for actor in range(4) for obj in range(4) for location in range(5)]
        maximum = max(differences) if differences else None
        renamings.append({"scenario_id": record["scenario_id"], "base": record["renamed_from"],
                          "max_abs_equivariance_error": maximum,
                          "pass": len(original) == len(transformed) == 32 and maximum <= protocol["renaming_tolerance"]})
    cohorts = {}
    for cohort in ("categorical", "distributed"):
        subset = [result for result in results if result["cohort"] == cohort]
        cohorts[cohort] = {"scenarios": len(subset),
                          "categorical_scenarios_passed": sum(value["categorical_scenario_pass"] for value in subset),
                          "distribution_scenarios_passed": sum(value["distribution_scenario_pass"] for value in subset),
                          **{name: sum(value[name] for value in subset) for name in
                             ("slot_count", "predictions", "categorical_errors", "probability_errors", "identity_errors", "runtime_errors", "unresolved")},
                          "max_abs_error": max(value["max_abs_error"] for value in subset)}
    result = {"schema": "cellforge-belief-post-selection-stress-result/1",
              "classification": protocol["classification"], "freeze_sha256": anchor,
              "parameter_hash": cell.parameter_hash, "elapsed_seconds": time.monotonic() - started,
              "cohorts": cohorts, "renamings": renamings, "scenarios": results,
              "author_calls": 0, "optimization_steps": 0, "admitted": False,
              "original_panels_changed": False,
              "identity_limit": "Distribution errors are measured against the original mixed probability mass; categorical correctness does not establish general probability identity."}
    write("RESULT.json", encoded(result))
    print(json.dumps({"freeze_sha256": anchor, "result_sha256": digest((ROOT / "RESULT.json").read_bytes()),
                      "elapsed_seconds": result["elapsed_seconds"], "cohorts": cohorts,
                      "renamings": renamings,
                      "scenario_summaries": [{key: value for key, value in scenario.items() if key not in ("events", "failures")}
                                             for scenario in results]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    evaluate = commands.add_parser("run")
    evaluate.add_argument("--expected-freeze", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run(args.expected_freeze)
