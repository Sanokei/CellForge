"""Derive separately declared query counts from exhaustive saved slot failures."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(anchor):
    raw = (ROOT / "SELECTED_QUERY_FREEZE.json").read_bytes()
    if sha(raw) != anchor:
        raise ValueError("selected-query freeze mismatch")
    manifest = json.loads(raw)
    for name, expected in manifest["files"].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError("selected-query declaration changed")
    raw = (ROOT / "FREEZE.json").read_bytes()
    if sha(raw) != manifest["every_slot_freeze_sha256"]:
        raise ValueError("every-slot freeze mismatch")
    for name, expected in json.loads(raw)["files"].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError("every-slot frozen artifact changed")
    result_raw = (ROOT / "RESULT.json").read_bytes()
    result = json.loads(result_raw)
    if result["freeze_sha256"] != manifest["every_slot_freeze_sha256"]:
        raise ValueError("results not bound to declared stress")
    records = json.loads((ROOT / "SCENARIOS.json").read_bytes())
    reports = []
    for record, outcome in zip(records, result["scenarios"]):
        if record["scenario_id"] != outcome["scenario_id"]:
            raise ValueError("scenario ordering changed")
        failures = {(f["event"], f["actor"], f["object"]): f for f in outcome["failures"] if "actor" in f}
        queries = []
        completed = len(outcome["events"])
        for index in range(32):
            actor, obj = (index // 4) % 4, index % 4
            if "renamed_from" in record:
                actor = record["actor_permutation"][actor]
            failure = failures.get((index, actor, obj), {})
            queries.append({"event": index, "actor": actor, "object": obj,
                            "resolved": index < completed,
                            "categorical_error": failure.get("categorical_error", False),
                            "probability_error": failure.get("probability_error", False)})
        predictions = sum(q["resolved"] for q in queries)
        categorical = sum(q["categorical_error"] for q in queries)
        probability = sum(q["probability_error"] for q in queries)
        reports.append({"scenario_id": record["scenario_id"], "cohort": record["cohort"],
                        "queries": 32, "predictions": predictions, "correct": predictions - categorical,
                        "categorical_errors": categorical, "probability_errors": probability,
                        "runtime_errors": outcome["runtime_errors"], "unresolved": 32 - predictions,
                        "categorical_scenario_pass": predictions == 32 and categorical == 0,
                        "distribution_scenario_pass": predictions == 32 and probability == 0,
                        "query_outcomes": queries})
    cohorts = {}
    for cohort in ("categorical", "distributed"):
        subset = [r for r in reports if r["cohort"] == cohort]
        cohorts[cohort] = {key: sum(r[key] for r in subset) for key in
                          ("queries", "predictions", "correct", "categorical_errors", "probability_errors", "runtime_errors", "unresolved",
                           "categorical_scenario_pass", "distribution_scenario_pass")}
    output = {"schema": "cellforge-belief-stress-selected-query-result/1",
              "classification": "post-selection known-rule expressivity stress; no held-out independence or ONE_STANDARD claim",
              "selected_query_freeze_sha256": anchor, "every_slot_result_sha256": sha(result_raw),
              "derivation": "One predeclared slot per event. Exact counts derived from exhaustive every-slot failure records; no new model execution, feedback or scoring-rule edits.",
              "cohorts": cohorts, "scenarios": reports}
    with (ROOT / "SELECTED_QUERY_RESULT.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cohorts": cohorts, "sha256": sha((ROOT / "SELECTED_QUERY_RESULT.json").read_bytes())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-freeze", required=True)
    main(parser.parse_args().expected_freeze)
