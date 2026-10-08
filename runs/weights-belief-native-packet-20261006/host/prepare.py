"""One-time host preparation. No model call, fitting or solution weights."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SEARCH_SEEDS = (2026100611, 2026100612, 2026100613, 2026100614)
DEVELOPMENT_SEEDS = (2026100621, 2026100622, 2026100623, 2026100624)
RANDOM_SEED = 2026100631
PACKET_ID = "belief-native-20261006-one"


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def load_runtime(path):
    spec = importlib.util.spec_from_file_location("frozen_belief", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scenario(b, seed, split):
    """Original event families: balanced query categories, not corpus tasks."""
    rng = random.Random(seed)
    size, cycles = (2, 2) if split == "search" else (4, 4)
    world = b.WorldState(tuple(rng.randrange(5) for _ in range(size)))
    initial = b.BeliefState(size, size, tuple(tuple(
        b.Distribution.at(rng.randrange(5)) for _ in range(size)) for _ in range(size)))
    state, events, queries, examples = initial, [], [], []
    for cycle in range(cycles):
        actor, obj = rng.randrange(size), rng.randrange(size)
        other_actor, other_obj = (actor + 1) % size, (obj + 1) % size
        old_location = max(range(5), key=state.beliefs[actor][obj].values.__getitem__)
        known = rng.choice([location for location in range(4) if location != old_location])
        unseen = (known + rng.randrange(1, 4)) % 4
        seen_by = (actor,) if split == "search" else tuple(sorted((actor, (actor + 2) % size)))
        hidden_by = (other_actor,) if split == "search" else tuple(sorted((other_actor, (actor + 3) % size)))
        block = [("seen", b.MoveEvent(obj, known, seen_by)),
                 ("unseen", b.MoveEvent(obj, unseen, hidden_by)),
                 ("unrelated", b.MoveEvent(other_obj, rng.randrange(5), seen_by)),
                 ("unknown", b.MoveEvent(obj, b.UNKNOWN, seen_by))]
        # Longer development histories reverse the two preservation contexts
        # on alternating cycles; no search event sequence is copied over.
        if split == "development" and cycle % 2:
            block[1], block[2] = block[2], block[1]
        for category, event in block:
            index = len(events)
            observed = actor in event.observed_by
            inputs = b.KernelInput(state.beliefs[actor][obj],
                b.Distribution.at(event.destination if observed else b.UNKNOWN),
                obj == event.object_id, observed)
            expected = b.reference_update(inputs)
            events.append({"object_id": event.object_id, "destination": event.destination,
                           "observed_by": list(event.observed_by)})
            queries.append({"event_index": index, "actor_id": actor, "object_id": obj,
                            "category": category, "expected": list(expected.values)})
            examples.append({"scenario_id": f"{split}-{seed}", "event_index": index,
                             "category": category, "input": list(inputs.vector()),
                             "expected": list(expected.values)})
            state = b.reference_step(state, event)
    return {"scenario_id": f"{split}-{seed}", "seed": seed,
            "family": "two-slot-two-cycle" if split == "search" else "four-slot-four-cycle-alternating",
            "actor_count": size, "object_count": size,
            "initial_world": list(world.object_locations),
            "initial_beliefs": [[list(value.values) for value in row] for row in initial.beliefs],
            "events": events, "queries": queries}, examples


def main():
    if (ROOT / "FREEZE.json").exists() or (ROOT / "author").exists():
        raise SystemExit("Preparation already started/frozen; never overwrite or regenerate this packet.")
    runtime = ROOT / "host/runtime/belief_cell.py"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    source = REPO / "scripts/belief_cell.py"
    with source.open("rb") as source_stream, runtime.open("xb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream)
    b = load_runtime(runtime)
    splits, search_examples = {}, []
    for split, seeds in (("search", SEARCH_SEEDS), ("development", DEVELOPMENT_SEEDS)):
        values = []
        for seed in seeds:
            record, examples = scenario(b, seed, split)
            values.append(record)
            if split == "search":
                search_examples.extend(examples)
        splits[split] = values
        write(ROOT / f"host/{split.upper()}.json", encoded(values))
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema",
              "title": "One literal belief-cell response", "type": "object",
              "additionalProperties": False, "required": ["schema", "packet_id", "parameters", "rationale"],
              "properties": {
                  "schema": {"const": "cellforge-belief-native-response/1"},
                  "packet_id": {"const": PACKET_ID},
                  "parameters": {"type": "array", "minItems": 293, "maxItems": 293,
                                 "items": {"type": "number", "minimum": -1.7976931348623157e308,
                                           "maximum": 1.7976931348623157e308}},
                  "rationale": {"type": "string", "minLength": 1, "maxLength": 4000}}}
    packet = {"packet_id": PACKET_ID, "task": "Author one complete set of 293 literal belief-cell MLP parameters.",
              "allowed_inputs": ["this packet", "the matching RESPONSE_SCHEMA.json"],
              "instructions": [
                  "Return exactly one JSON object matching the response schema. No Markdown or executable code.",
                  "Write the numeric literals directly. Do not run code, fit/train/search weights, call tools or use external data.",
                  "Do not read the repository, runtime/reference implementation, evaluator, host files, development data, seeds or outcomes.",
                  "Only the declared search examples below are provided. No evaluator access or feedback is authorized.",
                  "The host applies the same kernel to every actor/object slot and carries all predictions forward; it does not repair errors.",
                  "Representations are structured events only, not language, emotions, demographics, intentions or full social awareness.",
                  "This is one bounded literal-authoring attempt, not a claim of learned or general intelligence."],
              "parameter_schema": b.parameter_schema(),
              "forward_semantics": {
                  "input": "x[0:5]=old distribution, x[5:10]=observed-new distribution, x[10]=object-match, x[11]=observation",
                  "locations": "0,1,2,3,unknown=4; bits are numeric 0 or 1",
                  "mask": "When observation=0, observed-new is [0,0,0,0,1]. World locations never enter the kernel.",
                  "hidden": "h[j]=tanh(sum_i parameters[12*j+i]*x[i] + parameters[192+j]), j=0..15",
                  "logits": "z[k]=sum_j parameters[208+16*k+j]*h[j]+parameters[288+k], k=0..4",
                  "output": "p[k]=exp(z[k]-max(z))/sum_l exp(z[l]-max(z)); finite affine overflow is an error",
                  "intended_behavior": "Use observed-new when both bits are 1; otherwise preserve old. No host bypass implements this for the model.",
                  "score": "Primary: recursive rollout top-1 correctness; exact ties choose lowest index. Diagnostic: max probability error <=0.05 and mean L1; all are frozen before authoring."},
              "search_family": "Four two-actor/two-object scenarios, eight events each. The 32 query examples below are teacher-forced reference targets, balanced eight per category.",
              "search_examples": search_examples,
              "response_schema_file": "RESPONSE_SCHEMA.json"}
    write(ROOT / "author/RESPONSE_SCHEMA.json", encoded(schema))
    write(ROOT / "author/AUTHORING_PACKET.json", encoded(packet))
    protocol = {"schema": "cellforge-belief-native-protocol/1", "packet_id": PACKET_ID,
                "pre_author_freeze": True, "attempt_limit": 1, "parameter_count": 293,
                "search_seeds": list(SEARCH_SEEDS), "development_seeds": list(DEVELOPMENT_SEEDS),
                "search_family": "two-slot-two-cycle", "development_family": "four-slot-four-cycle-alternating",
                "query_categories": ["seen", "unseen", "unrelated", "unknown"],
                "search_queries": 32, "development_queries": 64,
                "category_weighting": "equal query counts; all actor/object slots still execute every event",
                "primary_metric": "recursive-rollout top-1 query accuracy; scenario solved iff every query correct and no runtime error",
                "tie_policy": "lowest index", "probability_tolerance": 0.05,
                "diagnostics": ["teacher-forced top-1", "probability error >0.05 count", "mean L1", "per-category and per-scenario outcomes"],
                "error_policy": "no repair/fallback; runtime failures recorded; remaining rollout queries unresolved and included in denominator",
                "random_seed": RANDOM_SEED, "random_distribution": "one uniform[-0.25,0.25] draw at each of 293 positions, stdlib Random",
                "charged_positions_per_attempt": {"native": 293, "random": 293},
                "zero_and_reference": "diagnostic controls, not charged proposals; reference is hand-coded oracle",
                "author_credit_usage": None, "author_credit_status": "unknown, not zero or free",
                "source_thread_id": None, "source_turn_id": None,
                "source_status": "awaiting parent-supplied native receipt; no author has been invoked by this preparation",
                "no_auto_admission": True, "no_retry": True, "development_feedback_to_author": False,
                "access_boundary": "parent must provide only author/ files through a tool-disabled native call; folders/hashes are not an OS sandbox",
                "limitations": ["original synthetic scenarios only", "single author attempt", "no natural-language competence", "no independently held-out generalization claim", "no ONE_STANDARD qualification"]}
    write(ROOT / "host/PROTOCOL.json", encoded(protocol))
    standard = {"arms": {"llm": {}, "random": {}}, "seeds": [RANDOM_SEED],
                "seeds_declared_before_run": True, "budget_per_arm": 0,
                "niches": ["structured-belief-update"], "founders": [], "genomes": [], "proposals": [],
                "scientific_acceptance": False, "status": "pre-author packet; no proposal consumed",
                "limitations": ["one predeclared control seed; below three-seed milestone", "author credit usage unknown", "no LLM-authored founder yet", "future attempt receipts are append-only host artifacts, not claims that this frozen declaration passed the standard"]}
    write(ROOT / "STANDARD.json", encoded(standard))
    paths = sorted(path for path in ROOT.rglob("*") if path.is_file())
    if any("__pycache__" in path.parts for path in paths):
        raise SystemExit("Unexpected bytecode; prepare using python -B only.")
    manifest = {"schema": "cellforge-belief-freeze/1", "packet_id": PACKET_ID,
                "source_path_at_freeze": "scripts/belief_cell.py",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "files": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    raw = encoded(manifest)
    write(ROOT / "FREEZE.json", raw)
    digest = hashlib.sha256(raw).hexdigest()
    write(ROOT / "FREEZE.sha256", (digest + "\n").encode("ascii"))
    print(json.dumps({"freeze_sha256": digest, "packet": str(ROOT / "author/AUTHORING_PACKET.json"),
                      "schema": str(ROOT / "author/RESPONSE_SCHEMA.json"),
                      "search_queries": 32, "development_queries": 64, "model_calls": 0}))


if __name__ == "__main__":
    main()
