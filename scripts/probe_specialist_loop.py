"""Bounded synthetic tool-use probe. Never reads benchmark or sealed targets."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np

from cellforge.ir.graph import Graph
from cellforge.runtime.specialists import (run_model_with_specialist,
    build_typed_observation_policy, typed_observation_action)


def probe_typed(destination):
    graph = build_typed_observation_policy(7)
    artifact = destination.parent / "typed-policy-artifact"
    if artifact.exists():
        raise SystemExit("refuse to overwrite typed artifact")
    graph.save(artifact)
    restored = Graph.load(artifact)
    rng = np.random.default_rng(90218)
    transferred = control = withheld = invalid = calls = 0
    for _ in range(128):
        payload = rng.normal(size=7)
        response = typed_observation_action(restored, [1, 1, 1], payload)
        zero = typed_observation_action(restored, [1, 1, 1], np.zeros(7))
        transferred += response["type"] == "respond" and np.array_equal(response["response_values"], payload)
        control += np.array_equal(zero["response_values"], payload)
        withheld += typed_observation_action(restored, [0, 0, 0], payload)["type"] == "abstain"
        invalid += typed_observation_action(restored, [1, 1, 0], payload)["type"] == "abstain"
        calls += typed_observation_action(restored, [1, 0, 0], np.zeros(7))["type"] == "call"
    source_paths = [Path(__file__), Path(__file__).parents[1] / "src/cellforge/runtime/specialists.py"]
    receipt = {"protocol": "typed-observation-transfer/1", "policy_hash": restored.hash(),
        "scope": "finite-state typed graph baseline; generic unseen numeric payload transfer; not LLM reasoning or AGI",
        "sources": {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "seed": 90218, "limits": {"payloads": 128, "width": 7, "policy_evaluations": 640},
        "aggregate": {"observation_transfer_exact": int(transferred), "zero_observation_exact": int(control),
                      "no_request_abstentions": int(withheld), "invalid_observation_abstentions": int(invalid),
                      "missing_observation_calls": int(calls)},
        "interpretation": "typed component routes and projects observations as authored; actual text policy remains unproven"}
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--specialist", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["language", "typed"], default="language")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise SystemExit("refuse to overwrite evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "typed":
        probe_typed(destination)
        return
    language = Graph.load(args.language)
    grid = Graph.load(args.specialist)
    prompts = [
        'Respond with the word hello.',
        'Use arc_grid with demonstrations [{"input":[[0]],"output":[[1]]}] and test_input [[2]]. Then report its prediction.',
        'Use arc_grid with demonstrations [{"input":[[4,4]],"output":[[7,7]]}] and test_input [[4],[4]]. Then report its prediction.',
    ]
    started = time.monotonic()
    results = [run_model_with_specialist(language, p, grid, max_steps=1,
                max_action_tokens=48, max_context_bytes=4096) for p in prompts]
    source_paths = [Path(__file__), Path(__file__).parents[1] / "src/cellforge/runtime/specialists.py"]
    receipt = {
        "protocol": "synthetic-specialist-policy-probe/1",
        "scope": "three authored synthetic prompts; not BFCL, ARC-AGI, reasoning benchmark or frontier comparison",
        "policy": "actual language graph greedy generation; no action repair or scripted routing",
        "language_model_hash": language.hash(), "specialist_hash": grid.hash(),
        "sources": {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "prompt_sha256": hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
        "limits": {"prompts": 3, "steps_per_prompt": 1, "tokens_per_action": 48, "context_bytes": 4096},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "elapsed_seconds": time.monotonic() - started,
        "aggregate": {"responded": sum(r["status"] == "responded" for r in results),
                      "tool_calls": sum(r["tool_calls"] for r in results),
                      "invalid_actions": sum(r["invalid_actions"] for r in results),
                      "generated_tokens": sum(r["generated_tokens"] for r in results)},
        "decision": "preserve protocol scaffolding; language policy competence requires independent evidence",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
