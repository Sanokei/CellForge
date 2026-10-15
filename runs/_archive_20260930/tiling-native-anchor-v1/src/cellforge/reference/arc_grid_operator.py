"""ARC-AGI-2 grid scoring contract for future authored grid artifacts.

This module scores a predictor that consumes demonstrations and a test input.
It does not adapt a text language model into a grid predictor. Keep public
evaluation task contents on the evaluator; publish only aggregate results.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import islice
from typing import Any

from .bench_operator import wilson

Grid = tuple[tuple[int, ...], ...]
Predictor = Callable[[Sequence[dict[str, Any]], list[list[int]]], Iterable[Any]]


def validate_grid(value: Any) -> Grid:
    """Validate an ARC grid, preserving dimensions for exact comparison."""
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        raise ValueError("grid height must be 1..30")
    width = None
    rows = []
    for row in value:
        if not isinstance(row, list) or not 1 <= len(row) <= 30:
            raise ValueError("grid rows must be lists of width 1..30")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("grid must be rectangular")
        if any(type(color) is not int or not 0 <= color <= 9 for color in row):
            raise ValueError("grid colors must be integers 0..9")
        rows.append(tuple(row))
    return tuple(rows)


def score_tasks(tasks: Iterable[tuple[str, dict[str, Any]]], predictor: Predictor) -> dict[str, Any]:
    """Score exact task success with up to two attempts per test input.

    ``tasks`` must come from the evaluation operator's private dataset. The
    predictor must generate answers from demonstrations and each test input;
    lookup submissions keyed by task ID are outside this contract.
    """
    n_tasks = n_solved = n_inputs = n_correct = n_invalid = 0
    for _task_id, task in tasks:
        train = task["train"]
        test = task["test"]
        if not train or not test:
            raise ValueError("ARC task must have demonstrations and test inputs")
        for demo in train:
            validate_grid(demo["input"])
            validate_grid(demo["output"])
        solved = True
        for item in test:
            validate_grid(item["input"])
            target = validate_grid(item["output"])
            n_inputs += 1
            try:
                attempts = list(islice(predictor(train, item["input"]), 3))
                if len(attempts) > 2:
                    raise ValueError("at most two grid attempts are permitted")
                predicted = []
                for grid in attempts:
                    try:
                        predicted.append(validate_grid(grid))
                    except ValueError:
                        n_invalid += 1
                correct = target in predicted
            except (TypeError, ValueError, KeyError, IndexError):
                n_invalid += 1
                correct = False
            n_correct += int(correct)
            solved &= correct
        n_tasks += 1
        n_solved += int(solved)
    return {
        "protocol": "arc-grid/1",
        "n_tasks": n_tasks,
        "tasks_solved": n_solved,
        "task_acc": round(n_solved / n_tasks, 4) if n_tasks else 0.0,
        "task_acc_ci95": wilson(n_solved, n_tasks),
        "n_test_inputs": n_inputs,
        "test_inputs_correct": n_correct,
        "invalid_outputs": n_invalid,
    }


def main(argv=None):
    """Server operator: aggregate-only development/audit on public training.

    Public evaluation tasks are deliberately outside this CLI. Training tasks
    are partitioned by ID hash before authoring; audit is reserved for a frozen
    candidate and must be disclosed as development exposure if used later.
    """
    import argparse
    import hashlib
    import json
    import time
    from pathlib import Path

    from ..ir.graph import Graph
    from ..runtime.grid import GridPredictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--partition", choices=["development", "audit"], default="development")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("--n must be positive")
    directory = Path(args.private) / "arcagi2/data/training"
    files = list(directory.glob("*.json"))
    if not files:
        parser.error("official ARC-AGI-2 training directory is absent or empty")
    keyed = sorted((hashlib.sha256(p.stem.encode()).hexdigest(), p) for p in files)
    audit = args.partition == "audit"
    selected = [p for digest, p in keyed if (int(digest, 16) % 5 == 0) == audit][:args.n]
    if not selected:
        parser.error("partition is empty")
    manifest = hashlib.sha256()
    tasks = []
    for path in selected:
        raw = path.read_bytes()
        manifest.update(path.stem.encode() + b":" + hashlib.sha256(raw).digest())
        tasks.append((path.stem, json.loads(raw)))
    graph = Graph.load(args.artifact)
    start = time.perf_counter()
    result = score_tasks(tasks, GridPredictor(graph))
    result.update({"model_hash": graph.hash(), "label": args.label,
                   "dataset": "ARC-AGI-2 public training", "partition": args.partition,
                   "partition_protocol": "task-id-sha256-mod5/1: zero=audit, nonzero=development; hash order",
                   "selected_manifest_sha256": manifest.hexdigest(), "requested_n": args.n,
                   "seconds": round(time.perf_counter() - start, 3),
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "operator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    public = Path(args.public)
    public.mkdir(parents=True, exist_ok=True)
    with (public / "arc_grid_results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
