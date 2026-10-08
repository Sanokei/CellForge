"""Training-only necessary conditions for the unchanged, fixed-T8 NCA.

No model execution, weight construction, optimization, task selection, or network
access. Inputs must be explicitly derived demo-only payloads. A negative result
is a mathematical obstruction; a pass is not proof of capacity or generalization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

RADIUS = 8


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_grid(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        raise ValueError("grid height must be 1..30")
    if not isinstance(value[0], list) or not 1 <= len(value[0]) <= 30:
        raise ValueError("grid width must be 1..30")
    width = len(value[0])
    if any(not isinstance(row, list) or len(row) != width for row in value):
        raise ValueError("grid must be rectangular")
    if any(type(color) is not int or not 0 <= color <= 9
           for row in value for color in row):
        raise ValueError("grid colors must be integers 0..9")
    return len(value), width


def validate_task(task):
    if not isinstance(task, dict) or set(task) != {"task_id", "train"}:
        raise ValueError("only task_id and train are allowed; no query/test fields")
    task_id = task["task_id"]
    if (not isinstance(task_id, str) or len(task_id) != 8
            or any(c not in "0123456789abcdef" for c in task_id)):
        raise ValueError("eight-character task ID required")
    if not isinstance(task["train"], list) or not task["train"]:
        raise ValueError("nonempty demonstration list required")
    for pair in task["train"]:
        if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
            raise ValueError("only demonstration input/output fields allowed")
        validate_grid(pair["input"])
        validate_grid(pair["output"])


def neighborhood(grid, y, x):
    """Exact radius-eight input+mask encoding; -1 is outside, 0 is black.

    Including the entire square is a conservative dependency superset. No
    mask-free color-only encoding is used, since border geometry is accessible.
    """
    height, width = len(grid), len(grid[0])
    return tuple(grid[yy][xx] if 0 <= yy < height and 0 <= xx < width else -1
                 for yy in range(y - RADIUS, y + RADIUS + 1)
                 for xx in range(x - RADIUS, x + RADIUS + 1))


def analyze(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("nonempty task panel required")
    for task in tasks:
        validate_task(task)
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate tasks are not separate panel observations")
    tasks = sorted(tasks, key=lambda task: task["task_id"])
    shape_failures = []
    neighborhoods = defaultdict(list)
    whole_inputs = defaultdict(list)
    summaries = {}
    total_pairs = total_pixels = same_size_pixels = 0
    for task in tasks:
        task_id = task["task_id"]
        summary = {"task_id": task_id, "demonstrations": len(task["train"]),
                   "shape_mismatch_pairs": [], "within_task_conflict_classes": 0,
                   "cross_task_conflict_partners": []}
        summaries[task_id] = summary
        for index, pair in enumerate(task["train"]):
            total_pairs += 1
            source, target = pair["input"], pair["output"]
            source_shape, target_shape = validate_grid(source), validate_grid(target)
            total_pixels += target_shape[0] * target_shape[1]
            ref = {"task_id": task_id, "demo_index": index}
            whole_inputs[canonical(source)].append(dict(ref, target=target))
            if source_shape != target_shape:
                summary["shape_mismatch_pairs"].append(index)
                shape_failures.append(dict(ref, input_shape=list(source_shape),
                                           output_shape=list(target_shape)))
                continue
            same_size_pixels += target_shape[0] * target_shape[1]
            for y, row in enumerate(target):
                for x, color in enumerate(row):
                    neighborhoods[neighborhood(source, y, x)].append(
                        dict(ref, y=y, x=x, required_color=color))
    conflicts = []
    edges = set()
    unavoidable_pixel_errors = 0
    for signature, occurrences in sorted(neighborhoods.items()):
        counts = Counter(row["required_color"] for row in occurrences)
        unavoidable_pixel_errors += len(occurrences) - max(counts.values())
        if len(counts) < 2:
            continue
        grouped = defaultdict(set)
        for row in occurrences:
            grouped[row["task_id"]].add(row["required_color"])
        for task_id, colors in grouped.items():
            if len(colors) > 1:
                summaries[task_id]["within_task_conflict_classes"] += 1
        involved = sorted(grouped)
        for i, left in enumerate(involved):
            for right in involved[i + 1:]:
                if any(a != b for a in grouped[left] for b in grouped[right]):
                    edges.add((left, right))
        conflicts.append({"neighborhood_sha256": digest(signature),
                          "required_color_counts": dict(sorted(counts.items())),
                          "occurrences": occurrences})
    demo_conflicts = []
    for source, occurrences in sorted(whole_inputs.items()):
        by_target = defaultdict(list)
        for occurrence in occurrences:
            by_target[canonical(occurrence["target"])].append(
                {key: occurrence[key] for key in ("task_id", "demo_index")})
        if len(by_target) > 1:
            demo_conflicts.append({"input_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                   "distinct_required_outputs": [
                                       {"output_sha256": hashlib.sha256(target.encode()).hexdigest(),
                                        "occurrences": refs}
                                       for target, refs in sorted(by_target.items())]})
    for left, right in sorted(edges):
        summaries[left]["cross_task_conflict_partners"].append(right)
        summaries[right]["cross_task_conflict_partners"].append(left)
    for summary in summaries.values():
        blocked = bool(summary["shape_mismatch_pairs"] or summary["within_task_conflict_classes"])
        summary["individual_necessary_conditions"] = "ruled_out" if blocked else "not_ruled_out"
    return {
        "schema": "cellforge-nca-training-capacity-diagnostic/1",
        "radius": RADIUS, "iterations": 8,
        "input_payload_sha256": digest(tasks),
        "tasks": len(tasks), "demonstrations": total_pairs,
        "target_pixels": total_pixels, "same_size_target_pixels": same_size_pixels,
        "shape_mismatch_pairs": shape_failures,
        "neighborhood_conflict_classes": conflicts,
        "demonstration_invariance_conflicts": demo_conflicts,
        "cross_task_conflict_edges": [list(edge) for edge in sorted(edges)],
        "same_size_unavoidable_pixel_errors_lower_bound": unavoidable_pixel_errors,
        "task_status": [summaries[key] for key in sorted(summaries)],
        "whole_panel_necessary_conditions": "ruled_out" if shape_failures or conflicts or demo_conflicts else "not_ruled_out",
        "qualification": "Necessary conditions only; no NCA predictions, capacity proof, author calls, weight proposals, or evaluation outputs.",
    }


def load_frozen_panel(repo, inventory_path, materialization_path):
    inventory_raw = inventory_path.read_bytes()
    inventory = json.loads(inventory_raw)
    materialization_raw = materialization_path.read_bytes()
    materialization = json.loads(materialization_raw)
    inventory_hash = hashlib.sha256(inventory_raw).hexdigest()
    if materialization["input_inventory_sha256"] != inventory_hash:
        raise ValueError("frozen inventory hash mismatch")
    if materialization["base_commit"] != inventory["base_commit"]:
        raise ValueError("materialization source commit mismatch")
    entries = {row["task_id"]: row for row in inventory["entries"]}
    rows = materialization["rows"]
    if (len(entries) != len(inventory["entries"]) or len(rows) != len(entries)
            or {row["task_id"] for row in rows} != set(entries)):
        raise ValueError("materialized panel must exactly match frozen task inventory")
    tasks, input_files = [], []
    for row in sorted(rows, key=lambda value: value["task_id"]):
        entry = entries[row["task_id"]]
        if (row["source_packet_path"] != entry["path"]
                or row["source_git_blob_sha1"] != entry["git_blob_sha1"]):
            raise ValueError("source packet provenance mismatch")
        path = (repo / row["derived_path"]).resolve()
        if not path.is_relative_to(repo.resolve()):
            raise ValueError("derived input path escapes repository")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["derived_sha256"]:
            raise ValueError("immutable demonstration content changed")
        task = json.loads(raw)
        validate_task(task)
        if task["task_id"] != row["task_id"]:
            raise ValueError("payload task identity mismatch")
        tasks.append(task)
        input_files.append({"path": row["derived_path"], "sha256": row["derived_sha256"]})
    return tasks, {"input_inventory_sha256": inventory_hash,
                   "materialization_sha256": hashlib.sha256(materialization_raw).hexdigest(),
                   "base_commit": inventory["base_commit"], "input_files": input_files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("fresh diagnostic output path required")
    tasks, provenance = load_frozen_panel(args.repo, args.inventory, args.materialization)
    result = analyze(tasks)
    result.update(provenance)
    result["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
