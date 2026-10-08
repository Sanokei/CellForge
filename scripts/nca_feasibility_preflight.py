"""Frozen, training-only T8 NCA controls. No author dispatch or optimization.

Commands: freeze (no predictions), verify (no predictions), baseline, replay.
Baseline/replay require the reviewed manifest SHA256 as an explicit argument.
Only the four predeclared arms can execute. A second seed is reserved, not drawn.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

BASE_COMMIT = "02beaf18251fca5ed1f1eb55e92f00a3e3820e88"
PANEL_DIR = "reports/nca-capacity-20261008"
ALL_IDS = ("0c786b71", "1a07d186", "3194b014", "46442a0e", "712bf12e", "984d8a3e", "bd283c4a", "c8cbb738", "ed36ccf7", "ef135b50", "f5b8619d")
COMPATIBLE_IDS = ("1a07d186", "712bf12e", "984d8a3e", "bd283c4a", "ed36ccf7", "ef135b50")
ARMS = ("identity", "neighbor_majority", "zero", "paired_random_1")
REFERENCES = ("identity", "neighbor_majority", "zero")
VOCABULARY = (-1, -0.5, -0.25, 0, 0.25, 0.5, 1)
SEEDS = (2026100801, 2026100802)
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
LIMITS = {"arm_wall_seconds": 30, "arm_cpu_seconds": 25, "demo_wall_seconds": 5,
          "address_space_bytes": 1024 * 1024 * 1024, "file_size_bytes": 8 * 1024 * 1024,
          "open_files": 64, "numerical_threads": 1, "concurrency": 1}
SOURCE_PATHS = ("scripts/nca_feasibility_preflight.py", "scripts/nca_capacity_diagnostic.py",
                "src/cellforge/__init__.py", "src/cellforge/construction/__init__.py",
                "src/cellforge/construction/nca_gene.py", "src/cellforge/ir/__init__.py",
                "src/cellforge/ir/graph.py", "src/cellforge/runtime/__init__.py",
                "src/cellforge/runtime/grid.py", "src/cellforge/runtime/interp.py",
                "src/cellforge/runtime/ops.py")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + x)))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def relative_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("artifact path escapes run")
    return path


def runtime_imports():
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source / "scripts"))
    import numpy as np
    from cellforge.construction.nca_gene import build_nca_gene
    from cellforge.ir.graph import Graph, Limits, array_hash
    from cellforge.runtime.grid import GridPredictor
    import nca_capacity_diagnostic as diagnostic
    return np, build_nca_gene, Graph, Limits, array_hash, GridPredictor, diagnostic


def graph_structure(graph):
    return {"defs": graph.defs, "entries": graph.entries, "meta": graph.meta,
            "shapes": {name: list(array.shape) for name, array in sorted(graph.params.items())}}


def control_graph(arm):
    np, build, _, limits, _, _, _ = runtime_imports()
    if arm not in ARMS:
        raise ValueError("only four frozen baseline arms permitted; second random draw prohibited")
    graph = build(iterations=8, seed="identity")
    if arm == "neighbor_majority":
        reference = build(iterations=8, seed="neighbor_majority")
        graph.params = {name: array.copy() for name, array in reference.params.items()}
    elif arm == "zero":
        graph.params = {name: np.zeros_like(array) for name, array in graph.params.items()}
    elif arm == "paired_random_1":
        rng = random.Random(SEEDS[0])
        for name in sorted(graph.params):
            array = graph.params[name]
            graph.params[name] = np.array([rng.choice(VOCABULARY) for _ in range(array.size)],
                                          dtype=np.float64).reshape(array.shape)
    graph.validate(limits(strict_microcells=True))
    active = [node["use"] for node in graph.defs["nca_step"]["nodes"] if node["use"] in graph.defs and graph.defs[node["use"]]["kind"] == "cell"]
    if len(active) != 48 or len(set(active)) != 48 or len(graph.params) != 96:
        raise ValueError("fixed active cell/parameter roster changed")
    if sum(array.size for array in graph.params.values()) != 1488:
        raise ValueError("unexpected stored position count")
    return graph


def load_panel(repo):
    *_, diagnostic = runtime_imports()
    panel = repo / PANEL_DIR
    tasks, provenance = diagnostic.load_frozen_panel(repo, panel / "INPUT_INVENTORY.json", panel / "MATERIALIZATION.json")
    if provenance["base_commit"] != BASE_COMMIT or tuple(task["task_id"] for task in tasks) != ALL_IDS:
        raise ValueError("frozen eleven-task panel changed")
    actual = diagnostic.analyze(tasks)
    recorded = read_json(panel / "RESULT.json")
    for key, value in actual.items():
        if recorded.get(key) != value:
            raise ValueError("capacity source/result mismatch: " + key)
    if recorded.get("diagnostic_source_sha256") != file_hash(repo / "scripts/nca_capacity_diagnostic.py"):
        raise ValueError("capacity diagnostic source binding failure")
    if any(recorded.get(key) != value for key, value in provenance.items()):
        raise ValueError("capacity provenance mismatch")
    compatible = [task for task in tasks if task["task_id"] in COMPATIBLE_IDS]
    checked = diagnostic.analyze(compatible)
    if checked["whole_panel_necessary_conditions"] != "not_ruled_out":
        raise ValueError("structural contradiction in restricted panel")
    remaining = [row for row in actual["task_status"] if row["task_id"] not in COMPATIBLE_IDS]
    if (len(compatible) != 6 or checked["demonstrations"] != 17 or actual["demonstrations"] != 32
            or len(actual["shape_mismatch_pairs"]) != 15
            or any(row["individual_necessary_conditions"] != "ruled_out" for row in remaining)):
        raise ValueError("capacity accounting changed")
    return tasks, compatible, actual


def future_protocol():
    return {
        "authorization": "Design only. Baseline preflight does not authorize any author/model dispatch.",
        "hypothesis": "One globally shared fixed-T8 literal tensor improves exposed training behavior after training-only counterexamples.",
        "maximum_author_calls": 2, "founders": 1, "conditional_mutations": 1,
        "mutation_eligible_only_if": "Founder is valid and has at least one failing demonstration; parent authorization still required.",
        "invalid_or_refused": "Consumes full attempt, paired full-random charge, and stops; no repair, replacement or retry.",
        "charged_positions_per_consumed_author_attempt": 1488,
        "charged_positions_per_paired_full_random_attempt": 1488,
        "maximum_positions_each_arm": 2976,
        "charge_even_if": ["invalid", "refused", "neutral", "numeric duplicate", "functional duplicate"],
        "paired_random_seeds_in_attempt_order": list(SEEDS),
        "first_random_preflight": "Evaluated once in preflight; reuse exactly this saved tensor as founder counterpart, no redraw. Preflight computation is separate from author-attempt charges.",
        "second_random": "Seed reserved; tensor not constructed or evaluated in preflight. Only evaluate if attempt two is consumed.",
        "vocabulary": list(VOCABULARY), "tensor_order": "Lexicographically sorted parameter names, then C/row-major positions; all 1488 positions replaced.",
        "random_distribution": "Independent uniform seven-value draws, Python random.Random(seed).choice(vocabulary), same shapes and vocabulary as author.",
        "fixed_references": "Identity and existing neighbor_majority/smoothing; zero separately labeled. Existing smoothing 1/9 and 1/36 coefficients are a fixed-reference exception to the candidate/random vocabulary.",
        "metrics": ["exact demonstrations /17", "exact tasks (all task demos exact) /6", "all-11-task and 32-demo coverage", "mean per-demo normalized pixel error", "runtime/validation errors", "numeric and functional duplicates", "lineage"],
        "selected_baseline_rule": "Best valid fixed reference by exact demonstrations, then exact tasks, then lower mean normalized pixel error, then identity/neighbor_majority/zero order; selected once before authoring.",
        "positive_feasibility": "Valid candidate must have strictly more exact demonstrations than the best fixed reference AND every matched random consumed so far, and preserve every demonstration solved by the selected baseline.",
        "positive_mutation_claim": "Additionally strictly more exact demonstrations than valid founder, preserving every founder-solved demonstration; save founder hash and deterministic training-only failures.",
        "error_policy": "Invalid controls or runtime errors stop before authoring; errors are null outcomes, never synthetic ordinary failures. No score comparison with invalid arms.",
        "stop_before_author": ["any fixed or paired-random baseline perfect on 17 demos", "any invalid control", "source binding failure", "restricted-panel structural contradiction", "STOP sentinel"],
        "founder_packet": "Architecture, tensor schema and training demonstrations only. No random tensors/scores or baseline selection outcomes.",
        "mutation_packet": "Same frozen task context plus founder literals and deterministic training failures ordered task ID then demo index. No random tensors/scores.",
        "lineage": "One founder, then at most one mutation with the founder tensor hash as sole parent; no crossover claim.",
        "qualification": "Exposed public-training development feasibility only; not held-out transfer, generality, ARC benchmark, admission or ONE_STANDARD qualification.",
    }


def freeze(repo, run):
    if run.exists():
        raise ValueError("fresh run directory required; never overwrite an earlier freeze")
    np, _, _, _, array_hash, _, _ = runtime_imports()
    tasks, compatible, capacity = load_panel(repo)
    run.mkdir(parents=True)
    sources = {}
    for name in SOURCE_PATHS:
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, destination)
        sources[name] = file_hash(destination)
    bindings = {}
    for name in ("INPUT_INVENTORY.json", "MATERIALIZATION.json", "RESULT.json"):
        shutil.copyfile(repo / PANEL_DIR / name, run / name)
        bindings[name] = file_hash(run / name)
    write_json(run / "PANEL.json", {"schema": "demo-only/1", "tasks": tasks})
    bindings["PANEL.json"] = file_hash(run / "PANEL.json")
    graph = control_graph("identity")
    structure = graph_structure(graph)
    write_json(run / "GRAPH_STRUCTURE.json", structure)
    bindings["GRAPH_STRUCTURE.json"] = file_hash(run / "GRAPH_STRUCTURE.json")
    tensor_schema = [{"name": name, "shape": list(array.shape), "positions": int(array.size)}
                     for name, array in sorted(graph.params.items())]
    arms = []
    for arm in ARMS:
        candidate = control_graph(arm)
        if graph_structure(candidate) != structure:
            raise ValueError("baseline graph structure changed")
        path = "tensors/" + arm + ".json"
        tensor = {"schema": "nca-literal-tensors/1", "arm": arm,
                  "tensors": {name: value.tolist() for name, value in sorted(candidate.params.items())}}
        write_json(run / path, tensor)
        bindings[path] = file_hash(run / path)
        arms.append({"arm": arm, "role": "matched_full_random" if arm.startswith("paired") else ("zero_control" if arm == "zero" else "fixed_reference"),
                     "tensor_file": path, "tensor_file_sha256": bindings[path], "tensor_sha256": digest(tensor["tensors"]),
                     "graph_sha256": candidate.hash(), "stored_positions": 1488, "parents": [],
                     "numeric_provenance": ("Existing neighbor_majority T8 builder parameters copied onto canonical identity topology" if arm == "neighbor_majority" else "Existing identity T8 builder" if arm == "identity" else "All stored positions zero" if arm == "zero" else "Frozen uniform seven-value full-random draw, seed 2026100801"),
                     "per_tensor_sha256": {name: array_hash(value) for name, value in sorted(candidate.params.items())}})
    standard = {"arms": {"llm": {}, "random": {}}, "budget_per_arm": 0,
                "founders": [], "genomes": [], "niches": [], "proposals": [],
                "seeds": list(SEEDS), "seeds_declared_before_run": True, "scientific_acceptance": False,
                "limitations": ["Training-only baseline preflight; no author calls, no consumed author/random attempt charges. Not ONE_STANDARD-qualified."]}
    write_json(run / "STANDARD.json", standard)
    bindings["STANDARD.json"] = file_hash(run / "STANDARD.json")
    manifest = {
        "schema": "cellforge-nca-feasibility-preflight/1", "base_commit": BASE_COMMIT,
        "phase": "frozen_before_baseline_predictions", "source_sha256": sources, "artifact_sha256": bindings,
        "runtime": {"python": sys.version, "numpy": np.__version__, "backend": "reference", "dtype": "float64",
                    "decoder": "Unchanged GridPredictor dimension/color argmax; ties choose first/lowest index.",
                    "leave_one_out": "Each scored demo's entire pair is excluded from graph context; other task demos in original order. No source query/test fields are present."},
        "resource_limits": LIMITS,
        "execution_security_scope": "Trusted host code in resource-bounded isolated subprocesses inside the existing cloud execution sandbox. No authored Python executes; this is not proof of OS network isolation or a sandbox for arbitrary model code.",
        "graph_metadata_provenance": "All arms retain canonical identity-builder topology metadata; actual numeric reference/random provenance is separately recorded per arm. Seed metadata does not describe overwritten numeric tensors.",
        "arms": arms, "arm_execution_order": list(ARMS),
        "graph": {"iterations": 8, "structure_sha256": digest(structure), "shared_global_tensor": True,
                  "stored_positions": 1488, "parameter_arrays": 96, "active_microcells": 48,
                  "hidden_bias_redundant_directions": 24,
                  "bias_explanation": "Three hidden chunk biases sum per row, yielding two real-arithmetic structural redundant directions for each of 12 rows. Rebalancing is not claimed bitwise equivalent in float64. All 36 hidden biases remain stored and charged; no reparameterization or effective-dimension claim.",
                  "tensor_schema": tensor_schema},
        "panel": {"all_task_ids": list(ALL_IDS), "all_tasks": 11, "all_demonstrations": 32,
                  "compatible_task_ids": list(COMPATIBLE_IDS), "compatible_tasks": 6, "compatible_demonstrations": 17,
                  "excluded_tasks": 5, "excluded_demonstrations": 15,
                  "coverage": capacity["task_status"],
                  "scope": "Previously exposed public-training demonstrations only. Five shape-incompatible tasks retained in accounting and never executed. Six-task restriction is not whole-panel feasibility.",
                  "cancelled_task": "e7639916 remains excluded, unopened, unscored and never replaced."},
        "accounting": {"author_calls": 0, "author_charged_positions": 0, "paired_random_attempt_charged_positions": 0,
                       "preflight_random_evaluated_positions_if_run": 1488, "preflight_random_evaluations_if_run": 1,
                       "second_random_constructed": False, "credits": "unknown; no author/model call dispatched by this preflight"},
        "future_author_protocol": future_protocol(),
        "baseline_authorization": "Explicit parent review of this manifest hash required before baseline command.",
        "author_dispatch_authorized": False,
        "replay_path": "python source/scripts/nca_feasibility_preflight.py replay --run . --manifest-sha256 <MANIFEST.sha256>",
    }
    write_json(run / "MANIFEST.json", manifest)
    manifest_hash = file_hash(run / "MANIFEST.json")
    with (run / "MANIFEST.sha256").open("x") as handle:
        handle.write(manifest_hash + "\n")
    return manifest_hash


def verify_freeze(run, expected, repo=None):
    if os.path.lexists(run / "STOP"):
        raise ValueError("STOP sentinel present")
    if file_hash(run / "MANIFEST.json") != expected or (run / "MANIFEST.sha256").read_text().strip() != expected:
        raise ValueError("reviewed manifest hash mismatch")
    manifest = read_json(run / "MANIFEST.json")
    if manifest["schema"] != "cellforge-nca-feasibility-preflight/1" or manifest["arm_execution_order"] != list(ARMS):
        raise ValueError("unexpected protocol")
    if set(manifest["source_sha256"]) != set(SOURCE_PATHS):
        raise ValueError("incomplete runtime source binding")
    for name, expected_hash in manifest["source_sha256"].items():
        if file_hash(relative_file(run, "source/" + name)) != expected_hash:
            raise ValueError("saved source binding failure: " + name)
        if repo is not None and file_hash(repo / name) != expected_hash:
            raise ValueError("working source binding failure: " + name)
    for name, expected_hash in manifest["artifact_sha256"].items():
        if file_hash(relative_file(run, name)) != expected_hash:
            raise ValueError("artifact binding failure: " + name)
    if manifest["future_author_protocol"] != future_protocol() or manifest["resource_limits"] != LIMITS:
        raise ValueError("protocol constants drift")
    if file_hash(Path(__file__)) != manifest["source_sha256"]["scripts/nca_feasibility_preflight.py"]:
        raise ValueError("executing harness source binding failure")
    np, _, _, _, _, _, diagnostic = runtime_imports()
    module_sources = {"cellforge": "src/cellforge/__init__.py", "cellforge.construction": "src/cellforge/construction/__init__.py",
                      "cellforge.construction.nca_gene": "src/cellforge/construction/nca_gene.py",
                      "cellforge.ir": "src/cellforge/ir/__init__.py", "cellforge.ir.graph": "src/cellforge/ir/graph.py",
                      "cellforge.runtime": "src/cellforge/runtime/__init__.py", "cellforge.runtime.grid": "src/cellforge/runtime/grid.py",
                      "cellforge.runtime.interp": "src/cellforge/runtime/interp.py", "cellforge.runtime.ops": "src/cellforge/runtime/ops.py",
                      "nca_capacity_diagnostic": "scripts/nca_capacity_diagnostic.py"}
    for module_name, path in module_sources.items():
        if file_hash(sys.modules[module_name].__file__) != manifest["source_sha256"][path]:
            raise ValueError("imported runtime source binding failure: " + module_name)
    if manifest["runtime"]["python"] != sys.version or manifest["runtime"]["numpy"] != np.__version__:
        raise ValueError("runtime version drift")
    panel = read_json(run / "PANEL.json")
    if set(panel) != {"schema", "tasks"} or panel["schema"] != "demo-only/1":
        raise ValueError("invalid demo-only panel")
    tasks = panel["tasks"]
    if tuple(task["task_id"] for task in tasks) != ALL_IDS:
        raise ValueError("all-eleven accounting changed")
    checked = diagnostic.analyze(tasks)
    if checked["demonstrations"] != 32:
        raise ValueError("all-32 demonstration accounting changed")
    compatible = [task for task in tasks if task["task_id"] in COMPATIBLE_IDS]
    if diagnostic.analyze(compatible)["whole_panel_necessary_conditions"] != "not_ruled_out":
        raise ValueError("restricted-panel structural contradiction")
    for arm in ARMS:
        load_graph(run, manifest, arm)
    return manifest, compatible


def load_graph(run, manifest, arm):
    np, build, _, limits, array_hash, _, _ = runtime_imports()
    if arm not in ARMS:
        raise ValueError("unauthorized arm")
    spec = next(item for item in manifest["arms"] if item["arm"] == arm)
    payload = read_json(run / spec["tensor_file"])
    if set(payload) != {"schema", "arm", "tensors"} or payload["schema"] != "nca-literal-tensors/1" or payload["arm"] != arm:
        raise ValueError("invalid tensor envelope")
    graph = build(iterations=8, seed="identity")
    if set(payload["tensors"]) != set(graph.params):
        raise ValueError("tensor key set mismatch")
    for name in graph.params:
        value = payload["tensors"][name]
        def numeric(item):
            return (all(numeric(child) for child in item) if isinstance(item, list)
                    else type(item) in (int, float) and math.isfinite(item))
        if not numeric(value):
            raise ValueError("literal finite numbers required")
        array = np.asarray(value, dtype=np.float64)
        if array.shape != graph.params[name].shape:
            raise ValueError("tensor shape mismatch")
        if arm == "paired_random_1" and not np.isin(array, VOCABULARY).all():
            raise ValueError("random vocabulary mismatch")
        if array_hash(array) != spec["per_tensor_sha256"][name]:
            raise ValueError("per-tensor binding failure")
        graph.params[name] = array
    graph.validate(limits(strict_microcells=True))
    if (graph.hash() != spec["graph_sha256"] or digest(payload["tensors"]) != spec["tensor_sha256"]
            or digest(graph_structure(graph)) != manifest["graph"]["structure_sha256"]
            or sum(array.size for array in graph.params.values()) != 1488):
        raise ValueError("graph/source binding failure")
    return graph


def leave_one_out(task, index):
    others = [pair for i, pair in enumerate(task["train"]) if i != index]
    if not others:
        raise ValueError("leave-one-out requires at least two demonstrations")
    return others


def error_row(task_id, index, error, elapsed=0.0):
    return {"task_id": task_id, "demo_index": index, "status": "runtime_error", "prediction": None,
            "exact": None, "pixel_errors": None, "normalized_pixel_error": None,
            "runtime_error": error, "elapsed_seconds": elapsed, "exec_stats": None}


def evaluate_demo(predictor, task, index):
    pair = task["train"][index]
    start = time.perf_counter()
    try:
        predictions = predictor(leave_one_out(task, index), pair["input"])
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise ValueError("one prediction required")
        predicted = predictions[0]
        if (len(predicted) != len(pair["output"])
                or any(len(a) != len(b) for a, b in zip(predicted, pair["output"]))):
            raise ValueError("fixed-size graph returned wrong shape")
        errors = sum(a != b for row, target in zip(predicted, pair["output"]) for a, b in zip(row, target))
        count = sum(len(row) for row in pair["output"])
        return {"task_id": task["task_id"], "demo_index": index, "status": "valid", "prediction": predicted,
                "exact": errors == 0, "pixel_errors": errors, "normalized_pixel_error": errors / count,
                "runtime_error": None, "elapsed_seconds": time.perf_counter() - start, "exec_stats": predictor.last_exec}
    except Exception as exc:
        return error_row(task["task_id"], index, {"type": type(exc).__name__, "message": str(exc)}, time.perf_counter() - start)


def apply_limits():
    import resource
    for key in THREAD_KEYS:
        if os.environ.get(key) != "1":
            raise ValueError("one numerical thread must be configured before process startup")
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["arm_cpu_seconds"], LIMITS["arm_cpu_seconds"]))
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS["address_space_bytes"], LIMITS["address_space_bytes"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (LIMITS["file_size_bytes"], LIMITS["file_size_bytes"]))
    resource.setrlimit(resource.RLIMIT_NOFILE, (LIMITS["open_files"], LIMITS["open_files"]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def worker(run, expected, arm, out):
    apply_limits()
    manifest, tasks = verify_freeze(run, expected)
    *_, predictor_class, _ = runtime_imports()
    predictor = predictor_class(load_graph(run, manifest, arm), backend="reference", cache_demo_state=False)
    def timed_out(signum, frame):
        raise TimeoutError("per-demo wall limit exceeded")
    signal.signal(signal.SIGALRM, timed_out)
    with out.open("x") as handle:
        for task in tasks:
            for index in range(len(task["train"])):
                if os.path.lexists(run / "STOP"):
                    raise ValueError("STOP sentinel present")
                signal.setitimer(signal.ITIMER_REAL, LIMITS["demo_wall_seconds"])
                try:
                    row = evaluate_demo(predictor, task, index)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
    return {"arm": arm, "completed": True}


def summarize(rows, tasks):
    errors = [row for row in rows if row["status"] != "valid"]
    by_task = {task["task_id"]: [row for row in rows if row["task_id"] == task["task_id"]] for task in tasks}
    task_scores = {key: (None if any(row["exact"] is None for row in value) else all(row["exact"] for row in value)) for key, value in by_task.items()}
    return {"valid": not errors, "runtime_errors": len(errors), "demonstration_denominator": len(rows),
            "task_denominator": len(tasks), "exact_demonstrations": None if errors else sum(row["exact"] for row in rows),
            "exact_tasks": None if errors else sum(task_scores.values()), "task_exact": task_scores,
            "observed_exact_valid_demos": sum(row["exact"] is True for row in rows),
            "mean_normalized_pixel_error": None if errors else sum(row["normalized_pixel_error"] for row in rows) / len(rows),
            "solved_demo_keys": [[row["task_id"], row["demo_index"]] for row in rows if row["exact"] is True]}


def semantic_result(value):
    """Deterministic replay comparison excludes timings only, not error validity."""
    if isinstance(value, dict):
        return {key: semantic_result(item) for key, item in value.items()
                if key not in {"elapsed_seconds", "process_elapsed_seconds"}}
    if isinstance(value, list):
        return [semantic_result(item) for item in value]
    return value


def run_arm(run, expected, manifest, tasks, arm, output_dir):
    progress = output_dir / (arm + ".progress.jsonl")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    for key in THREAD_KEYS:
        env[key] = "1"
    env["PYTHONPATH"] = str(run / "source/src") + os.pathsep + str(run / "source/scripts")
    cmd = [sys.executable, str(run / "source/scripts/nca_feasibility_preflight.py"), "_worker", "--run", str(run),
           "--manifest-sha256", expected, "--arm", arm, "--out", str(progress)]
    start = time.perf_counter()
    error = None
    try:
        completed = subprocess.run(cmd, cwd=run, env=env, timeout=LIMITS["arm_wall_seconds"], capture_output=True, text=True, check=False)
        stdout, stderr = completed.stdout, completed.stderr
        returncode = completed.returncode
        if returncode != 0:
            error = {"type": "SubprocessError", "message": "worker exit " + str(returncode)}
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        returncode = None
        error = {"type": "ArmTimeout", "message": "arm wall limit exceeded"}
    duration = time.perf_counter() - start
    for suffix, content in (("stdout", stdout), ("stderr", stderr)):
        with (output_dir / (arm + "." + suffix + ".txt")).open("x") as handle:
            handle.write(content)
    rows = []
    if progress.exists():
        for line in progress.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                error = {"type": "IncompleteProgress", "message": "partial worker record"}
                break
    expected_keys = [(task["task_id"], i) for task in tasks for i in range(len(task["train"]))]
    actual_keys = [(row.get("task_id"), row.get("demo_index")) for row in rows]
    if actual_keys != expected_keys[:len(actual_keys)]:
        raise ValueError("worker observation order mismatch")
    if len(rows) != len(expected_keys) and error is None:
        error = {"type": "IncompleteArm", "message": "worker did not record every demonstration"}
    for task_id, index in expected_keys[len(rows):]:
        rows.append(error_row(task_id, index, error))
    summary = summarize(rows, tasks)
    if error is not None:
        summary.update(valid=False, exact_demonstrations=None, exact_tasks=None, mean_normalized_pixel_error=None)
    result = {"schema": "nca-baseline-arm-result/1", "arm": arm, "manifest_sha256": expected,
              "graph_sha256": next(spec["graph_sha256"] for spec in manifest["arms"] if spec["arm"] == arm),
              "process_elapsed_seconds": duration, "returncode": returncode, "process_error": error,
              "numerical_thread_environment": {key: env[key] for key in THREAD_KEYS},
              "summary": summary, "demonstrations": rows}
    write_json(output_dir / (arm + ".json"), result)
    return result


def duplication_metrics(results, manifest):
    groups = defaultdict(list)
    numeric = defaultdict(list)
    pairs = []
    for spec in manifest["arms"]:
        numeric[spec["tensor_sha256"]].append(spec["arm"])
    valid = [result for result in results if result["summary"]["valid"]]
    for result in valid:
        groups[digest([row["prediction"] for row in result["demonstrations"]])].append(result["arm"])
    for i, left in enumerate(valid):
        for right in valid[i + 1:]:
            equal = sum(a["prediction"] == b["prediction"] for a, b in zip(left["demonstrations"], right["demonstrations"]))
            pairs.append({"left": left["arm"], "right": right["arm"], "identical_demo_predictions": equal, "denominator": 17})
    return {"numeric_duplicate_groups": [value for value in numeric.values() if len(value) > 1],
            "functional_duplicate_groups": [value for value in groups.values() if len(value) > 1],
            "functional_scope": "Only complete valid prediction vectors on these 17 exposed training demonstrations; no global functional-equivalence claim.",
            "unique_valid_prediction_vectors": len(groups), "pairwise_demo_prediction_agreement": pairs}


def summarize_baseline(results, manifest):
    valid_references = [result for result in results if result["arm"] in REFERENCES and result["summary"]["valid"]]
    selected = min(valid_references, key=lambda result: (-result["summary"]["exact_demonstrations"], -result["summary"]["exact_tasks"],
                     result["summary"]["mean_normalized_pixel_error"], REFERENCES.index(result["arm"]))) if valid_references else None
    stop_reasons = []
    if any(not result["summary"]["valid"] for result in results):
        stop_reasons.append("invalid_control")
    if any(result["summary"]["valid"] and result["summary"]["exact_demonstrations"] == 17 for result in results):
        stop_reasons.append("baseline_perfect")
    return {"schema": "nca-baseline-preflight-result/1", "manifest_sha256": None,
            "status": "stop_before_author" if stop_reasons else "baseline_valid_parent_decision_required",
            "stop_reasons": stop_reasons, "author_dispatch_authorized": False,
            "selected_baseline": selected["arm"] if selected else None,
            "selected_baseline_solved_demo_keys": selected["summary"]["solved_demo_keys"] if selected else [],
            "arms": {result["arm"]: result["summary"] for result in results},
            "coverage": manifest["panel"], "duplication": duplication_metrics(results, manifest),
            "accounting": dict(manifest["accounting"], preflight_random_evaluated_positions=1488, preflight_random_evaluations=1),
            "qualification": manifest["future_author_protocol"]["qualification"],
            "baseline_result_is_not_an_authored_gain": True}


def execute(run, expected, replay=False):
    manifest, tasks = verify_freeze(run, expected)
    output_dir = run / ("replay" if replay else "baseline")
    if output_dir.exists():
        raise ValueError("this finite phase already started; no overwrite or silent retry")
    if replay:
        if not (run / "BASELINE_RESULT.sha256").exists():
            raise ValueError("completed baseline receipt required for replay")
        if file_hash(run / "BASELINE_RESULT.json") != (run / "BASELINE_RESULT.sha256").read_text().strip():
            raise ValueError("baseline result hash mismatch")
        for name, hash_value in read_json(run / "BASELINE_RESULT.json")["artifact_sha256"].items():
            if file_hash(relative_file(run, name)) != hash_value:
                raise ValueError("baseline artifact changed before replay")
    output_dir.mkdir()
    results = []
    for arm in ARMS:
        verify_freeze(run, expected)
        results.append(run_arm(run, expected, manifest, tasks, arm, output_dir))
    result = summarize_baseline(results, manifest)
    result["manifest_sha256"] = expected
    result["artifact_sha256"] = {str(path.relative_to(run)): file_hash(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
    if replay:
        comparisons = {}
        for arm, saved in zip(ARMS, results):
            original = read_json(run / "baseline" / (arm + ".json"))
            comparisons[arm] = semantic_result(saved) == semantic_result(original)
        baseline = read_json(run / "BASELINE_RESULT.json")
        comparable = {key: value for key, value in result.items() if key != "artifact_sha256"}
        baseline_comparable = {key: value for key, value in baseline.items() if key != "artifact_sha256"}
        result["replay"] = {"arms_match": comparisons, "aggregate_match": comparable == baseline_comparable,
                            "independent_processes": True, "saved_sources_and_tensors": True,
                            "passed": all(comparisons.values()) and comparable == baseline_comparable,
                            "baseline_result_sha256": file_hash(run / "BASELINE_RESULT.json")}
        write_json(run / "REPLAY_RESULT.json", result)
    else:
        write_json(run / "BASELINE_RESULT.json", result)
        with (run / "BASELINE_RESULT.sha256").open("x") as handle:
            handle.write(file_hash(run / "BASELINE_RESULT.json") + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "baseline", "replay", "_worker"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if args.command == "freeze":
        result = {"manifest_sha256": freeze(args.repo.resolve(), run), "predictions_executed": False}
    else:
        if not args.manifest_sha256:
            parser.error("explicit reviewed --manifest-sha256 required")
        if args.command == "verify":
            manifest, tasks = verify_freeze(run, args.manifest_sha256, args.repo.resolve())
            result = {"verified": True, "tasks": len(tasks), "demos": sum(len(task["train"]) for task in tasks), "author_dispatch_authorized": False}
        elif args.command == "_worker":
            if args.arm is None or args.out is None:
                parser.error("worker requires arm and output")
            result = worker(run, args.manifest_sha256, args.arm, args.out)
        else:
            result = execute(run, args.manifest_sha256, replay=args.command == "replay")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
