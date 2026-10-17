"""Arena judge: scores researcher submissions on HELD-OUT test pairs.

Anti-cheat:
 1. Researchers never see test.json (lives in judge/heldout only).
 2. weights-load-bearing: perturb weights.json -> output MUST change.
    Catches hardcoded algorithms that ignore weights.
 3. Determinism: 3 runs must agree.
 4. Answer-scan: test outputs must not appear verbatim in the artifact.
 5. Timeout + no-network sandbox for solve.py.

Usage: python3 judge.py <researcher> [round]   (alpha|beta, round 1|2)
Round 1: test.json / submission / results_<researcher>.json
Round 2: test_round2.json / submission_round2 / results_<researcher>_round2.json
Records weight purity (load-bearing groups / total groups) as a tiebreak.
"""
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time

ARENA = "/srv/cellforge-arena"
TIMEOUT_S = 10


def load_test(round_n):
    name = "test.json" if round_n == 1 else f"test_round{round_n}.json"
    with open(f"{ARENA}/judge/heldout/{name}") as f:
        return json.load(f)


def load_submission(researcher, round_n):
    subdir = "submission" if round_n == 1 else f"submission_round{round_n}"
    sub = f"{ARENA}/researchers/{researcher}/{subdir}"
    solve_py = f"{sub}/solve.py"
    weights_json = f"{sub}/weights.json"
    if not os.path.exists(solve_py):
        return None, "missing solve.py"
    if not os.path.exists(weights_json):
        return None, "missing weights.json"
    with open(weights_json) as f:
        weights = json.load(f)
    spec = importlib.util.spec_from_file_location(f"solve_{researcher}", solve_py)
    mod = importlib.util.module_from_spec(spec)
    # sandbox: block network in submissions
    import socket
    orig_socket = socket.socket
    socket.socket = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network blocked"))
    try:
        spec.loader.exec_module(mod)
    finally:
        socket.socket = orig_socket
    if not hasattr(mod, "solve"):
        return None, "solve.py has no solve(grid) function"
    return {"mod": mod, "weights": weights, "dir": sub}, None


def run_solve(sub, grid, weights):
    start = time.time()
    out = sub["mod"].solve([row[:] for row in grid], copy.deepcopy(weights))
    if time.time() - start > TIMEOUT_S:
        raise TimeoutError("solve() exceeded time limit")
    return out


def main():
    researcher = sys.argv[1]
    round_n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    test = load_test(round_n)
    sub, err = load_submission(researcher, round_n)
    result = {"researcher": researcher, "round": round_n, "scores": {}, "checks": {},
              "total": 0, "possible": 0, "verdict": "FAIL"}

    if err:
        result["checks"]["submission"] = f"INVALID: {err}"
        print(json.dumps(result, indent=1))
        return

    result["checks"]["submission"] = "OK"
    all_weights = sub["weights"]

    # --- check 1: answer scan (test outputs must not be embedded) ---
    with open(f'{sub["dir"]}/solve.py') as f:
        code = f.read()
    with open(f'{sub["dir"]}/weights.json') as f:
        wtext = f.read()
    leaked = 0
    for task, pairs in test.items():
        for p in pairs:
            sig = json.dumps(p["output"])
            if sig in code or sig in wtext:
                leaked += 1
    result["checks"]["answer_scan"] = "PASS" if leaked == 0 else f"FAIL: {leaked} test outputs embedded"

    # --- check 2: determinism + scoring on held-out ---
    for task, pairs in test.items():
        correct = 0
        det_ok = True
        for p in pairs:
            r1 = run_solve(sub, p["input"], all_weights)
            r2 = run_solve(sub, p["input"], all_weights)
            if r1 != r2:
                det_ok = False
            if r1 == p["output"]:
                correct += 1
        result["scores"][task] = {"correct": correct, "total": len(pairs)}
        result["total"] += correct
        result["possible"] += len(pairs)
        result["checks"][f"determinism_{task}"] = "PASS" if det_ok else "FAIL"

    # --- check 3: weights load-bearing ---
    # Perturb EACH top-level weight group separately and run ALL test samples.
    # A group is load-bearing if perturbing it changes any output.
    # Verdict passes if a majority of groups are load-bearing.
    def perturb_value(v):
        if isinstance(v, bool):
            return not v
        if isinstance(v, int):
            return v + 1
        if isinstance(v, float):
            return v + 0.5
        return None

    def perturb_group(obj):
        """Perturb every numeric leaf in obj in-place. Returns True if any changed."""
        changed_any = False
        if isinstance(obj, dict):
            for k, v in obj.items():
                nv = perturb_value(v)
                if nv is not None and nv != v:
                    obj[k] = nv
                    changed_any = True
                elif perturb_group(v):
                    changed_any = True
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                nv = perturb_value(v)
                if nv is not None and nv != v:
                    obj[i] = nv
                    changed_any = True
                elif perturb_group(v):
                    changed_any = True
        return changed_any

    # baseline outputs for all test samples
    baselines = {}
    for task, pairs in test.items():
        for idx, p in enumerate(pairs):
            baselines[(task, idx)] = run_solve(sub, p["input"], all_weights)

    # Fix: perturb_group on a top-level scalar (e.g. background: 0) missed it.
    # Handle scalar top-level groups directly.
    groups = [k for k in all_weights.keys()]
    load_bearing = []
    dead = []
    for g in groups:
        pert = copy.deepcopy(all_weights)
        val = pert[g]
        nv = perturb_value(val)
        if nv is not None and nv != val:
            pert[g] = nv
            perturbed_ok = True
        else:
            perturbed_ok = perturb_group(val)
        if not perturbed_ok:
            dead.append(f"{g}(unperturbable)")
            continue
        hit = False
        for (task, idx), base_out in baselines.items():
            p = test[task][idx]
            try:
                out2 = run_solve(sub, p["input"], pert)
            except Exception:
                hit = True
                break
            if out2 != base_out:
                hit = True
                break
        (load_bearing if hit else dead).append(g)
    result["checks"]["weights_load_bearing_detail"] = {
        "load_bearing": load_bearing, "dead": dead}
    # purity tiebreak: fraction of weight groups individually load-bearing
    n_groups = len(groups)
    result["purity"] = (len(load_bearing) / n_groups) if n_groups else 0
    result["purity_detail"] = {"load_bearing": len(load_bearing),
                              "total_groups": n_groups}

    # Global test: zero ALL weights -> outputs must change on a majority of
    # samples. This catches purely decorative weights.
    def zero(o):
        if isinstance(o, dict):
            for k in o:
                v = o[k]
                if isinstance(v, bool):
                    o[k] = False
                elif isinstance(o[k], (int, float)):
                    o[k] = 0
                else:
                    zero(v)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                if isinstance(v, bool):
                    o[i] = False
                elif isinstance(v, (int, float)):
                    o[i] = 0
                else:
                    zero(v)
    z = copy.deepcopy(all_weights)
    zero(z)
    changed_n = 0
    for (task, idx), base_out in baselines.items():
        p = test[task][idx]
        try:
            out2 = run_solve(sub, p["input"], z)
        except Exception:
            changed_n += 1
            continue
        if out2 != base_out:
            changed_n += 1
    total_n = len(baselines)
    result["checks"]["weights_global_zero"] = (
        f"PASS ({changed_n}/{total_n} outputs changed)"
        if changed_n > total_n // 2 else
        f"FAIL: zeroing all weights changed only {changed_n}/{total_n} outputs (decorative weights?)"
    )
    result["checks"]["weights_load_bearing"] = result["checks"]["weights_global_zero"]

    # --- verdict ---
    cheats = [k for k, v in result["checks"].items()
              if isinstance(v, str) and v.startswith("FAIL")]
    if not cheats and result["possible"] > 0:
        result["verdict"] = "SCORE"
    else:
        result["verdict"] = "CHEAT_FAIL" if any("answer_scan" in c or "load_bearing" in c for c in cheats) else "FAIL"
    result["accuracy"] = result["total"] / result["possible"] if result["possible"] else 0

    os.makedirs(f"{ARENA}/ledger", exist_ok=True)
    suffix = "" if round_n == 1 else f"_round{round_n}"
    with open(f"{ARENA}/ledger/results_{researcher}{suffix}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
