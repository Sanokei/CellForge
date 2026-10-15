"""One immutable public-development experiment; no model calls or networking."""
import hashlib
import json
import os
import pathlib
import random
import sys
import time

import arc3_env as E
import arc3cell as C

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def atomic(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(canonical(obj) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)

def main():
    run = pathlib.Path(sys.argv[1]).resolve()
    config = json.loads((run / "CONFIG.json").read_text())
    root = run.parent.parent
    source = "def act" + (root / "policy_v2.py").read_text().split("def act", 1)[1]
    if sha(source.encode()) != config["policy_source_sha256"]:
        raise RuntimeError("policy source changed")
    ok, reason = C.validate(source)
    if not ok:
        raise RuntimeError("existing sandbox rejected policy: " + reason)
    params = config["params"]
    shapes = {k: [len(v)] for k, v in params.items()}
    ok, reason = C.check_shapes(shapes, params)
    if not ok:
        raise RuntimeError("invalid coefficient shapes: " + reason)
    rng = random.Random(config["control_seed"])
    random_params = {k: [rng.uniform(-4, 4) for _ in v] for k, v in params.items()}
    arms = {"assistant_authored": params, "matched_random": random_params,
            "zero": C.zeros_like(params),
            "spatial_off": dict(params, pixel=[0.0] * 6),
            "action_off": dict(params, action=[0.0] * 8), "blind": params}
    atomic(run / "STANDARD.json", {
        "qualified": False, "scope": "new spatial-exploration development diagnostic",
        "literal_positions": 16, "charged_assistant_positions": 16,
        "charged_random_positions": 16, "remote_maker_calls": 0,
        "provenance": config["provenance"],
        "limitations": ["host assistant-authored weights, not server maker receipts",
                       "no mutation/crossover population qualification",
                       "three already-exposed public development games",
                       "not held-out, novelty, AGI, or service admission evidence"]})
    atomic(run / "ARMS.json", {"source": source, "params": arms})
    rows = []
    start = time.monotonic()
    for arm, weights in arms.items():
        for game in config["games"]:
            if (root / "STOP").exists() or (run / "STOP").exists():
                raise RuntimeError("STOP requested")
            atomic(run / "PROGRESS.json", {"arm": arm, "game": game, "completed_plays": len(rows)})
            result = E.play(source, weights, game, config["actions"], seed=config["seed"],
                            blind=(arm == "blind"), wall=45, keep_trace=True)
            frame = result.pop("final_frame", None)
            result["final_frame_sha256"] = sha((frame or "").encode())
            result["trace_limit"] = 400
            rows.append({"arm": arm, "result": result})
            atomic(run / "RESULTS.json", rows)
    totals = {arm: {
        "levels": sum(r["result"]["levels"] for r in rows if r["arm"] == arm),
        "runtime_errors": sum(bool(r["result"]["error"]) for r in rows if r["arm"] == arm),
        "noops": sum(r["result"]["noops"] for r in rows if r["arm"] == arm),
        "distinct_frames": sum(r["result"]["distinct"] for r in rows if r["arm"] == arm)}
        for arm in arms}
    atomic(run / "SUMMARY.json", {"fingerprint": config["fingerprint"], "hypothesis": config["hypothesis"],
                                "seed": config["seed"], "plays": len(rows),
                                "seconds": time.monotonic() - start, "totals": totals,
                                "promoted": False, "qualification": "development diagnostic only"})
    return 0 if sum(v["runtime_errors"] for v in totals.values()) == 0 else 2

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        atomic(pathlib.Path(sys.argv[1]) / "FAILURE.json",
               {"error": type(error).__name__ + ": " + str(error)[:800]})
        raise SystemExit(1)

