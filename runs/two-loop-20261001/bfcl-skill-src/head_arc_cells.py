"""Head-center ARC cell-authoring loop. gpt-6.1-sol (low) writes NEW literal-weight cells from failing ARC training tasks.

Runs on the server in its own namespace (default /srv/cellforge-inbox/head-20261001/runs/weights-head-arc-cells); reuses the frozen
two-loop source read-only (cellforge.construction.arc_weight_moe / joint_weight_population) and the owner-configured maker route
(`codex exec -m gpt-6.1-sol -c model_reasoning_effort=low`, ChatGPT subscription, read-only sandbox, no tools).

Each generation: score champion on the 100 declared DEV tasks (public ARC training, test outputs only inside the evaluator) -> pick failing
tasks -> maker sees ONLY demonstration pairs (+ its previous failed attempt with predicted vs expected) and writes one cell per task
(template + literal tensors, 10 colours) -> a cell is eligible only if it reproduces every demonstration of its task exactly -> candidate =
champion + eligible cells, router = identity over demonstration-error features -> admitted only on strict DEV gain and no per-task
regression. Random arm: same templates and charged positions, uniform[-9,9] tensors, three independent lineages. Every submitted
position is charged. No gradients, no reporting split, no API key.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import time

import numpy as np

RUN_SRC = "/srv/cellforge-inbox/two-loop-20261001/runs/weights-arc-moe-smallskill-v2"
DATA = "/srv/cellforge-private/arcagi2/data/training"
CODEX = "/root/.local/bin/codex"
BATCH, MAX_ATTEMPTS, MAX_CELLS = 4, 3, 1800


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, sort_keys=True)
    os.replace(tmp, path)


def load(path):
    with open(path) as f:
        return json.load(f)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def setup(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("pop1_core.py", "arc_eval.py", "parent.json", "ARC_FLOOR.json"):
        src = pathlib.Path(RUN_SRC) / name
        if not (root / name).exists():
            (root / name).write_bytes(src.read_bytes())
    tasks = load(pathlib.Path(RUN_SRC) / "TASKS.json")["development"]
    rows = []
    for item in tasks:
        path = pathlib.Path(DATA) / (item["id"] + ".json")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("task file differs from declared hash: " + item["id"])
        rows.append({"id": item["id"], "task": load(path)})
    return rows


def grid_cells(task):
    return sum(len(g) * len(g[0]) for p in task["train"] for g in (p["input"], p["output"]))


def pick_tasks(rows, outcomes, attempts, rng):
    def key(row):
        t = row["task"]
        same = all(len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0]) for p in t["train"])
        return (len(attempts.get(row["id"], [])), 0 if same else 1, grid_cells(t) > 1500, rng.random())
    pool = [r for r in rows if not outcomes[r["id"]] and len(attempts.get(r["id"], [])) < MAX_ATTEMPTS and grid_cells(r["task"]) <= 3000]
    return sorted(pool, key=key)[:BATCH]


def shapes_text(core):
    return "\n".join("%s: tensors %s. %s" % (t, {k: list(v) for k, v in core.SHAPES[t].items()}, core.DOCS[t]) for t in core.SHAPES
                     if t in core.DOCS)


def describe_attempt(core, attempt, task):
    if not attempt.get("valid"):
        return {"note": "previous submission was malformed (wrong template name or tensor shapes); write a complete valid cell"}
    """Previous failed attempt as a counterexample: predicted vs expected on the first demonstration it got wrong."""
    for i, pair in enumerate(task["train"]):
        try:
            predicted = core.PREDICTORS[attempt["template"]](attempt["params"], pair["input"], False)[0]
        except Exception as error:
            return {"template": attempt["template"], "params": attempt["params"], "demo": i, "error": type(error).__name__}
        if predicted != pair["output"]:
            return {"template": attempt["template"], "params": attempt["params"], "demo": i, "input": pair["input"],
                    "expected": pair["output"], "predicted": predicted}
    return {"template": attempt["template"], "params": attempt["params"], "note": "reproduced every demonstration but failed the query"}


def build_prompt(core, chosen, attempts):
    items = []
    for row in chosen:
        item = {"task_id": row["id"], "demonstrations": [{"input": p["input"], "output": p["output"]} for p in row["task"]["train"]],
                "query_input_shape": [len(t["input"]) for t in row["task"]["test"]]}
        if attempts.get(row["id"]):
            item["previous_failed_attempt"] = describe_attempt(core, attempts[row["id"]][-1], row["task"])
        items.append(item)
    text = ("You write literal weight tensors for small grid cells (no code, no tools, no files). Colours are integers 0-9. For EACH task below, "
            "choose ONE template and write its complete literal tensors (finite numbers in [-9, 9]) so that the cell reproduces EVERY demonstration "
            "input->output exactly and would apply the same rule to a new input. If previous_failed_attempt is given, read its predicted vs expected "
            "and write a genuinely different revision that fixes that failure. If no template can express the rule, still submit your best-fitting "
            "cell. Templates and their exact semantics:\n" + shapes_text(core) +
            "\nReturn ONLY a JSON array, one object per task: [{\"task_id\": str, \"template\": str, \"params\": {tensor_name: nested lists}, "
            "\"rationale\": str naming the concrete demonstration evidence}].\nTasks:\n" + json.dumps(items))
    return text, items


def ask_maker(text, work, fake, core, chosen):
    work.mkdir(parents=True, exist_ok=True)
    if fake:
        rng = random.Random(7)
        out = [{"task_id": r["id"], "template": "colormap", "params": {"map.W": [[rng.uniform(-9, 9) for _ in range(10)] for _ in range(10)]},
                "rationale": "fake maker for plumbing test: demonstration %s" % r["id"]} for r in chosen]
        write(work / "response.json", out)
        return out
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")}
    cmd = [CODEX, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "-m", "gpt-6.1-sol",
           "-c", 'model_reasoning_effort="low"', "--json", "-C", str(work), "-o", str(work / "response.json"), "-"]
    write(work / "START.json", {"utc": now(), "model": "gpt-6.1-sol", "reasoning": "low", "auth": "ChatGPT subscription", "prompt_sha256": sha(text)})
    started = time.monotonic()
    with open(work / "events.jsonl", "w") as log:
        done = subprocess.run(cmd, input=text, text=True, stdout=log, stderr=log, env=env, timeout=900)
    events = (work / "events.jsonl").read_text()
    write(work / "EXIT.json", {"returncode": done.returncode, "seconds": time.monotonic() - started})
    if done.returncode or '"type":"command_execution"' in events or '"type":"mcp_tool_call"' in events:
        raise RuntimeError("maker failed or invoked tools; no release")
    raw = (work / "response.json").read_text().strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def valid_cell(core, cell):
    shapes = core.SHAPES.get(cell.get("template"))
    params = cell.get("params")
    if not shapes or not isinstance(params, dict) or set(params) != set(shapes):
        return False
    for name, shape in shapes.items():
        arr = np.asarray(params[name], dtype=float) if params[name] is not None else None
        if arr is None or arr.shape != tuple(shape) or not np.isfinite(arr).all() or np.abs(arr).max() > 9:
            return False
    return True


def positions(cell):
    return int(sum(np.asarray(v).size for v in cell["params"].values()))


def fits(core, cell, task):
    try:
        return all(core.PREDICTORS[cell["template"]](cell["params"], p["input"], False)[0] == p["output"] for p in task["train"])
    except Exception:
        return False


def random_like(core, cell, rng):
    return {"template": cell["template"], "params": {n: np.array([rng.uniform(-9, 9) for _ in range(int(np.prod(s)))]).reshape(s).tolist()
                                                     for n, s in core.SHAPES[cell["template"]].items()}}


def with_cells(tensors, cells):
    out = {k: v for k, v in tensors.items() if not k.startswith("route.")}
    index = 1 + max(int(k.split(":")[0]) for k in out)
    for cell in cells:
        for name, value in cell["params"].items():
            out["%d:%s" % (index, name)] = copy.deepcopy(value)
        index += 1
    n = len({k.split(":")[0] for k in out})
    out["route.W"] = np.eye(n).tolist()
    out["route.b"] = [0.0] * n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from cellforge.construction.arc_weight_moe import admission, make_core, score
    from cellforge.construction.joint_weight_population import digest
    root = a.root
    rows = setup(root)
    core = make_core(root / "pop1_core.py")
    floor = load(root / "ARC_FLOOR.json")["min_solved_fraction"]
    rng = random.Random(a.seed)
    state_path = root / "STATE.json"
    if state_path.exists():
        state = load(state_path)
    else:
        base = load(root / "parent.json")["tensors"]
        ev = score(base, rows, core)
        state = {"generation": 0, "attempts": {}, "llm": {"tensors": base, "evidence": ev, "history": [ev]},
                 "random": {str(s): {"tensors": base, "evidence": ev, "history": [ev]} for s in (501, 502, 503)}}
        write(root / "ARC_HISTORY.json", [{"iteration": 0, "solved": ev["solved"], "weights_hash": digest(base)}])
    while state["generation"] < a.generations:
        if (root / "STOP").exists():
            write(root / "STATUS.json", {"status": "user_stop", "generation": state["generation"], "utc": now(), "pid": os.getpid()})
            return
        n = state["generation"] + 1
        champion = state["llm"]
        outcomes = champion["evidence"]["outcomes"]
        chosen = pick_tasks(rows, outcomes, state["attempts"], rng)
        if not chosen:
            write(root / "STATUS.json", {"status": "no_unattempted_failing_tasks", "generation": n - 1, "utc": now(), "pid": os.getpid()})
            return
        text, items = build_prompt(core, chosen, {k: [c for c in v] for k, v in state["attempts"].items()})
        write(root / "STATUS.json", {"status": "authoring", "generation": n, "tasks": [r["id"] for r in chosen], "utc": now(), "pid": os.getpid()})
        raw = ask_maker(text, root / ("AUTHOR-%d" % n), a.fake, core, chosen)
        by_id = {r["id"]: r for r in rows}
        submitted, charged, notes = [], 0, []
        for entry in raw:
            tid = entry.get("task_id")
            cell = {"template": entry.get("template"), "params": entry.get("params"), "task_id": tid}
            ok = tid in by_id and valid_cell(core, cell)
            charged += positions(cell) if ok else 1
            if tid in by_id:
                state["attempts"].setdefault(tid, []).append({"template": cell["template"], "params": cell["params"], "valid": ok})
            if ok:
                submitted.append(cell)
            else:
                notes.append({"task_id": tid, "reason": "invalid_cell"})
        for cell in submitted:
            if not fits(core, cell, by_id[cell["task_id"]]["task"]):
                notes.append({"task_id": cell["task_id"], "reason": "does_not_reproduce_demonstrations", "template": cell["template"]})
        decisions, charges = [], []
        arms = [("llm", a.seed, state["llm"])] + [("random", int(s), state["random"][s]) for s in sorted(state["random"])]
        for arm, seed, lineage in arms:
            if arm == "llm":
                proposals = submitted
            else:  # matched control: same templates, same number of literal positions, uniform[-9,9]
                r = random.Random(seed * 1000 + n)
                proposals = [dict(random_like(core, c, r), task_id=c["task_id"]) for c in submitted]
            use = [c for c in proposals if fits(core, c, by_id[c["task_id"]]["task"])]
            charges.append({"generation": n, "seed": seed, "arm": arm, "positions": charged, "submitted_cells": len(proposals), "eligible_cells": len(use)})
            if not use:
                decisions.append({"seed": seed, "niche": None, "arm": arm, "candidate": None, "solved": lineage["evidence"]["solved"],
                                  "accepted": False, "reasons": ["no_eligible_cell"]})
                continue
            candidate = with_cells(lineage["tensors"], use)
            ev = score(candidate, rows, core)
            decision = admission(lineage["evidence"], ev, lineage["history"], floor)
            decisions.append({"seed": seed, "niche": None, "arm": arm, "candidate": digest(candidate), "solved": ev["solved"],
                              "accepted": decision["accepted"], "reasons": decision["reasons"]})
            if decision["accepted"]:
                lineage.update({"tensors": candidate, "evidence": ev})
                lineage["history"].append(ev)
        state["generation"] = n
        champ = state["llm"]
        history = load(root / "ARC_HISTORY.json")
        history.append({"iteration": n, "solved": champ["evidence"]["solved"], "weights_hash": digest(champ["tensors"])})
        write(root / "ARC_HISTORY.json", history)
        write(root / ("GENERATION-%d.json" % n), {"state": {"generation": n}, "decisions": decisions, "notes": notes, "tasks": [r["id"] for r in chosen]})
        with open(root / ("CHARGES-%d.jsonl" % n), "w") as f:
            for c in charges:
                f.write(json.dumps(c) + "\n")
        write(root / ("PROMPTS-%d.json" % n), [{"seed": a.seed, "prompt_sha256": sha(text), "task_ids": [r["id"] for r in chosen]}])
        write(root / ("AUTHOR-%d" % n) / "prompt.txt.json", {"prompt": text})
        write(state_path, state)
        write(root / "FINAL_WEIGHTS.json", {"tensors": champ["tensors"]})
        zero = {k: np.zeros_like(np.asarray(v)).tolist() for k, v in champ["tensors"].items()}
        rand = [state["random"][s]["evidence"]["solved"] for s in sorted(state["random"])]
        write(root / "ARC_SCORE.json", {
            "split": "training", "task_ids": [r["id"] for r in rows], "weights_file": "FINAL_WEIGHTS.json", "evaluator": "arc_eval.py",
            "claimed": {"solved": champ["evidence"]["solved"], "total": len(rows), "per_task": champ["evidence"]["outcomes"]},
            "controls": {"zero_weights": {"solved": score(zero, rows, core)["solved"]},
                         "random_weights": {"solved": max(rand), "per_seed": rand, "seeds": [501, 502, 503], "scope": "matched random cells, same templates and charged positions"}},
            "history": history, "arc1_arc2_merged_by_task_id": True, "scientific_acceptance": False,
            "scope": "100 declared public-training DEV tasks; development only, per-task cells authored from demonstrations; not held-out transfer"})
        write(root / "STATUS.json", {"status": "generation_complete", "generation": n, "solved": champ["evidence"]["solved"], "utc": now(), "pid": os.getpid()})
        print(json.dumps({"generation": n, "tasks": [r["id"] for r in chosen], "submitted": len(submitted), "charged": charged,
                          "llm_solved": champ["evidence"]["solved"], "random": rand, "notes": notes}), flush=True)
    write(root / "STATUS.json", {"status": "generations_complete", "generation": state["generation"], "utc": now(), "pid": os.getpid()})


if __name__ == "__main__":
    main()
