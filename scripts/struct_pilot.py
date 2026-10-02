"""Pilot: do houses that may WRITE THE STRUCTURE (restricted Python in a sandbox) solve real ARC tasks that the fixed templates cannot?

Per (task, house): one maker call shows ONLY the task's demonstration pairs and its test INPUTS (never any output). The maker returns restricted Python
`run(params, grid, demos)` plus the literal tensors. Acceptance (scripts/structcode.py): static whitelist, leave-one-out fit on every demonstration, failure
with zeroed tensors and with three random tensor draws, at most 256 tensor entries. A task counts as KILLED only if the accepted structure then reproduces the
hidden test output exactly. One retry shows the house its own failing demonstration (never test outputs). Tasks come from the selection split (dev + validation);
the report split is never listed here, so a kill is a selection result and nothing more.

  python struct_pilot.py runs/weights-head-structpilot1 --tasks 009d5c81,05f2a901 --sample 20 --houses 3
"""
import argparse
import json
import pathlib
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/srv/cellforge-inbox/head-20261001")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import head_arc_cells as H  # noqa: E402
import structcode as X  # noqa: E402

ARC = pathlib.Path("/srv/cellforge-private/arcagi2/data/training")
HOUSES = ("Ember", "Tide", "Moss", "Flint")
BOARD = "/srv/cellforge-inbox/head-20261001/runs/weights-head-arena1/ARENA_BOARD.json"  # unsolved SELECTION-split ids per family (never the report split)
_local = threading.local()


def load_task(tid):
    """`<id>` is the public task. `<id>~v` is the same task with colours 1-9 permuted and every grid transposed (a contamination probe: a solution recalled
    from training data keyed to the original colours and orientation does not survive it, a rule read from the demonstrations does)."""
    base, _, variant = tid.partition("~")
    task = json.loads((ARC / (base + ".json")).read_text())
    if not variant:
        return task
    rng = random.Random("variant:" + tid)
    colours = list(range(1, 10))
    rng.shuffle(colours)
    perm = [0] + colours

    def grid(g):
        return [[perm[v] for v in row] for row in zip(*g)]  # transpose, then recolour

    return {split: [{k: grid(v) for k, v in ex.items()} for ex in task[split]] for split in ("train", "test")}


BOX_TIMEOUT = 6.0  # struct_solve lowers this to the library's runtime timeout so a structure is accepted under the limit it will later run with


def sandbox():
    if not hasattr(_local, "box"):
        _local.box = X.Sandbox(timeout=BOX_TIMEOUT)
    return _local.box


CONTRACT = """You write one STRUCTURE and its literal weight tensors for a grid-transformation cell, as restricted Python (no tools, no files, no network).

def run(params, grid, demos):
    # params: dict tensor_name -> float numpy array (exactly the shapes you declare)
    # grid:   2-D numpy int array (the input grid, values 0..9)
    # demos:  list of (input_grid, output_grid) numpy pairs: the OTHER demonstrations of this task (read context from them if the rule needs it)
    # return: 2-D numpy int array, values 0..9, at most 30x30

Rules that are checked by code:
- Only `def` at top level (helper functions allowed), `np` is available, nothing else. No imports, no try/with/class/global/f-strings/decorators, no names or
  attributes starting with an underscore, no open/exec/eval/getattr/print. Numpy only through named functions: array, zeros, ones, full, arange, where, argwhere,
  roll, rot90, flip, fliplr, flipud, unique, abs, concatenate, stack, tile, kron, bincount, minimum, maximum, pad, count_nonzero, diff, isin, equal,
  logical_and/or/not, sum/max/min/any/all/argmax/mean/nonzero/astype/copy/reshape/flatten/tolist/T and the usual list/dict/set methods.
- EVERY task-specific number (colours, sizes, offsets, thresholds, which direction, which neighbour) must be read from `params`; read colours as
  int(round(params['x'][i])). The structure is judged with ZEROED tensors and with RANDOM tensors: it must FAIL the demonstrations then. Code that solves the
  task without reading its tensors is rejected.
- At most 256 tensor entries in total, finite numbers within [-1000, 1000]. The tensors are the genome; keep them meaningful and small.
- The structure must reproduce EVERY demonstration (each one with the other demonstrations as `demos`) and generalise to the unseen test inputs.
Return ONLY one JSON array with ONE object, no markdown:
[{"task_id": str, "source_lines": [one string per line of Python], "shapes": {"tensor_name": [dims]}, "params": {"tensor_name": nested lists}, "rationale": str}]
If you believe no structure of this kind can express the rule, still give your best attempt and add "cannot_express": true.
"""


def prompt(house, tid, task, previous=None, feedback=None):
    body = {"task_id": tid, "demonstrations": task["train"], "test_inputs": [t["input"] for t in task["test"]]}
    if previous:
        body["previous_attempt"] = {"source_lines": previous["source_lines"], "shapes": previous["shapes"], "params": previous["params"], "feedback": feedback}
    return ("CELL WARS boss fight on a real ARC task. You are House %s; other houses attempt the same task independently and cannot see your work. Only a "
            "structure that passes every check and then reproduces a hidden test output exactly takes the kill.\n%s\nTask:\n%s" % (house, CONTRACT, json.dumps(body)))


def feedback_of(res, task, box, source, params):
    if not res["valid"]:
        return {"rejected": res["errors"][:1]}
    if res["demos_fit"] < res["n_demos"]:
        d = task["train"][0]
        out, error = box.call(source, params, d["input"], task["train"][1:])
        return {"demonstrations_reproduced": "%d of %d" % (res["demos_fit"], res["n_demos"]), "first_demonstration": {"input": d["input"], "expected": d["output"],
                                                                                                                       "predicted": out if out is not None else error}}
    return {"tensors_do_not_matter": "zeroed tensors fit %s of %s demonstrations, random tensors fit %s" % (res["zero_fit"], res["n_demos"], res["random_fit"])}


def attempt(root, house, tid, task_full):
    box = sandbox()
    task = {"train": task_full["train"], "test": [{"input": t["input"]} for t in task_full["test"]]}  # the maker never sees a test output
    record = {"task": tid, "house": house, "attempts": []}
    previous, feedback = None, None
    for n in (1, 2):
        text = prompt(house, tid, task, previous, feedback)
        try:
            raw = H.ask_maker(text, root / ("AUTHOR-%s-%s-a%d" % (tid, house, n)), False, None, [{"id": tid}])
        except Exception as error:  # a failed or tool-using maker call is an invalid attempt, logged, never faked
            record["attempts"].append({"n": n, "maker_error": "%s: %s" % (type(error).__name__, str(error)[:160])})
            continue
        entry = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], dict) else {}
        source = "\n".join(str(x) for x in entry["source_lines"]) if isinstance(entry.get("source_lines"), list) else str(entry.get("source") or "")
        shapes, params = entry.get("shapes"), entry.get("params")
        res = X.evaluate(box, source, shapes or {}, params or {}, task_full)
        record["attempts"].append({"n": n, "source": source, "shapes": shapes, "params": params, "cannot_express": entry.get("cannot_express") is True,
                                   "positions": X.positions(shapes) if isinstance(shapes, dict) and res["valid"] else 0, "result": res,
                                   "prompt_chars": len(text), "rationale": str(entry.get("rationale", ""))[:300]})
        if res["valid"] and res["demos_fit"] == res["n_demos"] and res["controls_ok"]:
            break
        previous = {"source_lines": source.split("\n"), "shapes": shapes or {}, "params": params or {}}
        feedback = feedback_of(res, task, box, source, params)
    last = record["attempts"][-1] if record["attempts"] else {}
    res = last.get("result") or {}
    record["accepted"] = bool(res.get("valid") and res.get("demos_fit") == res.get("n_demos") and res.get("controls_ok"))
    record["killed"] = bool(record["accepted"] and res.get("solved_test"))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--houses", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    ids = [t for t in a.tasks.split(",") if t]
    if a.sample:
        pool = sorted(i for v in json.loads(pathlib.Path(BOARD).read_text())["bounty_task_ids"].values() for i in v if i not in ids)
        ids += random.Random(7).sample(pool, a.sample)
    houses = HOUSES[:a.houses]
    jobs = [(h, t) for t in ids for h in houses]
    print("tasks %d, houses %d, jobs %d (each: up to 2 maker calls)" % (len(ids), len(houses), len(jobs)), flush=True)
    results, lock = [], threading.Lock()

    def run(job):
        h, t = job
        rec = attempt(a.root, h, t, load_task(t))
        with lock:
            results.append(rec)
            (a.root / "RESULTS.jsonl").open("a").write(json.dumps(rec) + "\n")
            last = (rec["attempts"] or [{}])[-1].get("result") or {}
            print("%-9s %-6s attempts %d  accepted=%s killed=%s  fit %s/%s zero %s random %s" % (t, h, len(rec["attempts"]), rec["accepted"], rec["killed"],
                                                                                              last.get("demos_fit"), last.get("n_demos"), last.get("zero_fit"), last.get("random_fit")), flush=True)
        return rec

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run, jobs))
    by_task = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    killed = [t for t, rs in by_task.items() if any(r["killed"] for r in rs)]
    print("\nSUMMARY tasks %d: killed by at least one house %d | accepted structures %d of %d attempts-sets | killed per house %s" % (
        len(ids), len(killed), sum(r["accepted"] for r in results), len(results), {h: sum(r["killed"] for r in results if r["house"] == h) for h in houses}), flush=True)
    print("killed tasks:", sorted(killed), flush=True)
    (a.root / "SUMMARY.json").write_text(json.dumps({"tasks": ids, "killed": sorted(killed), "houses": list(houses)}))


if __name__ == "__main__":
    main()
