"""Transfer of the maker-written structure library: how many report tasks does the bank still solve when the structure authored FOR that task is removed?

The test-time run authors one structure per task from that task's own demonstrations, so 'solved' is partly 'solved by its own bespoke program'. This isolates
generalisation: for each report task, route through the whole bank (old cells, chains, and every structure authored for OTHER tasks) with that task's own
structure excluded. Routing uses demonstrations only; correctness is checked against the hidden test output afterwards, as for every other score.

  python library_transfer.py runs/weights-head-structbank1
"""
import collections
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/srv/cellforge-inbox/head-20261001")
sys.path.insert(0, "/srv/cellforge-inbox/two-loop-20261001/runs/weights-arc-moe-smallskill-v2/src")
import head_driver as D  # noqa: E402
import skillcore as S  # noqa: E402
import structlib as L  # noqa: E402

_G = {}


def init(root):
    L.reset()  # a forked worker must not share the parent's sandbox pipe
    d = D.Driver(pathlib.Path("/srv/cellforge-inbox/head-20261001/scratch/transfer"), fake=True)
    L.load(S, root)
    d.registry = json.load(open(root / "REGISTRY.json"))
    _G["d"], _G["cells"] = d, d.cells()


def errors_for(row_id):
    d, cells = _G["d"], _G["cells"]
    row = next(r for r in d.rows_rep if r["id"] == row_id)
    return row_id, [d.all.demo_error(c, row) for c in cells]


def main():
    root = pathlib.Path(sys.argv[1])
    init(root)
    d, cells = _G["d"], _G["cells"]
    ids = [r["id"] for r in d.rows_rep]
    with mp.get_context("fork").Pool(2, initializer=init, initargs=(root,)) as pool:
        errs = dict(pool.map(errors_for, ids, chunksize=4))
    router = S.default_router(cells, legacy=d.legacy_router)
    W, b = np.asarray(router["route.W"], float), np.asarray(router["route.b"], float)
    own = collections.defaultdict(list)
    for i, c in enumerate(cells):
        if c.get("authored_for"):
            own[c["authored_for"]].append(i)
    out = {"with_own": 0, "without_own": 0, "had_own": 0, "solved_by_other_structure": 0, "solved_by_old_cell": 0, "per_task": {}}
    for row in d.rows_rep:
        tid, f = row["id"], -np.array(errs[row["id"]])
        for label, drop in (("with_own", []), ("without_own", own.get(tid, []))):
            score = W @ f + b
            for i in drop:
                score[i] = -np.inf
            k = int(np.argmax(score))
            ok = True
            for t in row["task"]["test"]:
                try:
                    ok &= S.run(cells[k], t["input"], row["task"]["train"]) == t["output"]
                except Exception:
                    ok = False
            out[label] += int(ok)
            if label == "without_own":
                out["per_task"][tid] = bool(ok)
                out["had_own"] += int(bool(own.get(tid)))
                if ok:
                    out["solved_by_other_structure" if cells[k].get("niche") == "proposed" else "solved_by_old_cell"] += 1
    print(json.dumps({k: v for k, v in out.items() if k != "per_task"}, indent=1))
    (root / "TRANSFER.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
