"""Compose authored cells (depth <= N), verify on the demo pairs, emit two ranked attempts.
Enumerative composition is the baseline search; the CellForge artifact would replace it with bred wiring."""
import sys, os, json, glob, time, itertools
import numpy as np
from multiprocessing import Pool
from cells import CELLS
from fitted import fit_local_rule
from fitted2 import fit_shape_changing

DEFAULT_ROOT = ("C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/"
                "48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc2/data")


def safe(fn, g):
    try:
        o = np.asarray(fn(g))
        if o.ndim != 2 or o.size == 0 or o.shape[0] > 30 or o.shape[1] > 30:
            return None
        return o
    except Exception:
        return None


def fit_colormap(preds, outs):
    """Task-level color map fitted by exact consistency on demos (counting, no gradients)."""
    m = {}
    for p, y in zip(preds, outs):
        if p.shape != y.shape:
            return None
        for a, b in zip(p.ravel(), y.ravel()):
            if m.setdefault(int(a), int(b)) != int(b):
                return None
    return m


def apply_map(g, m):
    o = g.copy()
    for a, b in m.items():
        o[g == a] = b
    return o


def run_chain(chain, g):
    for n in chain:
        g = safe(CELLS[n], g)
        if g is None:
            return None
    return g


def search(train, test_in, max_depth=2, tlimit=20.0):
    ins = [x for x, _ in train]
    outs = [y for _, y in train]
    t0 = time.time()
    sols, seen = [], set()
    names = list(CELLS)
    for depth in range(1, max_depth + 1):
        for chain in itertools.product(names, repeat=depth):
            if time.time() - t0 > tlimit:
                return sols
            if depth > 1 and chain[0] == "id":
                continue
            cur = [run_chain(chain, g) for g in ins]
            if any(c is None for c in cur):
                continue
            key = tuple(c.tobytes() + bytes(c.shape) for c in cur)
            if key in seen:
                continue
            seen.add(key)
            ok = all(c.shape == y.shape and np.array_equal(c, y) for c, y in zip(cur, outs))
            cm = None
            if not ok:
                cm = fit_colormap(cur, outs)
                if not cm or all(a == b for a, b in cm.items()):
                    continue
            preds = []
            for tg in test_in:
                o = run_chain(chain, tg)
                if o is not None and cm is not None:
                    o = apply_map(o, cm)
                preds.append(o)
            if all(p is not None for p in preds):
                sols.append((chain, cm is not None, preds))
                if len(sols) >= 4:
                    return sols
    return sols


def run_task(a):
    f, depth, tl = a
    d = json.load(open(f))
    tr = [(np.array(e["input"]), np.array(e["output"])) for e in d["train"]]
    te = [np.array(e["input"]) for e in d["test"]]
    tgt = [np.array(e["output"]) for e in d["test"]]
    t0 = time.time()
    sols = search(tr, te, depth, tl)
    for kind, preds in fit_local_rule(tr, te):
        sols.append((("fit:" + kind,), False, preds))
    for kind, preds in fit_shape_changing(tr, te):
        sols.append((("fit:" + kind,), False, preds))
    attempts = []
    for chain, cm, preds in sols:
        if not any(all(np.array_equal(p, q) for p, q in zip(preds, a2)) for a2 in attempts):
            attempts.append(preds)
        if len(attempts) == 2:
            break
    solved = any(all(np.array_equal(p, t) for p, t in zip(a2, tgt)) for a2 in attempts)
    return {"task": os.path.basename(f), "solved": bool(solved), "n_sols": len(sols),
            "chain": [list(s[0]) for s in sols[:2]], "sec": round(time.time() - t0, 2)}


if __name__ == "__main__":
    split, depth, tl, workers = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    lim = int(sys.argv[5]) if len(sys.argv) > 5 else None
    root = os.environ.get("ARC2", DEFAULT_ROOT)
    files = sorted(glob.glob(f"{root}/{split}/*.json"))[:lim]
    t0 = time.time()
    with Pool(workers) as p:
        res = p.map(run_task, [(f, depth, tl) for f in files], chunksize=1)
    ns = sum(r["solved"] for r in res)
    json.dump(res, open(f"results_{split}_d{depth}_v3.json", "w"))
    print(f"{split} depth<={depth}: solved {ns}/{len(res)} ({100 * ns / len(res):.1f}%) in {time.time() - t0:.0f}s; "
          f"tasks with any demo-consistent chain: {sum(r['n_sols'] > 0 for r in res)}")
