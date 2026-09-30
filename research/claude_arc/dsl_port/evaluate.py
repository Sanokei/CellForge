"""Evaluate baseline (existing authored+fitted stack, depth<=2, copied from solve.py minus the concurrently edited objects.py)
and the dsl_port families on ARC-AGI-2 public TRAINING tasks only. Split: crc32(name)%5==0 -> VALIDATION, else DEVELOPMENT.
Stages (each resumable, appends JSONL, <=2 worker processes):
  python evaluate.py base  <out.jsonl> [--split dev|val|all] [--workers 2] [--limit N]
  python evaluate.py dsl   <out.jsonl> [--split ...] [--tcap 12] [--gen-k 20]
The dsl stage stores, per task, every demo-consistent candidate with exact-correctness, a prediction hash and (dev only,
when cache/gen/<task>.json exists) the number of generated examples it gets WRONG / abstains on.
"""
import os, sys, json, glob, time, zlib, hashlib, itertools, argparse
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapter  # noqa: F401  (puts research/claude_arc and the DSL dir on sys.path)
from cells import CELLS
from fitted import fit_local_rule, fit_role_rule
from fitted2 import fit_shape_changing
import objfam

ROOT = os.environ.get("ARC2", "C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/"
                      "48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc2/data")
GEN = os.path.join(HERE, "cache", "gen")


def bucket(name):
    return zlib.crc32(name.encode()) % 5


def split_of(name):
    return "val" if bucket(name) == 0 else "dev"


def phash(preds):
    return hashlib.md5(b"".join(p.tobytes() + bytes(p.shape) for p in preds)).hexdigest()[:12]


def load(f):
    d = json.load(open(f))
    tr = [(np.array(e["input"]), np.array(e["output"])) for e in d["train"]]
    te = [np.array(e["input"]) for e in d["test"]]
    tg = [np.array(e["output"]) for e in d["test"]]
    return tr, te, tg


def ok_all(preds, tg):
    return all(p is not None and p.shape == t.shape and np.array_equal(p, t) for p, t in zip(preds, tg))


# ---------------------------------------------------------------- baseline (copy of solve.search + fitted cells)
def _safe(fn, g):
    try:
        o = np.asarray(fn(g))
        if o.ndim != 2 or o.size == 0 or o.shape[0] > 30 or o.shape[1] > 30:
            return None
        return o
    except Exception:
        return None


def _fit_cm(preds, outs):
    m = {}
    for p, y in zip(preds, outs):
        if p.shape != y.shape:
            return None
        for a, b in zip(p.ravel(), y.ravel()):
            if m.setdefault(int(a), int(b)) != int(b):
                return None
    return m


def _app(g, m):
    o = g.copy()
    for a, b in m.items():
        o[g == a] = b
    return o


def _run(chain, g):
    for n in chain:
        g = _safe(CELLS[n], g)
        if g is None:
            return None
    return g


def baseline_search(train, test_in, max_depth=2, tlimit=20.0):
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
            cur = [_run(chain, g) for g in ins]
            if any(c is None for c in cur):
                continue
            key = tuple(c.tobytes() + bytes(c.shape) for c in cur)
            if key in seen:
                continue
            seen.add(key)
            ok = all(c.shape == y.shape and np.array_equal(c, y) for c, y in zip(cur, outs))
            cm = None
            if not ok:
                cm = _fit_cm(cur, outs)
                if not cm or all(a == b for a, b in cm.items()):
                    continue
            preds = []
            for tg in test_in:
                o = _run(chain, tg)
                if o is not None and cm is not None:
                    o = _app(o, cm)
                preds.append(o)
            if all(p is not None for p in preds):
                sols.append(("+".join(chain), preds))
                if len(sols) >= 4:
                    return sols
    return sols


def base_task(f):
    name = os.path.basename(f)[:-5]
    tr, te, tg = load(f)
    t0 = time.time()
    sols = baseline_search(tr, te)
    for fn in (fit_local_rule, fit_role_rule, fit_shape_changing):
        try:
            for kind, preds in fn(tr, te):
                sols.append(("fit:" + kind, preds))
        except Exception:
            pass
    return {"task": name, "split": split_of(name), "sec": round(time.time() - t0, 2),
            "sols": [{"desc": d, "ok": bool(ok_all(p, tg)), "ph": phash(p)} for d, p in sols[:12]]}


# ---------------------------------------------------------------- dsl_port
def gen_examples(name, k):
    fp = os.path.join(GEN, name + ".json")
    if split_of(name) != "dev" or not os.path.exists(fp):
        return []
    ex = json.load(open(fp))[:k]
    return [(np.array(e["input"]), np.array(e["output"])) for e in ex]


def dsl_task(a):
    f, tcap, gen_k = a
    name = os.path.basename(f)[:-5]
    tr, te, tg = load(f)
    t0 = time.time()
    cands = objfam.run_all(tr, te, tcap)
    sec = time.time() - t0
    cands.sort(key=lambda c: c.cost)
    gex = gen_examples(name, gen_k)
    rows = []
    for c in cands[:400]:
        wrong = abst = 0
        for gi, go in gex:
            try:
                p = c.prog(gi)
            except Exception:
                p = None
            if p is None:
                abst += 1
            elif p.shape != go.shape or not np.array_equal(p, go):
                wrong += 1
        rows.append({"fam": c.family, "desc": c.desc, "cost": c.cost, "ok": bool(ok_all(c.preds, tg)), "ph": phash(c.preds),
                     "gwrong": wrong, "gabst": abst})
    return {"task": name, "split": split_of(name), "sec": round(sec, 2), "n_gen": len(gex), "n_cands_total": len(cands), "cands": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "dsl"])
    ap.add_argument("out")
    ap.add_argument("--split", default="all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tcap", type=float, default=12.0)
    ap.add_argument("--gen-k", type=int, default=20)
    a = ap.parse_args()
    workers = min(2, a.workers)
    files = sorted(glob.glob(ROOT + "/training/*.json"))
    files = [f for f in files if a.split in ("all", split_of(os.path.basename(f)[:-5]))]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["task"] for l in open(a.out) if l.strip()}
    files = [f for f in files if os.path.basename(f)[:-5] not in done][:a.limit]
    fn, args = (base_task, files) if a.stage == "base" else (dsl_task, [(f, a.tcap, a.gen_k) for f in files])
    t0 = time.time()
    with open(a.out, "a") as fo, Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(fn, args, chunksize=1)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if i % 50 == 0:
                print(i, len(files), round(time.time() - t0), flush=True)
    print("done", len(files), round(time.time() - t0))


if __name__ == "__main__":
    main()
