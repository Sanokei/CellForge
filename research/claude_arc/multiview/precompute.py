"""Per-task candidate pools for every arm (computed ONCE per task, arms are derived post hoc in analyze.py).
Existing stack (solve.py semantics): depth<=2 enumeration over cells.CELLS with demo verification, then fit_local_rule, fit_role_rule,
fit_shape_changing, fit_objects. New: multi-view fitters (cells_mv). Only TRAINING tasks are read.
Usage: python precompute.py <dev|val|all> <out.jsonl> [start stride] [limit]
"""
import sys, os, json, glob, time, zlib, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from cells import CELLS                                   # noqa: E402
from solve import safe, fit_colormap, apply_map           # noqa: E402
from fitted import fit_local_rule, fit_role_rule          # noqa: E402
from fitted2 import fit_shape_changing                    # noqa: E402
from objects import fit_objects                           # noqa: E402
from cells_mv import fit_multiview                        # noqa: E402

ROOT = os.environ.get("ARC2_TRAIN", "C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/"
                      "48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc2/data/training")
NAMES = list(CELLS)


def bucket(name):
    return zlib.crc32(name.encode()) % 5


def files(split):
    fs = sorted(glob.glob(f"{ROOT}/*.json"))
    b = [bucket(os.path.basename(f)) for f in fs]
    if split == "val":
        return [f for f, k in zip(fs, b) if k == 0]
    if split == "dev":
        return [f for f, k in zip(fs, b) if k != 0]
    return fs


def sim(p, y):
    return float((p == y).mean()) if p.shape == y.shape else 0.0


def chain_candidates(ins, outs, tests, tlimit=20.0):
    """All demo-consistent chains (depth1 first, then depth2) with first-cell prefilter rank and per-first-cell time."""
    t0 = time.time()
    c1 = {n: [safe(CELLS[n], g) for g in ins] for n in NAMES}
    score = {}
    for n in NAMES:
        s = c1[n]
        score[n] = -1.0 if any(v is None for v in s) else float(np.mean([sim(v, y) for v, y in zip(s, outs)]))
    order = sorted(NAMES, key=lambda n: (-score[n], NAMES.index(n)))   # prefilter rank (0 = most similar to target)
    rank = {n: i for i, n in enumerate(order)}
    t_d1 = time.time() - t0
    cands, seen, block_t = [], set(), {}

    def consider(chain, cur, depth, r1):
        key = tuple(c.tobytes() + bytes(c.shape) for c in cur)
        if key in seen:
            return
        seen.add(key)
        ok = all(c.shape == y.shape and np.array_equal(c, y) for c, y in zip(cur, outs))
        cm = None
        if not ok:
            cm = fit_colormap(cur, outs)
            if not cm or all(a == b for a, b in cm.items()):
                return
        preds = []
        for tg in tests:
            o = run(chain, tg)
            if o is not None and cm is not None:
                o = apply_map(o, cm)
            preds.append(o)
        if all(p is not None for p in preds):
            cands.append(dict(fam="chain", depth=depth, r1=r1, chain=list(chain), preds=preds))

    def run(chain, g):
        for n in chain:
            g = safe(CELLS[n], g)
            if g is None:
                return None
        return g

    for n in NAMES:                       # depth 1 (original order)
        if any(v is None for v in c1[n]):
            continue
        consider((n,), c1[n], 1, rank[n])
    for n1 in NAMES:                      # depth 2
        if n1 == "id" or any(v is None for v in c1[n1]):
            continue
        tb = time.time()
        for n2 in NAMES:
            if time.time() - t0 > tlimit:
                break
            cur = [safe(CELLS[n2], g) for g in c1[n1]]
            if any(c is None for c in cur):
                continue
            consider((n1, n2), cur, 2, rank[n1])
        block_t[n1] = time.time() - tb
    return cands, t_d1, {rank[n]: t for n, t in block_t.items()}


def process(f):
    d = json.load(open(f))
    tr = [(np.array(e["input"]), np.array(e["output"])) for e in d["train"]]
    te = [np.array(e["input"]) for e in d["test"]]
    tgt = [np.array(e["output"]) for e in d["test"]]
    ins = [x for x, _ in tr]; outs = [y for _, y in tr]
    rec = dict(task=os.path.basename(f), bucket=bucket(os.path.basename(f)), cands=[])

    def add(fam, idx, preds, **kw):
        ok = all(p.shape == t.shape and np.array_equal(p, t) for p, t in zip(preds, tgt))
        sig = zlib.crc32(b"".join(p.tobytes() + bytes(p.shape) for p in preds))
        rec["cands"].append(dict(fam=fam, idx=idx, ok=bool(ok), sig=sig, **kw))

    t0 = time.time()
    ch, t_d1, block_t = chain_candidates(ins, outs, te)
    for i, c in enumerate(ch):
        add("chain", i, c["preds"], depth=c["depth"], r1=c["r1"])
    rec["t_chain_d1"] = t_d1; rec["t_chain_blocks"] = {str(k): v for k, v in block_t.items()}
    rec["t_chain"] = time.time() - t0
    t0 = time.time()
    for fam, fn in (("fit_local", fit_local_rule), ("fit_role", fit_role_rule), ("fit_shape", fit_shape_changing),
                    ("obj", fit_objects)):
        try:
            for i, (kind, preds) in enumerate(fn(tr, te)):
                add(fam, i, preds, kind=kind)
        except Exception:
            pass
    rec["t_fit"] = time.time() - t0
    t0 = time.time()
    mv = fit_multiview(tr, te)
    rec["t_mv"] = time.time() - t0
    cnt = {}
    for c in mv:
        k = c["kind"].split(":")[0]
        i = cnt.get(k, 0); cnt[k] = i + 1
        add("mv_" + k, i, c["preds"], kind=c["kind"], nfeat=c["nfeat"], nkeys=c["nkeys"], fams=c["fams"], sfams=c["sfams"])
    return rec


if __name__ == "__main__":
    split, out = sys.argv[1], sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    lim = int(sys.argv[5]) if len(sys.argv) > 5 else None
    fs = files(split)[start::stride][:lim]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["task"] for l in open(out)}
    with open(out, "a") as fo:
        for n, f in enumerate(fs):
            if os.path.basename(f) in done:
                continue
            r = process(f)
            fo.write(json.dumps(r) + "\n"); fo.flush()
            if n % 20 == 0:
                print(n, len(fs), r["task"], round(r["t_chain"], 1), round(r["t_fit"], 1), round(r["t_mv"], 1), flush=True)
