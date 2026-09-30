"""Shared helpers for evo/: data split, chain execution, time-capped ordered search.
Imports only cells.py, fitted.py, fitted2.py (never solve.py/objects*.py, which are being edited elsewhere)."""
import os, sys, glob, json, time, zlib, hashlib, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from cells import CELLS, components, bg_color          # noqa: E402
from fitted import fit_local_rule, fit_role_rule        # noqa: E402
from fitted2 import fit_shape_changing                  # noqa: E402

ROOT = os.environ.get("ARC2_TRAIN", "C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/"
                      "48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc2/data/training")
NAMES = list(CELLS)


def bucket(name):
    return zlib.crc32(name.encode()) % 5


def task_files(split):
    """split: 'val' (bucket 0) or 'dev' (buckets 1-4). TRAINING tasks only."""
    fs = sorted(glob.glob(f"{ROOT}/*.json"))
    if split == "val":
        return [f for f in fs if bucket(os.path.basename(f)) == 0]
    return [f for f in fs if bucket(os.path.basename(f)) != 0]


def load(f):
    d = json.load(open(f))
    tr = [(np.array(e["input"]), np.array(e["output"])) for e in d["train"]]
    te = [np.array(e["input"]) for e in d["test"]]
    tg = [np.array(e["output"]) for e in d["test"]]
    return tr, te, tg


def safe(fn, g):
    try:
        o = np.asarray(fn(g))
        if o.ndim != 2 or o.size == 0 or o.shape[0] > 30 or o.shape[1] > 30:
            return None
        return o
    except Exception:
        return None


def fit_colormap(preds, outs):
    m = {}
    for p, y in zip(preds, outs):
        if p.shape != y.shape:
            return None
        for a, b in zip(p.ravel().tolist(), y.ravel().tolist()):
            if m.setdefault(a, b) != b:
                return None
    return m


def apply_map(g, m):
    o = g.copy()
    for a, b in m.items():
        o[g == a] = b
    return o


def ghash(g):
    if g is None:
        return "N"
    return hashlib.md5(g.tobytes() + bytes(g.shape)).hexdigest()[:10]


class Runner:
    """Per-task executor with memoised outputs (depth<=2 prefixes) so chains share work."""
    def __init__(self, grids):
        self.grids = grids
        self.memo = {}

    def outs(self, chain):
        if chain in self.memo:
            return self.memo[chain]
        if len(chain) == 1:
            r = [safe(CELLS[chain[0]], g) for g in self.grids]
        else:
            prev = self.outs(chain[:-1])
            r = [None if p is None else safe(CELLS[chain[-1]], p) for p in prev]
        if len(chain) <= 2:
            self.memo[chain] = r
        return r


def run_chain(chain, g):
    for n in chain:
        g = safe(CELLS[n], g)
        if g is None:
            return None
    return g


def natural_chains(depth=2):
    import itertools
    out = []
    for d in range(1, depth + 1):
        for ch in itertools.product(NAMES, repeat=d):
            if d > 1 and ch[0] == "id":
                continue
            out.append(ch)
    return out


def consistent(chain, train, test_in, run=None):
    """None, or (colormap, test_preds) if the chain (optionally + fitted colour map) reproduces all demos."""
    outs = [y for _, y in train]
    n = len(train)
    if run is None:
        run = Runner([x for x, _ in train] + list(test_in))
    cur = run.outs(chain)
    if any(c is None for c in cur[:n]):
        return None
    if any(c.shape != y.shape for c, y in zip(cur[:n], outs)):
        return None
    ok = all(np.array_equal(c, y) for c, y in zip(cur[:n], outs))
    cm = None
    if not ok:
        cm = fit_colormap(cur[:n], outs)
        if not cm or all(a == b for a, b in cm.items()):
            return None
    preds = cur[n:]
    if any(p is None for p in preds):
        return None
    if cm is not None:
        preds = [apply_map(p, cm) for p in preds]
    return cm, preds


def fitted_attempts(train, te):
    """Fitted-cell routines (objects routine excluded: file under concurrent edit). Returns [(name, preds)]."""
    res = []
    for kind, p in fit_local_rule(train, te):
        res.append(("fit:" + kind, p))
    for kind, p in fit_role_rule(train, te):
        res.append(("fit:" + kind, p))
    for kind, p in fit_shape_changing(train, te):
        res.append(("fit:" + kind, p))
    return res


def same(pa, pb):
    return len(pa) == len(pb) and all(np.array_equal(a, b) for a, b in zip(pa, pb))


def ordered_search(train, te, tg, chain_list, tlimit, fit_cache=None, max_sols=4):
    """Try chains in the given order until tlimit seconds or max_sols demo-consistent chains; then fitted routines.
    Returns dict(solved, fp, n_sols, sec, tried, first_rank)."""
    t0 = time.time()
    run = Runner([x for x, _ in train] + list(te))
    sols, tried, first_rank = [], 0, None
    for i, ch in enumerate(chain_list):
        if time.time() - t0 > tlimit:
            break
        tried += 1
        r = consistent(ch, train, te, run)
        if r is None:
            continue
        if any(same(r[1], s[1]) for s in sols):
            continue
        sols.append((ch, r[1]))
        if first_rank is None:
            first_rank = i
        if len(sols) >= max_sols:
            break
    if fit_cache:
        sols += fit_cache
    attempts = []
    for _, preds in sols:
        if not any(same(preds, a) for a in attempts):
            attempts.append(preds)
        if len(attempts) == 2:
            break
    solved = any(same(a, tg) for a in attempts)
    return dict(solved=bool(solved), fp=bool(attempts and not solved), n_sols=len(sols),
                sec=round(time.time() - t0, 3), tried=tried, first_rank=first_rank)


FIT_CACHE_FILE = os.path.join(HERE, "fit_cache.pkl")


def _fit_one(f):
    tr, te, tg = load(f)
    try:
        return os.path.basename(f), fitted_attempts(tr, te)
    except Exception:
        return os.path.basename(f), []


def fit_cache_parallel(files, workers=3):
    """Fitted routines are identical across arms; compute once per task and cache."""
    from multiprocessing import Pool
    cache = pickle.load(open(FIT_CACHE_FILE, "rb")) if os.path.exists(FIT_CACHE_FILE) else {}
    todo = [f for f in files if os.path.basename(f) not in cache]
    if todo:
        with Pool(workers) as p:
            for k, v in p.imap_unordered(_fit_one, todo, chunksize=4):
                cache[k] = v
        pickle.dump(cache, open(FIT_CACHE_FILE, "wb"))
    return cache


def niche_key(train):
    """Niche from demo pairs only (no test outputs): shape relation | object-count bucket | colour-count bucket."""
    if all(x.shape == y.shape for x, y in train):
        rel = "eq"
    else:
        ra = np.mean([y.size / x.size for x, y in train])
        rel = "big" if ra > 1.05 else ("small" if ra < 0.95 else "mixed")
    try:
        no = np.mean([len(components(x)) for x, _ in train])
    except Exception:
        no = 0
    ob = "o0-2" if no <= 2 else "o3-6" if no <= 6 else "o7-14" if no <= 14 else "o15+"
    nc = np.mean([len(np.unique(x)) for x, _ in train])
    cb = "c<=2" if nc <= 2.5 else "c3-4" if nc <= 4.5 else "c5+"
    return f"{rel}|{ob}|{cb}"
