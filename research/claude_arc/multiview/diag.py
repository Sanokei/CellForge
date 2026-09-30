"""Why are multiview rules rarely accepted? Stage counts over (task, feature-subset) on VALIDATION same-shape tasks.
Usage: python diag.py [limit]"""
import sys, json, itertools, numpy as np
from collections import Counter
import cells_mv as m
from views import all_views, FAM
from precompute import files
lim = int(sys.argv[1]) if len(sys.argv) > 1 else 60
C = Counter(); tasks_cons = 0; n_tasks = 0
for f in files("val")[:400]:
    d = json.load(open(f))
    tr = [(np.array(e["input"]), np.array(e["output"])) for e in d["train"]]
    te = [np.array(e["input"]) for e in d["test"]]
    if not all(x.shape == y.shape for x, y in tr): continue
    n_tasks += 1
    if n_tasks > lim: break
    pools = []; fn = sn = None
    for x, _ in tr:
        fn, F, sn, S = all_views(x); pools.append((F, S))
    tp = [all_views(x)[1::2] for x in te]
    Ftr = np.concatenate([F.reshape(-1, F.shape[-1]) for F, _ in pools])[None]
    Str = np.concatenate([S.reshape(-1, S.shape[-1]) for _, S in pools])[None]
    y = np.concatenate([yy.reshape(-1) for _, yy in tr])
    E = m._tokens(Str[0], y)
    if E[:, 0].all(): C["identity task"] += 1; continue
    subs = [(i,) for i in range(len(fn))] + [(i, j) for i, j in itertools.combinations(range(len(fn)), 2) if FAM[fn[i]] != FAM[fn[j]]]
    anyc = False
    for idx in subs:
        keys = m._encode(Ftr, idx).reshape(-1); uk, inv = np.unique(keys, return_inverse=True)
        bad = np.stack([np.bincount(inv, weights=(~E[:, t]).astype(float), minlength=len(uk)) for t in range(E.shape[1])], 1)
        valid = bad == 0
        if not valid.any(1).all(): C["conflict"] += 1; continue
        if (np.argmax(valid, 1) == 0).all(): C["identity-rule"] += 1; continue
        anyc = True
        if len(uk) > m.MAXKEYS or len(uk) * 3 > len(keys): C["too many keys (overfit guard)"] += 1; continue
        r1 = [m._predict(F, S, idx, uk, valid, False) for F, S in tp]
        if any(p is None for p, _ in r1): C["uncovered/undefined test key"] += 1; continue
        r2 = [m._predict(F, S, idx, uk, valid, True) for F, S in tp]
        if any(not np.array_equal(a[0], b[0]) for a, b in zip(r1, r2)): C["ambiguous token"] += 1; continue
        C["accepted"] += 1
    tasks_cons += anyc
print("same-shape non-identity tasks:", n_tasks - C["identity task"], "; tasks with any demo-consistent non-identity subset:", tasks_cons)
for k, v in C.most_common(): print(f"{v:8d}  {k}")
