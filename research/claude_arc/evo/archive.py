"""Mechanism 1: behavioural-novelty rejection + MAP-Elites style niche archive over depth<=2 chains.

Steps (run `python archive.py build` then `python archive.py eval`):
  build: probe-set signatures -> unique chains -> per-DEV-task solve sets -> niche archive -> archive.json
  eval : VALIDATION bucket, same per-task time cap, arms: baseline order / novelty-dedup order / niche-archive order.
"""
import sys, os, json, time, random
from multiprocessing import Pool
import numpy as np
from common import *


# ---- signature / novelty ---------------------------------------------------------------------------------------
def signature(chain, probes):
    """Tuple of output hashes of the chain on each probe grid ('N' = cell failed / invalid)."""
    return tuple(ghash(g) for g in _sig_outs(chain, probes))


def _sig_outs(chain, probes):
    return [run_chain(chain, g) for g in probes]


def similarity(a, b):
    """Hamming-style similarity = fraction of probes with identical output hash."""
    return sum(x == y for x, y in zip(a, b)) / len(a)


class NoveltyArchive:
    """Rejects a candidate if its signature equals an archived one ('dup'); counts near-dups (similarity >= near)."""
    def __init__(self, probes, near=0.98):
        self.probes, self.near = probes, near
        self.sigs, self.chains, self.exact = [], [], {}
        self.stats = dict(new=0, dup=0, dead=0, near=0)

    def add(self, chain, sig=None):
        sig = sig or signature(chain, self.probes)
        if all(s == "N" for s in sig):
            self.stats["dead"] += 1
            return "dead"
        if sig in self.exact:
            self.stats["dup"] += 1
            return "dup"
        if any(similarity(sig, s) >= self.near for s in self.sigs):
            self.stats["near"] += 1                # still admitted (see README: near-dup rejection hurts colour-specific cells)
        self.exact[sig] = chain
        self.sigs.append(sig)
        self.chains.append(chain)
        self.stats["new"] += 1
        return "new"


class NicheArchive:
    def __init__(self, k=8):
        self.k, self.cells = k, {}

    def update(self, niche, chain, score):
        lst = self.cells.setdefault(niche, {})
        lst[chain] = score

    def top(self, niche):
        d = self.cells.get(niche, {})
        rk = sorted(d.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
        return [c for c, s in rk[:self.k] if s > 0]


def make_probes(n=150):
    fs = task_files("dev")
    step = max(1, len(fs) // n)
    probes = []
    for f in fs[::step][:n]:
        tr, _, _ = load(f)
        probes.append(tr[0][0])
    return probes


# ---- dev solve table -------------------------------------------------------------------------------------------
def _dev_task(a):
    f, chains, cap = a
    tr, te, tg = load(f)
    run = Runner([x for x, _ in tr] + list(te))
    t0 = time.time()
    solves, cons = [], []
    for i, ch in enumerate(chains):
        if time.time() - t0 > cap:
            break
        r = consistent(ch, tr, te, run)
        if r is None:
            continue
        cons.append(i)
        if same(r[1], tg):
            solves.append(i)
    return os.path.basename(f), niche_key(tr), solves, cons, round(time.time() - t0, 1)


def build(k=8, cap=25.0, workers=3):
    t0 = time.time()
    probes = make_probes(150)
    allc = natural_chains(2)
    na = NoveltyArchive(probes, near=0.98)
    for ch in allc:
        na.add(ch)
    uniq = na.chains
    # near-duplicate rate at several thresholds (over unique chains, vs earlier unique chains)
    nd = {}
    for thr in (0.90, 0.95, 0.98, 0.99):
        c = sum(any(similarity(s, na.sigs[j]) >= thr for j in range(i)) for i, s in enumerate(na.sigs))
        nd[str(thr)] = round(c / len(na.sigs), 4)
    files = task_files("dev")
    with Pool(workers) as p:
        res = p.map(_dev_task, [(f, uniq, cap) for f in files], chunksize=2)
    # niche archive: score = number of DEV tasks solved in that niche
    nich = NicheArchive(k)
    glob_score = {}
    for tname, nk, solves, cons, _ in res:
        for i in solves:
            ch = uniq[i]
            glob_score[ch] = glob_score.get(ch, 0) + 1
    per = {}
    for tname, nk, solves, cons, _ in res:
        for i in solves:
            per.setdefault(nk, {}).setdefault(uniq[i], 0)
            per[nk][uniq[i]] += 1
    for nk, d in per.items():
        for ch, s in d.items():
            nich.update(nk, ch, s)
    # near-dup ablation on DEV: tasks solved only by chains that a 0.98-near-dup filter would drop
    order_idx = {c: i for i, c in enumerate(uniq)}
    dropped = set(i for i, s in enumerate(na.sigs) if any(similarity(s, na.sigs[j]) >= 0.98 for j in range(i)))
    lost = sum(1 for _, _, solves, _, _ in res if solves and all(i in dropped for i in solves))
    dev_any = sum(1 for _, _, solves, _, _ in res if solves)
    out = dict(n_candidates=len(allc), n_unique=len(uniq), stats=na.stats, near_dup_rate_of_unique=nd,
               dev_tasks=len(files), dev_tasks_solved_by_any_chain_cap=dev_any,
               dev_tasks_lost_if_near098_rejected=lost,
               niches={k2: [list(c) for c in nich.top(k2)] for k2 in nich.cells},
               niche_sizes={nk: sum(1 for r in res if r[1] == nk) for nk in set(r[1] for r in res)},
               global_rank=[[list(c), s] for c, s in sorted(glob_score.items(), key=lambda kv: -kv[1])],
               uniq=[list(c) for c in uniq], build_sec=round(time.time() - t0))
    json.dump(out, open(os.path.join(HERE, "archive.json"), "w"))
    print({k2: v for k2, v in out.items() if k2 not in ("niches", "global_rank", "uniq", "niche_sizes")})


# ---- validation ------------------------------------------------------------------------------------------------
def ordering(arm, nk, A):
    uniq = [tuple(c) for c in A["uniq"]]
    if arm == "baseline":
        return natural_chains(2)
    if arm == "dedup":
        return uniq
    top = [tuple(c) for c in A["niches"].get(nk, [])]
    gl = [tuple(c) for c, _ in A["global_rank"]]
    seen, out = set(), []
    for c in top + gl + uniq:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _val_task(a):
    f, arm, cap, A, fc = a
    tr, te, tg = load(f)
    r = ordering_cache(arm, niche_key(tr), A)
    res = ordered_search(tr, te, tg, r, cap, fit_cache=fc.get(os.path.basename(f)))
    res["task"] = os.path.basename(f)
    return res


_OC = {}


def ordering_cache(arm, nk, A):
    k = (arm, nk)
    if k not in _OC:
        _OC[k] = ordering(arm, nk, A)
    return _OC[k]


def evaluate(cap, workers=3):
    A = json.load(open(os.path.join(HERE, "archive.json")))
    fc = fit_cache_parallel(task_files("val"), workers)
    files = task_files("val")
    out = {}
    for arm in ("baseline", "dedup", "niche"):
        t0 = time.time()
        with Pool(workers) as p:
            res = p.map(_val_task, [(f, arm, cap, A, fc) for f in files], chunksize=2)
        out[arm] = dict(split="VALIDATION(bucket0)", n=len(res), solved=sum(r["solved"] for r in res),
                        false_pos=sum(r["fp"] for r in res), sec_per_task=round(float(np.mean([r["sec"] for r in res])), 3),
                        wall=round(time.time() - t0), cap=cap, mean_tried=round(float(np.mean([r["tried"] for r in res])), 1))
        print(arm, out[arm], flush=True)
    json.dump(out, open(os.path.join(HERE, f"mech1_val_cap{cap}.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        evaluate(float(sys.argv[2]))
