"""Mechanism 2: Red Queen arms race between task generators and solver chains (no gradients, no learned weights).

  python redqueen.py evolve <generations> <seed>   -> rq_seed<S>.json (arms race) + ctrl_seed<S>.json (matched-budget control)
  python redqueen.py val <cap_seconds> <seeds...>  -> mech2_val_cap<cap>.json (real training tasks: VALIDATION, plus DEV secondary)

Generator = chain of 1-3 library cells (+ optional random colour map) applied to random grids (<=20x20).
Solver    = chain of 1-3 library cells (optionally with task-level fitted colour map) or the special ("FIT",) which runs the
            fitted-cell routines (fitted.py / fitted2.py).
"""
import sys, os, json, time, random
from dataclasses import dataclass
import numpy as np
from common import *

FIT = ("FIT",)
_FITC = {}


@dataclass(frozen=True)
class Generator:
    chain: tuple
    cmap: tuple = ()          # ((from, to), ...)

    def __init__(self, chain, cmap=None):
        object.__setattr__(self, "chain", tuple(chain))
        object.__setattr__(self, "cmap", tuple(sorted((cmap or {}).items())) if isinstance(cmap, dict) else tuple(cmap or ()))

    def apply(self, x):
        y = run_chain(self.chain, x)
        if y is None:
            return None
        return apply_map(y, dict(self.cmap)) if self.cmap else y


# ---- random grids ----------------------------------------------------------------------------------------------
def random_input(rng):
    h, w = int(rng.randint(3, 21)), int(rng.randint(3, 21))
    kind = rng.randint(3)
    g = np.zeros((h, w), dtype=int)
    if kind == 0:                                             # random sparse grid
        k = rng.randint(1, 5)
        cols = rng.choice(np.arange(1, 10), k, replace=False)
        m = rng.rand(h, w) < rng.uniform(0.05, 0.35)
        g[m] = rng.choice(cols, int(m.sum()))
    elif kind == 1:                                           # random objects
        for _ in range(rng.randint(1, 7)):
            oh, ow = rng.randint(1, min(7, h) + 1), rng.randint(1, min(7, w) + 1)
            y0, x0 = rng.randint(0, h - oh + 1), rng.randint(0, w - ow + 1)
            mask = rng.rand(oh, ow) < 0.75
            mask[rng.randint(oh), rng.randint(ow)] = True
            g[y0:y0 + oh, x0:x0 + ow][mask] = rng.randint(1, 10)
    else:                                                     # dense few-colour grid
        cols = rng.choice(np.arange(0, 10), rng.randint(2, 4), replace=False)
        g = rng.choice(cols, (h, w), p=None)
    return g


def make_task(gen, n_demo, seed, tries=40):
    """n_demo demo pairs + 1 test pair (last). None if the generator cannot produce valid pairs."""
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(tries):
        x = random_input(rng)
        y = gen.apply(x)
        if y is None:
            continue
        pairs.append((x, y))
        if len(pairs) == n_demo + 1:
            break
    if len(pairs) < n_demo + 1 or all(x.shape == y.shape and np.array_equal(x, y) for x, y in pairs):
        return None
    if len(set(ghash(y) for _, y in pairs)) < 2:              # constant output carries no rule
        return None
    return pairs


def random_cmap(rng):
    cm = {}
    for _ in range(rng.randint(0, 3)):
        a, b = rng.randint(0, 10), rng.randint(0, 10)
        if a != b:
            cm[int(a)] = int(b)
    return cm


def random_generator(rng, maxd=3):
    d = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])) if maxd == 3 else 1
    chain = tuple(NAMES[rng.randint(len(NAMES))] for _ in range(d))
    return Generator(chain, random_cmap(rng) if rng.rand() < 0.5 else {})


def mutate_generator(g, rng):
    ch, cm = list(g.chain), dict(g.cmap)
    r = rng.rand()
    if r < 0.35:
        ch[rng.randint(len(ch))] = NAMES[rng.randint(len(NAMES))]
    elif r < 0.55 and len(ch) < 3:
        ch.insert(rng.randint(len(ch) + 1), NAMES[rng.randint(len(NAMES))])
    elif r < 0.7 and len(ch) > 1:
        ch.pop(rng.randint(len(ch)))
    else:
        cm = random_cmap(rng)
    return Generator(tuple(ch), cm)


# ---- solvers ---------------------------------------------------------------------------------------------------
def random_solver(rng):
    if rng.rand() < 0.03:
        return FIT
    d = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
    return tuple(NAMES[rng.randint(len(NAMES))] for _ in range(d))


def mutate_solver(s, rng):
    if s == FIT:
        return random_solver(rng)
    ch = list(s)
    r = rng.rand()
    if r < 0.45:
        ch[rng.randint(len(ch))] = NAMES[rng.randint(len(NAMES))]
    elif r < 0.7 and len(ch) < 3:
        ch.insert(rng.randint(len(ch) + 1), NAMES[rng.randint(len(NAMES))])
    elif r < 0.85 and len(ch) > 1:
        ch.pop(rng.randint(len(ch)))
    elif len(ch) > 1:
        i = rng.randint(len(ch) - 1)
        ch[i], ch[i + 1] = ch[i + 1], ch[i]
    return tuple(ch)


def crossover(a, b, rng):
    if a == FIT or b == FIT:
        return mutate_solver(a if b == FIT else b, rng)
    c = a[:rng.randint(1, len(a) + 1)] + b[rng.randint(0, len(b)):]
    return tuple(c[:3]) or a


def cost(s):
    return 3 if s == FIT else len(s)


def solves(solver, task, key=None):
    """Demo-consistent on all but the last pair and correct on the last (held-out) pair."""
    tr, (xt, yt) = task[:-1], task[-1]
    if solver == FIT:
        if key is not None and key in _FITC:
            att = _FITC[key]
        else:
            try:
                att = fitted_attempts(tr, [xt])
            except Exception:
                att = []
            if key is not None:
                _FITC[key] = att
        return any(same(p, [yt]) for _, p in att)
    r = consistent(solver, tr, [xt])
    return r is not None and same(r[1], [yt])


# ---- arms race -------------------------------------------------------------------------------------------------
class Tasks:
    def __init__(self):
        self.items = []          # (Generator, task, sig)
        self.sigs = set()

    @staticmethod
    def sig(gen, probes):
        return tuple(ghash(gen.apply(p)) for p in probes)


def probe_inputs(n=20):
    rng = np.random.RandomState(12345)
    return [random_input(rng) for _ in range(n)]


def evolve(G, seed, P_s=24, P_g=24, cap=60, n_demo=3):
    rng = np.random.RandomState(seed)
    probes = probe_inputs()
    evals = 0
    gens = []                     # dicts: gen, task, sig, id
    gsigs = set()
    hard = []                     # generators solved by no solver so far
    solv = []                     # dicts: chain, cov(set of gen ids), cost
    stats = dict(gen_proposed=0, gen_invalid=0, gen_dup=0, gen_too_easy=0, gen_too_hard=0, gen_added=0, gen_revived=0,
                 sol_proposed=0, sol_added=0, sol_replaced=0)
    hist = []
    nid = 0

    def cov_of(ch):
        nonlocal evals
        c = set()
        for gd in gens:
            evals += 1
            if solves(ch, gd["task"], gd["id"]):
                c.add(gd["id"])
        return c

    def try_gen(g, pool_solvers):
        nonlocal nid, evals
        stats["gen_proposed"] += 1
        tk = make_task(g, n_demo, int(rng.randint(1 << 30)))
        if tk is None:
            stats["gen_invalid"] += 1
            return None
        sg = Tasks.sig(g, probes)
        if sg in gsigs:
            stats["gen_dup"] += 1
            return None
        gid = nid
        nid += 1
        n_ok = 0
        for s in pool_solvers:
            evals += 1
            n_ok += solves(s, tk, gid)
        return dict(gen=g, task=tk, sig=sg, id=gid, n_ok=n_ok, n_sol=len(pool_solvers))

    for gen_i in range(G):
        # ---- solvers: propose, each must solve the generator archive; enters on coverage gain (or cheaper same coverage)
        pop = []
        for _ in range(P_s):
            r = rng.rand()
            if solv and r < 0.5:
                s = mutate_solver(solv[rng.randint(len(solv))]["chain"], rng)
            elif len(solv) > 1 and r < 0.75:
                s = crossover(solv[rng.randint(len(solv))]["chain"], solv[rng.randint(len(solv))]["chain"], rng)
            else:
                s = random_solver(rng)
            pop.append(s)
        for s in pop:
            stats["sol_proposed"] += 1
            if any(d["chain"] == s for d in solv):
                continue
            c = cov_of(s)
            if not c:
                continue
            union = set().union(*[d["cov"] for d in solv]) if solv else set()
            if c - union:
                solv.append(dict(chain=s, cov=c, cost=cost(s)))
                stats["sol_added"] += 1
                continue
            # same-or-superset coverage but cheaper than an archived solver -> replaces it
            for d in solv:
                if c >= d["cov"] and cost(s) < d["cost"]:
                    d.update(chain=s, cov=c, cost=cost(s))
                    stats["sol_replaced"] += 1
                    break
        # ---- generators: new candidates + revisit hard pool; keep 'not too easy, not too hard'
        chains = [d["chain"] for d in solv]
        cands = []
        for _ in range(P_g):
            if gens and rng.rand() < 0.5:
                cands.append(mutate_generator(gens[rng.randint(len(gens))]["gen"], rng))
            else:
                cands.append(random_generator(rng))
        scored = []
        for g in cands:
            e = try_gen(g, chains)
            if e is not None:
                scored.append((e, False))
        revisit = hard[:]
        hard.clear()
        for g in revisit[-30:]:
            stats["gen_proposed"] -= 1
            e = try_gen(g, chains)
            if e is not None:
                scored.append((e, True))
        for e, rev in scored:
            if chains and e["n_ok"] == 0:
                stats["gen_too_hard"] += 1
                hard.append(e["gen"])
            elif chains and e["n_ok"] == e["n_sol"] and e["n_sol"] > 1:
                stats["gen_too_easy"] += 1
            else:
                gens.append(e)
                gsigs.add(e["sig"])
                stats["gen_added"] += 1
                stats["gen_revived"] += int(rev)
                if not chains:
                    pass
        # generator archive cap: drop the oldest extra
        while len(gens) > 80:
            gone = gens.pop(0)
            for d in solv:
                d["cov"].discard(gone["id"])
        # coverage of newly admitted generators by existing archived solvers
        for e, _ in scored:
            if any(x is e for x in gens):
                for d in solv:
                    if solves(d["chain"], e["task"], e["id"]):
                        d["cov"].add(e["id"])
        hist.append(dict(g=gen_i, n_gens=len(gens), n_solv=len(solv), evals=evals,
                         covered=len(set().union(*[d["cov"] for d in solv])) if solv else 0))
    return dict(seed=seed, G=G, stats=stats, evals=evals, hist=hist[::max(1, G // 20)],
                solvers=[dict(chain=list(d["chain"]), cov=len(d["cov"]), cost=d["cost"]) for d in sorted(solv, key=lambda d: -len(d["cov"]))],
                n_generators=len(gens), gen_depth_hist=_hist([len(g["gen"].chain) for g in gens]))


def _hist(v):
    return {str(k): v.count(k) for k in sorted(set(v))}


def control(budget_evals, seed, archive_size, P_s=24, P_g=24, cap=60, n_demo=3):
    """Matched solver-evaluation budget: random generators (no novelty / difficulty filter / mutation) and random solvers
    (no mutation/crossover, no coverage-gain rule). Archive = solvers that solve >=1 task, top by coverage."""
    rng = np.random.RandomState(seed + 1000)
    tasks, evals, cand = [], 0, {}
    nid = 0
    while evals < budget_evals:
        for _ in range(P_g):
            tk = make_task(random_generator(rng), n_demo, int(rng.randint(1 << 30)))
            if tk is not None:
                tasks.append((nid, tk))
                nid += 1
        tasks = tasks[-cap:]
        for _ in range(P_s):
            s = random_solver(rng)
            c = 0
            for gid, tk in tasks:
                evals += 1
                c += solves(s, tk, gid)
            if c:
                cand[s] = max(cand.get(s, 0), c)
    top = sorted(cand.items(), key=lambda kv: (-kv[1], cost(kv[0])))[:archive_size]
    return dict(seed=seed, evals=evals, solvers=[dict(chain=list(c), cov=n, cost=cost(c)) for c, n in top],
                n_candidates=len(cand))


# ---- validation on real training tasks -------------------------------------------------------------------------
def _val_task(a):
    f, order, cap, fc = a
    tr, te, tg = load(f)
    r = ordered_search(tr, te, tg, order, cap, fit_cache=fc.get(os.path.basename(f)))
    return r


def val(cap, seeds, workers=3):
    from multiprocessing import Pool
    base = natural_chains(2)
    bset = set(base)
    arms = {"baseline": base}
    for s in seeds:
        rq = json.load(open(os.path.join(HERE, f"rq_seed{s}.json")))
        ct = json.load(open(os.path.join(HERE, f"ctrl_seed{s}.json")))
        for nm, d in (("redqueen", rq), ("control", ct)):
            pre = [tuple(x["chain"]) for x in d["solvers"] if x["chain"] != ["FIT"]]
            arms[f"{nm}_s{s}"] = pre + [c for c in base if c not in set(pre)]
            arms[f"{nm}_s{s}_ext"] = [c for c in pre if c not in bset]      # n extension chains (depth 3)
    files_v, files_d = task_files("val"), task_files("dev")
    fc = fit_cache_parallel(files_v + files_d, workers)
    out = {}
    with Pool(workers) as p:
        for split, files in (("VALIDATION(bucket0)", files_v), ("DEV-secondary(buckets1-4; archives never selected on real tasks)", files_d)):
            for arm, order in arms.items():
                if arm.endswith("_ext"):
                    continue
                t0 = time.time()
                res = p.map(_val_task, [(f, order, cap, fc) for f in files], chunksize=4)
                out[f"{split}|{arm}"] = dict(n=len(res), solved=sum(r["solved"] for r in res), false_pos=sum(r["fp"] for r in res),
                                             sec_per_task=round(float(np.mean([r["sec"] for r in res])), 3),
                                             n_depth3_in_front=len([c for c in order[:len(arms.get(arm + '_ext', []))] if len(c) == 3]),
                                             wall=round(time.time() - t0))
                print(split[:5], arm, out[f"{split}|{arm}"], flush=True)
    json.dump(out, open(os.path.join(HERE, f"mech2_val_cap{cap}.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "evolve":
        G, seed = int(sys.argv[2]), int(sys.argv[3])
        t0 = time.time()
        rq = evolve(G, seed)
        rq["sec"] = round(time.time() - t0)
        json.dump(rq, open(os.path.join(HERE, f"rq_seed{seed}.json"), "w"), indent=1)
        t0 = time.time()
        ct = control(rq["evals"], seed, len(rq["solvers"]))
        ct["sec"] = round(time.time() - t0)
        json.dump(ct, open(os.path.join(HERE, f"ctrl_seed{seed}.json"), "w"), indent=1)
        print("RQ", {k: rq[k] for k in ("evals", "n_generators", "gen_depth_hist", "stats", "sec")}, "solvers", len(rq["solvers"]))
        print("CTRL", {k: ct[k] for k in ("evals", "n_candidates", "sec")}, "solvers", len(ct["solvers"]))
    else:
        val(float(sys.argv[2]), [int(s) for s in sys.argv[3:]])
