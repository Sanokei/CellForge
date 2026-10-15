"""Semantic anti-cheat checks for runs/weights-*/ (docs/ONE_STANDARD.md). Owner-controlled.

Each check takes a run directory and returns a list of problem strings (empty = pass). Checks recompute claims from the raw
weights instead of trusting fields in STANDARD.json. tests/test_standard_semantic.py applies them to real runs and to synthetic
cheating runs (meta-tests proving each check catches its cheat).

Run layout (extends tests/test_one_standard.py):
  STANDARD.json  + "evaluator": "<importable module with make_tasks(seed, n) and predict(tensors, input)>",
                   "eval_seeds": [...], "maker_seen_seeds": [...], "random_vocab": {"tensors": [...], "low": x, "high": y},
                   "ledger_file": "ledger.jsonl"
  genomes[i] = {hash, niche, origin, parents, weights_file, accuracy, generation, arm, provenance}
  provenance/<id>.json = {id, kind: founder|llm_mutation|random_mutation|crossover, arm, seed, niche, actor, ts_outbox, ts_inbox, ts_eval,
                          shown: {counterexample_inputs: [...]}, response: {tensors}, rationale, rng_seed}
  ledger.jsonl: one JSON per line with {"prev": <sha of previous line or "GENESIS">, "event": {...}}
"""
import hashlib
import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ACC_TOL = 0.03
EPS_DISTINCT = 1e-4
N_EVAL = 200


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def weights_of(base, rel):
    d = load(os.path.join(base, rel))
    return d.get("tensors", d)


def flat(x):
    if isinstance(x, dict):
        out = []
        for k in sorted(x):
            out += flat(x[k])
        return out
    if isinstance(x, (list, tuple)):
        out = []
        for i in x:
            out += flat(i)
        return out
    return [float(x)]


def weights_hash(tensors):
    return hashlib.sha256(canon(tensors).encode()).hexdigest()


def entries(run):
    s = load(os.path.join(run, "STANDARD.json"))
    ents = {}
    for f in s.get("founders", []):
        ents[f["hash"]] = dict(f, origin="llm_founder", arm=f.get("arm", "llm"), parents=[], generation=f.get("generation", 0))
    for g in s.get("genomes", []):
        ents.setdefault(g["hash"], g)
    return s, ents


def evaluator(s):
    return importlib.import_module(s["evaluator"])


def accuracy(ev, tensors, seed, n=N_EVAL):
    tasks = ev.make_tasks(seed, n)
    ok = 0
    for x, y in tasks:
        ok += 1 if ev.predict(tensors, x) == y else 0
    return ok / max(1, len(tasks))


def check_hash_integrity(run):
    s, ents = entries(run)
    bad = []
    for h, e in ents.items():
        t = weights_of(run, e["weights_file"])
        if weights_hash(t) != h:
            bad.append("hash mismatch for genome %s (hash is not the content hash of its weights)" % h[:12])
    return bad


def check_accuracy_recomputed(run, tol=ACC_TOL):
    s, ents = entries(run)
    ev = evaluator(s)
    bad = []
    for h, e in ents.items():
        if e.get("accuracy") is None:
            continue
        t = weights_of(run, e["weights_file"])
        accs = [accuracy(ev, t, sd) for sd in s["eval_seeds"]]
        mean = sum(accs) / len(accs)
        if e["accuracy"] > mean + tol:
            bad.append("claimed accuracy %.3f exceeds recomputed %.3f for %s" % (e["accuracy"], mean, h[:12]))
    return bad


def check_task_not_trivial(run):
    """A trained genome must beat zero-weights and scrambled-weights on the same evaluator, otherwise the task/evaluator proves nothing."""
    s, ents = entries(run)
    ev = evaluator(s)
    bad = []
    by_niche = {}
    for h, e in ents.items():
        t = weights_of(run, e["weights_file"])
        acc = accuracy(ev, t, s["eval_seeds"][0])
        by_niche[e["niche"]] = max(by_niche.get(e["niche"], 0), acc)
    anyw = next(iter(ents.values()))
    t0 = weights_of(run, anyw["weights_file"])
    zero = {k: _zeros_like(v) for k, v in t0.items()}
    base = accuracy(ev, zero, s["eval_seeds"][0])
    for n, a in by_niche.items():
        if a <= base + 0.05:
            bad.append("best genome in niche %s (%.3f) does not beat zero-weights (%.3f): trivial task or evaluator" % (n, a, base))
    return bad


def _zeros_like(v):
    if isinstance(v, list):
        return [_zeros_like(i) for i in v]
    return 0.0


def check_seed_separation(run):
    s, ents = entries(run)
    bad = []
    if set(s["eval_seeds"]) & set(s.get("maker_seen_seeds", [])):
        bad.append("eval seeds overlap seeds the maker saw")
    ev = evaluator(s)
    seen_inputs = set()
    for f in os.listdir(os.path.join(run, "provenance")) if os.path.isdir(os.path.join(run, "provenance")) else []:
        p = load(os.path.join(run, "provenance", f))
        for x in p.get("shown", {}).get("counterexample_inputs", []):
            seen_inputs.add(canon(x))
    for sd in s["eval_seeds"]:
        for x, _ in ev.make_tasks(sd, N_EVAL):
            if canon(x) in seen_inputs:
                bad.append("eval input appears among counterexamples shown to the maker (seed %s)" % sd)
                break
    return bad


def check_provenance(run):
    s, ents = entries(run)
    bad = []
    pdir = os.path.join(run, "provenance")
    for h, e in ents.items():
        origin = e["origin"]
        if origin == "crossover":
            continue
        pid = e.get("provenance")
        pf = os.path.join(pdir, "%s.json" % pid) if pid else None
        if not pf or not os.path.exists(pf):
            bad.append("no provenance record for %s (%s)" % (h[:12], origin))
            continue
        p = load(pf)
        kind = {"llm_founder": "founder", "llm_mutation": "llm_mutation", "random_mutation": "random_mutation"}.get(origin)
        if p.get("kind") != kind:
            bad.append("provenance kind %s != %s for %s" % (p.get("kind"), kind, h[:12]))
        if not (p["ts_outbox"] <= p["ts_inbox"] <= p["ts_eval"]):
            bad.append("provenance timestamps out of order for %s" % h[:12])
        if origin in ("llm_founder", "llm_mutation"):
            if not p.get("actor") or not p.get("rationale"):
                bad.append("LLM record lacks actor/rationale for %s" % h[:12])
            if flat(p.get("response", {}).get("tensors", {})) != flat(weights_of(run, e["weights_file"])):
                bad.append("weights file differs from the LLM response recorded in provenance for %s" % h[:12])
            if origin == "llm_mutation" and not p.get("shown", {}).get("counterexample_inputs"):
                bad.append("LLM mutation %s was not shown any counterexample" % h[:12])
        if origin == "random_mutation":
            if p.get("shown", {}).get("counterexample_inputs"):
                bad.append("random mutation %s was shown counterexamples (not a control)" % h[:12])
            if p.get("rng_seed") is None:
                bad.append("random mutation %s has no rng seed" % h[:12])
    return bad


def check_near_duplicates(run, eps=EPS_DISTINCT, min_fraction=0.9):
    s, ents = entries(run)
    vecs = {}
    for h, e in ents.items():
        vecs.setdefault(e["niche"], []).append((h, flat(weights_of(run, e["weights_file"]))))
    claimed = len(ents)
    effective = 0
    for niche, items in vecs.items():
        reps = []
        for h, v in items:
            if all(len(v) != len(r) or max(abs(a - b) for a, b in zip(v, r)) > eps for r in reps):
                reps.append(v)
        effective += len(reps)
    if effective < min_fraction * claimed:
        return ["only %d effectively distinct weight vectors out of %d claimed (eps=%g): hash noise is inflating diversity" % (effective, claimed, eps)]
    return []


def check_random_control_fair(run):
    """Random and LLM edits must use the same tensors and value range (vocab declared in STANDARD.json)."""
    s, ents = entries(run)
    v = s.get("random_vocab")
    if not v:
        return ["STANDARD.json lacks random_vocab"]
    bad = []
    for h, e in ents.items():
        if e["origin"] not in ("llm_mutation", "random_mutation"):
            continue
        t = weights_of(run, e["weights_file"])
        for name, val in ((k, t[k]) for k in t):
            if name not in v["tensors"]:
                bad.append("%s edit uses tensor %s outside the shared vocabulary" % (e["origin"], name))
            xs = flat(val)
            if xs and (min(xs) < v["low"] - 1e-9 or max(xs) > v["high"] + 1e-9):
                bad.append("%s genome %s has values outside the shared range [%s,%s]" % (e["origin"], h[:12], v["low"], v["high"]))
                break
    return bad


def check_ledger_chain(run):
    s = load(os.path.join(run, "STANDARD.json"))
    p = os.path.join(run, s.get("ledger_file", "ledger.jsonl"))
    if not os.path.exists(p):
        return ["ledger missing"]
    prev = "GENESIS"
    n = 0
    with open(p, encoding="utf-8") as fh:
        raw = fh.read().splitlines()
    for line in raw:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("prev") != prev:
            return ["ledger chain broken at line %d (edited, reordered or truncated)" % (n + 1)]
        prev = hashlib.sha256(line.encode()).hexdigest()
        n += 1
    return [] if n else ["ledger empty"]


def check_budget_reconciled(run):
    s = load(os.path.join(run, "STANDARD.json"))
    pdir = os.path.join(run, "provenance")
    bad = []
    counts = {}
    if os.path.isdir(pdir):
        for f in os.listdir(pdir):
            p = load(os.path.join(pdir, f))
            if p["kind"] in ("llm_mutation", "random_mutation"):
                k = (p["arm"], p["seed"])
                counts[k] = counts.get(k, 0) + 1
    for k, n in counts.items():
        if n > s["budget_per_arm"]:
            bad.append("arm/seed %s made %d proposals, over budget %d" % (k, n, s["budget_per_arm"]))
    ledgered = 0
    p = os.path.join(run, s.get("ledger_file", "ledger.jsonl"))
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip() and json.loads(line).get("event", {}).get("type") == "proposal":
                    ledgered += 1
    total = sum(counts.values())
    if ledgered != len(s["proposals"]):
        bad.append("ledger records %d proposals but STANDARD.json lists %d" % (ledgered, len(s["proposals"])))
    if total > len(s["proposals"]):
        bad.append("provenance shows %d mutation proposals but only %d are charged" % (total, len(s["proposals"])))
    return bad


def check_llm_vs_random_recomputed(run):
    """Claimed LLM>random must hold when recomputed on the eval seeds, per seed, not just on pooled claimed numbers."""
    s, ents = entries(run)
    ev = evaluator(s)
    acc = {"llm": {}, "random": {}}
    for h, e in ents.items():
        if e["origin"] not in ("llm_mutation", "random_mutation"):
            continue
        t = weights_of(run, e["weights_file"])
        a = sum(accuracy(ev, t, sd) for sd in s["eval_seeds"]) / len(s["eval_seeds"])
        acc[e["arm"]].setdefault(e.get("seed", 0), []).append(a)
    if not acc["llm"] or not acc["random"]:
        return ["missing llm or random mutation genomes"]
    ml = sum(sum(v) / len(v) for v in acc["llm"].values()) / len(acc["llm"])
    mr = sum(sum(v) / len(v) for v in acc["random"].values()) / len(acc["random"])
    return [] if ml > mr else ["recomputed LLM-arm mean %.3f does not beat random %.3f" % (ml, mr)]


ALL = [check_hash_integrity, check_accuracy_recomputed, check_task_not_trivial, check_seed_separation, check_provenance,
       check_near_duplicates, check_random_control_fair, check_ledger_chain, check_budget_reconciled]


def check_router_present_and_matters(run, min_drop=0.05):
    """Joint genomes must carry router weights; replacing the router with a uniform one must hurt accuracy on a multi-niche eval.
    Evaluator contract: predict_routed(cells, router, input) and make_multi_tasks(seed, n) -> [(input, expected)] over all niches; uniform_router(router) -> router."""
    s, ents = entries(run)
    ev = evaluator(s)
    bad = []
    if not hasattr(ev, "predict_routed") or not hasattr(ev, "make_multi_tasks") or not hasattr(ev, "uniform_router"):
        return ["evaluator lacks predict_routed/make_multi_tasks/uniform_router: no router in the standard run"]
    best, best_acc = None, -1
    for h, e in ents.items():
        if not e.get("router_weights_file"):
            continue
        cells = weights_of(run, e["weights_file"])
        router = weights_of(run, e["router_weights_file"])
        tasks = ev.make_multi_tasks(s["eval_seeds"][0], N_EVAL)
        acc = sum(1 for x, y in tasks if ev.predict_routed(cells, router, x) == y) / max(1, len(tasks))
        if acc > best_acc:
            best, best_acc = (cells, router, tasks), acc
    if best is None:
        return ["no genome carries router_weights_file"]
    cells, router, tasks = best
    uni = ev.uniform_router(router)
    acc_u = sum(1 for x, y in tasks if ev.predict_routed(cells, uni, x) == y) / max(1, len(tasks))
    if best_acc < acc_u + min_drop:
        bad.append("router does not matter: routed %.3f vs uniform-router %.3f" % (best_acc, acc_u))
    return bad


def check_router_was_mutated_by_llm(run):
    s, ents = entries(run)
    n = sum(1 for e in ents.values() if e.get("origin") == "llm_mutation" and e.get("edit_scope") in ("router", "joint"))
    m = sum(1 for e in ents.values() if e.get("origin") == "random_mutation" and e.get("edit_scope") in ("router", "joint"))
    bad = []
    if n == 0:
        bad.append("no LLM-written router/joint mutations")
    if m == 0:
        bad.append("no random router/joint control mutations")
    return bad
