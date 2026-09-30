"""Score baseline vs baseline+dsl_port from evaluate.py outputs. Every number is tagged with its split.
usage: python analyze.py cache/res_base.jsonl cache/res_dsl.jsonl
Attempt policy (two ranked attempts, exact match on all test inputs):
  baseline alone : first two distinct baseline predictions (same as solve.py).
  dsl alone      : dsl prediction groups ordered by (min cost, -support); 'strict' keeps the task only if ALL consistent
                   dsl programs agree on the test predictions (ambiguity rejected); 'top2' takes the two best groups.
  combined       : attempt 1 = best baseline group, attempt 2 = best dsl group, then fill from the remainder.
"""
import sys, json, collections


def load(fp):
    return {json.loads(l)["task"]: json.loads(l) for l in open(fp) if l.strip()}


def groups(rows):
    g = collections.OrderedDict()
    for r in rows:
        x = g.setdefault(r["ph"], {"cost": r.get("cost", 0), "ok": r["ok"], "n": 0, "fam": r.get("fam", "base"), "desc": r["desc"]})
        x["n"] += 1
    return sorted(g.values(), key=lambda x: (x["cost"], -x["n"]))


def attempts_base(br):
    gs = groups(br["sols"]) if br else []
    # baseline keeps enumeration order (not cost order)
    seen, out = set(), []
    for r in (br["sols"] if br else []):
        if r["ph"] not in seen:
            seen.add(r["ph"]); out.append({"ok": r["ok"], "fam": "base", "desc": r["desc"]})
    return out


def attempts_dsl(dr, mode, gen_filter):
    rows = dr["cands"] if dr else []
    if gen_filter and dr and dr["n_gen"] > 0:
        if gen_filter == "hard":            # must fit every one of k generated examples (abstentions allowed)
            rows = [r for r in rows if r["gwrong"] == 0]
        elif gen_filter == "tol":           # tolerate up to 10% wrong generated examples
            rows = [r for r in rows if r["gwrong"] <= 0.1 * max(1, dr["n_gen"])]
        elif gen_filter == "rank":          # no rejection; generator agreement only re-ranks candidates
            rows = sorted(rows, key=lambda r: (r["gwrong"], r["cost"]))
            for r in rows:
                r["cost"] = r["gwrong"] * 100 + r["cost"]
    gs = groups(rows)
    if mode == "strict":
        return gs if len(gs) == 1 else []
    return gs[:2]


def combine(b, d):
    out, seen = [], set()
    for x in ([b[0]] if b else []) + ([d[0]] if d else []) + b[1:] + d[1:]:
        k = id(x)
        if k in seen:
            continue
        seen.add(k); out.append(x)
        if len(out) == 2:
            break
    return out


def score(tasks, B, Dd, mode, gen_filter, label):
    n = len(tasks)
    res = collections.defaultdict(int)
    gain = collections.Counter()
    for t in tasks:
        b = attempts_base(B.get(t))
        d = attempts_dsl(Dd.get(t), mode, gen_filter)
        bs = any(x["ok"] for x in b[:2])
        ds = any(x["ok"] for x in d[:2])
        cb = combine(b, d)
        cs = any(x["ok"] for x in cb)
        res["base_solved"] += bs
        res["base_consistent"] += bool(b)
        res["base_fp"] += bool(b) and not bs
        res["dsl_solved"] += ds
        res["dsl_attempted"] += bool(d)
        res["dsl_fp"] += bool(d) and not ds
        res["comb_solved"] += cs
        res["comb_attempted"] += bool(cb)
        res["comb_fp"] += bool(cb) and not cs
        if cs and not bs:
            for x in cb:
                if x["ok"]:
                    gain[x["fam"]] += 1
                    break
    print(f"[{label}] n={n} mode={mode} genfilter={gen_filter}: baseline {res['base_solved']} (attempted {res['base_consistent']}, FP {res['base_fp']}) | "
          f"dsl-alone {res['dsl_solved']} (attempted {res['dsl_attempted']}, FP {res['dsl_fp']}) | "
          f"combined {res['comb_solved']} (attempted {res['comb_attempted']}, FP {res['comb_fp']}) | gains by family {dict(gain)}")
    return res, gain


def main():
    B, Dd = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(B) & set(Dd))
    for split in ("dev", "val"):
        tasks = [t for t in common if (B[t]["split"] == split)]
        if not tasks:
            continue
        for mode in ("strict", "top2"):
            score(tasks, B, Dd, mode, False, split.upper())
        print(f"[{split.upper()}] sec/task: baseline {sum(B[t]['sec'] for t in tasks)/len(tasks):.2f}  dsl_port {sum(Dd[t]['sec'] for t in tasks)/len(tasks):.2f}")
    # generator filtering (DEV tasks having >=1 generated example)
    dev_gen = [t for t in common if Dd[t]["split"] == "dev" and Dd[t]["n_gen"] > 0]
    print(f"[DEV, generator-covered subset] {len(dev_gen)} tasks")
    for mode in ("strict", "top2"):
        for gf in (False, "hard", "tol", "rank"):
            score(dev_gen, B, Dd, mode, gf, "DEV-gen")
    pc = collections.Counter()
    for t in dev_gen:
        for r in Dd[t]["cands"]:
            k = "ok" if r["ok"] else "wrong"
            pc[(k, "before")] += 1
            if r["gwrong"] == 0:
                pc[(k, "after_hard")] += 1
            if r["gwrong"] <= 2:
                pc[(k, "after_tol2of20")] += 1
    print("[DEV-gen] candidate programs (demo-consistent, test-covered):", dict(pc))
    by_fam = collections.Counter()
    for t in common:
        seen = set()
        for r in Dd[t]["cands"]:
            if r["ok"] and r["fam"] not in seen:
                seen.add(r["fam"]); by_fam[(Dd[t]["split"], r["fam"])] += 1
    print("tasks with >=1 CORRECT candidate per family (any rank):", dict(by_fam))


main()
