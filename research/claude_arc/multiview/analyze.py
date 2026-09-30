"""Derive all arms from the cached candidate pools and report DEV / VALIDATION tables.
Usage: python analyze.py results/dev.jsonl results/val.jsonl results/summary.json"""
import sys, json, random
from collections import Counter

K_CHAIN = [0, 4, 10, 25, 57]      # disinhibition palette g=0..4: first cells kept for depth-2 composition (0 = depth 1 only)
M_FIT = [1, 2, 4, 8, 99]          # fitted candidates kept per fitter family (existing and multiview), ranked simplest first
PAIRS_FROM = 2                    # multiview rules using two features (cross-view pairs) only allowed from g>=2
EXIST_ORDER = ["chain", "fit_local", "fit_role", "fit_shape", "obj"]


def load(p):
    return [json.loads(l) for l in open(p)]


def kept_existing(c, g):
    if g is None:
        return True
    if c["fam"] == "chain":
        return c["depth"] == 1 or c["r1"] < K_CHAIN[g]
    return c["idx"] < M_FIT[g]


def kept_mv(c, g):
    if g is None:
        return False
    if c["nfeat"] > 1 and g < PAIRS_FROM:
        return False
    return c["idx"] < M_FIT[g]


def evaluate(rec, gE, gM):
    """gE: None = full existing stack (baseline) or g index; gM: None = no multiview cells, else g index."""
    ex = [c for c in rec["cands"] if c["fam"] in EXIST_ORDER and kept_existing(c, gE)]
    chains = [c for c in ex if c["fam"] == "chain"][:4]
    rest = sorted([c for c in ex if c["fam"] != "chain"], key=lambda c: (EXIST_ORDER.index(c["fam"]), c["idx"]))
    mv = [c for c in rec["cands"] if c["fam"].startswith("mv_") and kept_mv(c, gM)]
    mv.sort(key=lambda c: (c["fam"], c["idx"]))
    seq = chains + rest + mv
    att, seen = [], set()
    for c in seq:
        if c["sig"] not in seen:
            seen.add(c["sig"]); att.append(c)
        if len(att) == 2:
            break
    solved = any(c["ok"] for c in att)
    sec = 0.0
    blocks = rec["t_chain_blocks"]
    sec += rec["t_chain_d1"]
    k = 99 if gE is None else K_CHAIN[gE]
    sec += sum(v for r, v in blocks.items() if int(r) < k)
    sec += rec["t_fit"]
    if gM is not None:
        sec += rec["t_mv"]
    return dict(solved=solved, cons=len(seq) > 0, fp=(len(seq) > 0 and not solved), att=att, sec=sec)


def arm_stats(recs, cfg):
    solved = fp = cons = 0; sec = 0.0
    for r in recs:
        e = cfg(r)
        solved += e["solved"]; fp += e["fp"]; cons += e["cons"]; sec += e["sec"]
    n = len(recs)
    return dict(solved=solved, fp=fp, consistent=cons, sec_per_task=round(sec / n, 2), n=n)


def arms(recs):
    out = {}
    out["A baseline (existing, full)"] = arm_stats(recs, lambda r: evaluate(r, None, None))
    for g in range(5):
        out[f"B mv@g{g} + existing full"] = arm_stats(recs, lambda r, g=g: evaluate(r, None, g))
    for g in range(5):
        out[f"D existing@g{g} only (no mv)"] = arm_stats(recs, lambda r, g=g: evaluate(r, g, None))
    for g in range(5):
        out[f"E existing@g{g} + mv@g{g}"] = arm_stats(recs, lambda r, g=g: evaluate(r, g, g))
    rs = []
    for seed in range(5):
        rng = random.Random(seed)
        gs = {r["task"]: rng.randrange(5) for r in recs}
        rs.append(arm_stats(recs, lambda r: evaluate(r, gs[r["task"]], gs[r["task"]])))
    out["F random-g control (mean of 5 seeds; existing+mv)"] = dict(
        solved=round(sum(x["solved"] for x in rs) / 5, 1), fp=round(sum(x["fp"] for x in rs) / 5, 1),
        consistent=round(sum(x["consistent"] for x in rs) / 5, 1), sec_per_task=rs[0]["sec_per_task"], n=len(recs),
        solved_min=min(x["solved"] for x in rs), solved_max=max(x["solved"] for x in rs))
    return out


def contributions(recs):
    """Which view families solve tasks that the baseline does not (mv at loosest g; any kept correct candidate)."""
    base = {r["task"]: evaluate(r, None, None)["solved"] for r in recs}
    rescued, by_fam, fam_any = [], Counter(), Counter()
    for r in recs:
        okmv = [c for c in r["cands"] if c["fam"].startswith("mv_") and c["ok"]]
        for c in okmv:
            fam_any[c["fam"] + ("/" + "+".join(c["fams"]) if c["fam"] == "mv_px" else "")] += 0
        e = evaluate(r, None, 4)
        if e["solved"] and not base[r["task"]]:
            a = [c for c in e["att"] if c["ok"]][0]
            key = a["fam"] + ("/" + "+".join(a["fams"]) + ("|src:" + "+".join(a["sfams"]) if a["sfams"] else "") if a["fam"] == "mv_px" else "")
            by_fam[key] += 1; rescued.append((r["task"], a["kind"]))
    tasks_with_ok = Counter()
    for r in recs:
        seen = set()
        for c in r["cands"]:
            if c["fam"].startswith("mv_") and c["ok"]:
                k = c["fam"] + ("/" + "+".join(c["fams"]) if c["fam"] == "mv_px" else "")
                if k not in seen:
                    seen.add(k); tasks_with_ok[k] += 1
    harmed = [r["task"] for r in recs if base[r["task"]] and not evaluate(r, None, 4)["solved"]]
    return dict(rescued_by_family=dict(by_fam), rescued=rescued, tasks_with_a_correct_mv_candidate=dict(tasks_with_ok), harmed_by_mv_full=harmed)


def fmt(title, nrec, a):
    lines = [f"### {title} (n={nrec})", "| arm | solved | demo-consistent-but-wrong | tasks w/ any candidate | s/task |", "|---|---|---|---|---|"]
    for k, v in a.items():
        lines.append(f"| {k} | {v['solved']} | {v['fp']} | {v['consistent']} | {v['sec_per_task']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    dev, val = load(sys.argv[1]), load(sys.argv[2])
    assert all(r["bucket"] != 0 for r in dev) and all(r["bucket"] == 0 for r in val)
    ad, av = arms(dev), arms(val)
    # g selection on DEV only: best solved, then fewest false positives, among arms that use g
    cand = {k: v for k, v in ad.items() if k[0] in "BDE"}
    best = max(cand, key=lambda k: (cand[k]["solved"], -cand[k]["fp"]))
    summary = dict(dev_buckets="1-4", val_bucket="0", dev=ad, val=av, dev_selected_arm=best, val_of_selected=av[best],
                   contrib_dev=contributions(dev), contrib_val=contributions(val))
    json.dump(summary, open(sys.argv[3], "w"), indent=1)
    print(fmt("DEV (crc32 buckets 1-4)", len(dev), ad)); print()
    print(fmt("VALIDATION (bucket 0; never used for selection)", len(val), av)); print()
    print("DEV-selected arm:", best, "-> VALIDATION:", av[best])
    for nm, k in (("DEV", "contrib_dev"), ("VAL", "contrib_val")):
        c = summary[k]
        print(nm, "rescued by family:", c["rescued_by_family"], "| harmed by mv (full g):", c["harmed_by_mv_full"])
        print(nm, "tasks with a correct mv candidate:", c["tasks_with_a_correct_mv_candidate"])
