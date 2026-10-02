"""Rebuild the genome-progress graphs from the run ledgers.

Usage: python scripts/genome_progress.py [--runs runs] [--outdir reports]

Writes reports/genome_progress.png (overview), reports/genome_dynamics.png (accuracy, population, species)
and reports/genome_progress.json. Read-only on runs/.

Caveats baked into the charts: accuracy = solved/tasks on each run's own selection set, so task sets differ between
runs (24 tasks in most population runs); compare within a run, not across. "Species" = the set of non-filler genes
in a genome (geometry_0 is the default filler and is ignored), so two genomes are one species when they use the
same functional genes.
"""
import argparse
import collections
import datetime
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILLER = "geometry_0"
COL = {"founder": "#4c78a8", "mutation": "#f58518", "crossover": "#54a24b"}


def opkind(op):
    o = (op or "").lower()
    if "instantiate" in o:
        return "founder"
    if "crossover" in o:
        return "crossover"
    if "mutat" in o:
        return "mutation"
    return "other"


def species_of(alleles):
    s = sorted({a.replace(".gene", "") for a in alleles if a != FILLER})
    return "+".join(s) if s else "filler-only"


def collect(runs):
    ents, pops = {}, []
    fam_order = {}
    for dp, _, fn in os.walk(runs):
        if "result.json" in fn:
            try:
                rr = json.load(open(os.path.join(dp, "result.json"), encoding="utf-8"))
                fc = rr["rows"][0]["family_task_counts"]
                fam_order[os.path.relpath(dp, runs).replace("\\", "/")] = [f for f, n in fc.items() for _ in range(n)]
            except Exception:
                pass
    for dp, _, fn in os.walk(runs):
        if "ledger.jsonl" not in fn:
            continue
        path = os.path.join(dp, "ledger.jsonl")
        t = os.path.getmtime(path)
        run = os.path.relpath(dp, runs).replace("\\", "/")
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    r = json.loads(line)["payload"]
                except Exception:
                    continue
                if not isinstance(r, dict):
                    continue
                m = r.get("metrics")
                if isinstance(m, dict) and "generation" in m and "living" in m:
                    pops.append(dict(run=run, arm=r.get("arm"), gen=m["generation"], living=m["living"],
                                     archive=m.get("archive_genotypes"), t=t))
                g = r.get("genome")
                if not isinstance(g, dict) or "alleles" not in g:
                    continue
                h = r.get("artifact_hash") or r.get("canonical_genotype")
                fit = r.get("fitness") or {}
                acc = None
                if fit.get("tasks") and fit.get("solved") is not None and not fit.get("failed"):
                    acc = fit["solved"] / fit["tasks"]
                ents.setdefault(h, dict(kind=opkind(g.get("operator")), parents=g.get("parent_hashes") or [],
                                        alleles=g["alleles"], geno=r.get("canonical_genotype"), t=t, run=run,
                                        arm=r.get("arm"), acc=acc, species=species_of(g["alleles"]),
                                        solves=fit.get("solves") if not fit.get("failed") else None))
    for e in ents.values():
        e["families"] = fam_order.get(e["run"])
    return ents, pops


def summarize(ents, pops):
    ops = collections.Counter(e["kind"] for e in ents.values())
    genes = collections.Counter(a for e in ents.values() for a in e["alleles"])
    cross = [e for e in ents.values() if e["kind"] == "crossover"]
    two = [e for e in cross if len(set(e["parents"])) >= 2]
    mixed = 0
    for e in two:
        ps = [ents[p]["alleles"] for p in e["parents"] if p in ents]
        if len(ps) >= 2 and e["alleles"] not in ps:
            mixed += 1
    geno = collections.defaultdict(set)
    for e in ents.values():
        geno[e["kind"]].add(e["geno"])
    species = collections.Counter(e["species"] for e in ents.values())
    return dict(total=len(ents), ops=dict(ops), genes=dict(genes), crossover=len(cross), two_parents=len(two),
                new_mix=mixed, distinct_genotypes={k: len(v) for k, v in geno.items()}, species=dict(species))


def overview(s, ents, out):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle("CellForge genomes (from run ledgers, %s)" % datetime.date.today(), fontsize=14, weight="bold")
    g = sorted(s["genes"].items(), key=lambda x: x[1])
    ax[0][0].barh([k.replace(".gene", "") for k, _ in g], [v for _, v in g], color="#4c78a8")
    ax[0][0].set_title("Hand-written genes: %d distinct (uses)" % len(g))
    times = sorted((e["t"], e["kind"]) for e in ents.values())
    for k in COL:
        n, ys = 0, []
        for _, kk in times:
            n += kk == k
            ys.append(n)
        ax[0][1].plot([datetime.datetime.fromtimestamp(a) for a, _ in times], ys, label=k, color=COL[k], lw=2)
    ax[0][1].set_title("Genomes created over time")
    ax[0][1].legend()
    ax[0][1].tick_params(axis="x", rotation=30)
    vals = [s["crossover"] - s["two_parents"], s["two_parents"] - s["new_mix"], s["new_mix"]]
    ax[1][0].bar(["no 2nd parent", "copy of a parent", "new mix"], vals, color=["#bbbbbb", "#e45756", "#54a24b"])
    for i, v in enumerate(vals):
        ax[1][0].text(i, v + 1, str(v), ha="center")
    ax[1][0].set_title("Crossover children: %d" % s["crossover"])
    labs = list(COL)
    tot = [s["ops"].get(k, 0) for k in labs]
    dis = [s["distinct_genotypes"].get(k, 0) for k in labs]
    x = range(len(labs))
    ax[1][1].bar([i - 0.2 for i in x], tot, 0.4, label="built", color="#9ecae9")
    ax[1][1].bar([i + 0.2 for i in x], dis, 0.4, label="distinct genotypes", color="#3182bd")
    ax[1][1].set_xticks(list(x))
    ax[1][1].set_xticklabels(labs)
    ax[1][1].set_title("Built vs genuinely different")
    ax[1][1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=110)
    plt.close(fig)


def dynamics(ents, pops, out):
    fig, ax = plt.subplots(2, 3, figsize=(19, 9.5))
    fig.suptitle("Accuracy, population and species (accuracy = solved/tasks on each run's own set)", fontsize=13, weight="bold")
    scored = sorted((e for e in ents.values() if e["acc"] is not None), key=lambda e: e["t"])

    # accuracy over time: per-genome scatter + running best and running mean (window 30)
    a0 = ax[0][0]
    for k in COL:
        pts = [(datetime.datetime.fromtimestamp(e["t"]), e["acc"]) for e in scored if e["kind"] == k]
        if pts:
            a0.scatter(*zip(*pts), s=8, alpha=0.35, color=COL[k], label=k)
    best, bs = 0, []
    for e in scored:
        best = max(best, e["acc"])
        bs.append(best)
    xs = [datetime.datetime.fromtimestamp(e["t"]) for e in scored]
    a0.plot(xs, bs, color="black", lw=2, label="best so far")
    w = 30
    accs = [e["acc"] for e in scored]
    ma = [sum(accs[max(0, i - w + 1):i + 1]) / len(accs[max(0, i - w + 1):i + 1]) for i in range(len(accs))]
    a0.plot(xs, ma, color="#e45756", lw=2, label="running mean (30)")
    a0.set_title("Accuracy over time (ledger time)")
    a0.set_ylim(0, 1.02)
    a0.legend(fontsize=8)
    a0.tick_params(axis="x", rotation=30)

    # accuracy by kind
    data = [[e["acc"] for e in scored if e["kind"] == k] for k in COL]
    ax[0][1].boxplot(data, labels=["%s\n(n=%d)" % (k, len(d)) for k, d in zip(COL, data)], showmeans=True)
    ax[0][1].set_title("Accuracy distribution by origin")
    ax[0][1].set_ylim(0, 1.02)

    # gene -> mean accuracy of genomes containing it
    gacc = collections.defaultdict(list)
    for e in scored:
        for a in set(e["alleles"]):
            gacc[a.replace(".gene", "")].append(e["acc"])
    gl = sorted(((sum(v) / len(v), k, len(v)) for k, v in gacc.items()))
    ax[0][2].barh(["%s (n=%d)" % (k, n) for _, k, n in gl], [m for m, _, _ in gl], color="#72b7b2")
    ax[0][2].set_title("Mean accuracy of genomes containing each gene\n(association, not causation)")
    ax[0][2].set_xlim(0, 1)

    # population per generation per arm (mean over runs)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    byarch = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pops:
        by[p["arm"]][p["gen"]].append(p["living"])
        if p["archive"] is not None:
            byarch[p["arm"]][p["gen"]].append(p["archive"])
    colors = {"crossover": "#54a24b", "mutation_only": "#f58518", "random_search": "#888888"}
    for arm, d in sorted(by.items()):
        gs = sorted(d)
        ax[1][0].plot(gs, [sum(d[g]) / len(d[g]) for g in gs], lw=2, label="%s living" % arm, color=colors.get(arm))
        da = byarch[arm]
        ax[1][0].plot(sorted(da), [sum(da[g]) / len(da[g]) for g in sorted(da)], lw=1.5, ls="--",
                      color=colors.get(arm), label="%s archive" % arm)
    ax[1][0].set_title("Population per generation (%d series)" % len({(p['run'], p['arm']) for p in pops}))
    ax[1][0].set_xlabel("generation")
    ax[1][0].legend(fontsize=7)

    # species: stacked counts by origin for top species
    sp = collections.Counter(e["species"] for e in ents.values())
    top = [k for k, _ in sp.most_common(10)]
    bottom = [0] * len(top)
    for k in COL:
        v = [sum(1 for e in ents.values() if e["species"] == s_ and e["kind"] == k) for s_ in top]
        ax[1][1].barh(range(len(top)), v, left=bottom, color=COL[k], label=k)
        bottom = [a + b for a, b in zip(bottom, v)]
    ax[1][1].set_yticks(range(len(top)))
    ax[1][1].set_yticklabels([t[:38] for t in top], fontsize=7)
    ax[1][1].invert_yaxis()
    ax[1][1].set_title("Top species by genome count (%d species total)" % len(sp))
    ax[1][1].legend(fontsize=8)

    # species richness over time
    seen, xs2, ys2 = set(), [], []
    for e in sorted(ents.values(), key=lambda e: e["t"]):
        seen.add(e["species"])
        xs2.append(datetime.datetime.fromtimestamp(e["t"]))
        ys2.append(len(seen))
    ax[1][2].plot(xs2, ys2, color="#b279a2", lw=2)
    ax[1][2].set_title("Cumulative distinct species discovered")
    ax[1][2].tick_params(axis="x", rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=105)
    plt.close(fig)


def families(ents, out):
    tot = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for e in ents.values():
        fams, sol = e.get("families"), e.get("solves")
        if not fams or not sol or len(fams) != len(sol):
            continue
        for f, ok in zip(fams, sol):
            tot[f][e["kind"]][0] += bool(ok)
            tot[f][e["kind"]][1] += 1
    if not tot:
        return False
    names = sorted(tot, key=lambda f: sum(v[0] for v in tot[f].values()) / max(1, sum(v[1] for v in tot[f].values())))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, k in enumerate(COL):
        ax.barh([j + (i - 1) * 0.27 for j in range(len(names))],
                [tot[f][k][0] / tot[f][k][1] if tot[f][k][1] else 0 for f in names], 0.27, color=COL[k], label=k)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1)
    ax.set_title("Solve rate per task family, by genome origin\n(expanded-bank runs, 12 families x 2 tasks; family order taken from the run's result.json)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)
    return True


def lineage(ents, out):
    adj = collections.defaultdict(set)
    for h, e in ents.items():
        for p in e["parents"]:
            if p in ents:
                adj[h].add(p)
                adj[p].add(h)
    seen, best = set(), []
    for h in ents:
        if h in seen or h not in adj:
            continue
        comp, stack = [], [h]
        seen.add(h)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(comp) > len(best):
            best = comp
    if not best:
        return False
    nodes = set(best)
    depth = {}
    def d(h):
        if h in depth:
            return depth[h]
        depth[h] = 0
        ps = [p for p in ents[h]["parents"] if p in nodes]
        depth[h] = 1 + max((d(p) for p in ps), default=-1)
        return depth[h]
    for h in nodes:
        d(h)
    cols = collections.defaultdict(list)
    for h in sorted(nodes, key=lambda h: (ents[h]["t"], h)):
        cols[depth[h]].append(h)
    pos = {h: (dx, j - len(v) / 2) for dx, v in cols.items() for j, h in enumerate(v)}
    fig, ax = plt.subplots(figsize=(14, 8))
    for h in nodes:
        for p in ents[h]["parents"]:
            if p in nodes:
                ax.plot([pos[p][0], pos[h][0]], [pos[p][1], pos[h][1]], color="#bbbbbb", lw=0.6, zorder=1)
    for k in COL:
        hs = [h for h in nodes if ents[h]["kind"] == k]
        ax.scatter([pos[h][0] for h in hs], [pos[h][1] for h in hs],
                   s=[30 + 120 * (ents[h]["acc"] or 0) for h in hs], color=COL[k], label="%s (%d)" % (k, len(hs)), zorder=2,
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel("generation depth (longest ancestry path)")
    ax.set_yticks([])
    ax.set_title("Largest lineage family: %d genomes, %d generations (dot size = accuracy)" % (len(nodes), max(depth.values()) + 1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--outdir", default="reports")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ents, pops = collect(a.runs)
    s = summarize(ents, pops)
    overview(s, ents, os.path.join(a.outdir, "genome_progress.png"))
    dynamics(ents, pops, os.path.join(a.outdir, "genome_dynamics.png"))
    print("families chart:", families(ents, os.path.join(a.outdir, "genome_families.png")))
    print("lineage chart:", lineage(ents, os.path.join(a.outdir, "genome_lineage.png")))
    with open(os.path.join(a.outdir, "genome_progress.json"), "w") as f:
        json.dump(s, f, indent=1)
    print({k: v for k, v in s.items() if k not in ("genes", "species")}, "species", len(s["species"]),
          "pop series", len(pops))
