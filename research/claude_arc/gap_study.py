"""Distribution study: public ARC-AGI-2 training vs evaluation. Statistics only (no solutions inspected, no tuning)."""
import json, glob, sys, collections, numpy as np
from scipy import ndimage
from cells import bg_color
import fitted
ROOT = "C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc2/data"
def stats(split):
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/{split}/*.json")):
        d = json.load(open(f)); tr = [(np.array(e["input"]), np.array(e["output"])) for e in d["train"]]
        same = all(x.shape == y.shape for x, y in tr)
        nobj = np.mean([ndimage.label(x != bg_color(x), structure=np.ones((3, 3)))[1] for x, _ in tr])
        ncol = np.mean([len(np.unique(x)) for x, _ in tr])
        size = np.mean([x.size for x, _ in tr]); osz = np.mean([y.size for _, y in tr])
        chg = np.mean([(x != y).mean() for x, y in tr]) if same else np.nan
        # best local-rule conflict rate over feature kinds (same-shape only): fraction of demo cells whose key maps to >1 output
        best = np.nan
        if same:
            best = 1.0
            for kind in fitted.KINDS:
                tab = collections.defaultdict(collections.Counter)
                for x, y in tr:
                    ft = fitted.features(x, kind)
                    for i in range(x.shape[0]):
                        for j in range(x.shape[1]): tab[ft[i][j]][int(y[i, j])] += 1
                tot = sum(sum(c.values()) for c in tab.values()); bad = sum(sum(c.values()) - max(c.values()) for c in tab.values())
                best = min(best, bad / tot)
        rows.append(dict(same=same, nobj=nobj, ncol=ncol, size=size, osz=osz, chg=chg, conflict=best, ntrain=len(tr), ntest=len(d["test"])))
    return rows
out = {}
for split in ("training", "evaluation"):
    r = stats(split)
    g = lambda k: np.nanmean([x[k] for x in r])
    med = lambda k: np.nanmedian([x[k] for x in r])
    same = [x for x in r if x["same"]]
    out[split] = dict(n=len(r), same_shape=np.mean([x["same"] for x in r]), mean_objects=g("nobj"), mean_colors=g("ncol"),
        mean_in_cells=g("size"), mean_out_cells=g("osz"), changed_frac_same=g("chg"),
        best_local_conflict_mean=g("conflict"), best_local_conflict_median=med("conflict"),
        frac_same_with_zero_conflict=np.mean([x["conflict"] == 0 for x in same]) if same else np.nan,
        mean_train_pairs=g("ntrain"), mean_test_inputs=g("ntest"))
for k in out["training"]:
    print(f"{k:32s} training={out['training'][k]:.3f} evaluation={out['evaluation'][k]:.3f}")
json.dump(out, open("gap_study.json", "w"), indent=1)
