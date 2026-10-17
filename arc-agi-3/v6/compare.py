"""usage: compare.py split(dev|heldout) file1.jsonl[:label] ...   Per-game levels / score, totals over the split."""
import json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
split = json.load(open(os.path.join(here, "split.json")))[sys.argv[1]]
cols = []
for spec in sys.argv[2:]:
    path, _, label = spec.partition(":")
    rows = {}
    for l in open(path):
        if l.startswith("RESULT"):
            l = l[7:]
        if not l.strip().startswith("{"):
            continue
        d = json.loads(l)
        rows[d["game"]] = d
    cols.append((label or os.path.basename(path), rows))
print("game  " + " | ".join(f"{c[0][:22]:>22}" for c in cols))
tot = [[0, 0.0, 0, 0] for _ in cols]
for g in split:
    cells = []
    for i, (_, rows) in enumerate(cols):
        d = rows.get(g)
        if d is None:
            cells.append(f"{'missing':>22}")
            continue
        sc = d["score"] or 0.0
        tot[i][0] += d["levels"] or 0; tot[i][1] += sc; tot[i][2] += d["n_levels"]; tot[i][3] += 1
        cells.append(f"{d['levels']:>3}/{d['n_levels']:<2} score {sc:7.3f}     ")
    print(f"{g:5} " + " | ".join(cells))
print("TOTAL " + " | ".join(f"levels {t[0]:>3}/{t[2]} mean score {t[1] / max(t[3], 1):.3f} (n={t[3]})" for t in tot))
