"""Summarise RESULT jsonl files: python table.py random.jsonl explorer_v1.jsonl ..."""
import json, sys
rs = {}
for f in sys.argv[1:]:
    rs[f] = {}
    for l in open(f):
        l = l.strip()
        if l.startswith("RESULT "): l = l[7:]
        if l: d = json.loads(l); rs[f][d["game"]] = d
games = sorted({g for r in rs.values() for g in r})
print("game  lv | " + " | ".join(f"{f[:14]:>14}" for f in rs))
tot = {f: [0, 0.0, 0.0, 0] for f in rs}
for g in games:
    row = []
    for f in rs:
        d = rs[f].get(g)
        if d:
            row.append(f"{d['levels']}/{d['n_levels']} {d['score']:6.2f} {d['secs']:5.0f}s {d['peak_mb']}M")
            t = tot[f]; t[0] += d["levels"]; t[1] += d["score"]; t[2] += d["secs"]; t[3] = max(t[3], d["peak_mb"])
        else: row.append("-")
    print(f"{g:5} | " + " | ".join(f"{x:>24}" for x in row))
for f, t in tot.items():
    n = len(rs[f]); print(f"{f}: levels {t[0]}  mean score {t[1]/max(n,1):.3f}  total secs {t[2]:.0f}  peak MB {t[3]}")
