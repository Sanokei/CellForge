"""Compare RESULT jsonl files on their common games, with actions per cleared level.
usage: python table2.py a.jsonl b.jsonl ...   (per-file: levels, mean score, mean actions/cleared level, secs, peak MB)"""
import json, sys
rs = {}
for f in sys.argv[1:]:
    rs[f] = {}
    for l in open(f):
        l = l.strip()
        if l.startswith("RESULT "): l = l[7:]
        if l:
            d = json.loads(l); rs[f][d["game"]] = d
common = sorted(set.intersection(*[set(r) for r in rs.values()]))
print(f"common games: {len(common)}")
for f, r in rs.items():
    lv = sum(r[g]["levels"] for g in common)
    sc = sum(r[g]["score"] for g in common) / len(common)
    acts = [a for g in common for a in (r[g]["level_actions"] or [])[:r[g]["levels"]]]
    apl = sum(acts) / len(acts) if acts else float("nan")
    med = sorted(acts)[len(acts) // 2] if acts else float("nan")
    print(f"{f:28} levels {lv:3}  mean score {sc:.3f}  actions/cleared level mean {apl:6.0f} median {med:5}  secs {sum(r[g]['secs'] for g in common):5.0f}  peakMB {max(r[g]['peak_mb'] for g in common)}")
