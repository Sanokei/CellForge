"""Summarise cache/coverage.jsonl into coverage_table.md (per generator family x cell family solve rate)."""
import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "cache/coverage.jsonl")]
F = ["SEL", "CROP", "REC", "LINE", "MACRO"]
rows, agg, anyfam, top1 = [], collections.Counter(), 0, 0
ep_total = ep_any = ep_top1 = ep_cand = ep_wrong_only = 0
for r in sorted(R, key=lambda r: r["task"]):
    E = r["episodes"]
    if not E:
        continue
    rate = {f: sum(f in e["solved_by"] for e in E) / len(E) for f in F}
    anys = sum(bool(e["solved_by"]) for e in E) / len(E)
    rows.append((r["task"], rate, anys))
    for f in F:
        agg[f] += rate[f] > 0
    anyfam += anys > 0
    for e in E:
        ep_total += 1; ep_any += bool(e["solved_by"]); ep_top1 += e["top1_ok"]; ep_cand += e["n"] > 0
        ep_wrong_only += (e["n"] > 0 and not e["solved_by"])
out = ["# dsl_port coverage on re-arc generated examples (DEVELOPMENT families only; offline curriculum, not test-time)",
       "", f"generator families: {len(rows)}; episodes: {ep_total} (3 generated demos + 1 generated test each)",
       f"episodes with >=1 candidate: {ep_cand}; with a correct candidate: {ep_any}; top-1 (lowest cost) correct: {ep_top1}; "
       f"candidates exist but none correct: {ep_wrong_only}", "",
       "families solved in >=1 episode: " + ", ".join(f"{f} {agg[f]}" for f in F) + f"; any {anyfam}", "",
       "| task | " + " | ".join(F) + " | any |", "|---|" + "---|" * (len(F) + 1)]
for t, rate, a in rows:
    if a > 0:
        out.append(f"| {t} | " + " | ".join(f"{rate[f]:.2f}" for f in F) + f" | {a:.2f} |")
open("coverage_table.md", "w").write("\n".join(out) + "\n")
print("\n".join(out[:8]))
