import json,sys,math,collections
import numpy as np
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/2/n)/d; h=z*math.sqrt(p*(1-p)/n+z*z/4/n/n)/d; return (max(0,c-h),min(1,c+h))
R=[json.loads(l) for l in open(sys.argv[1])]
def arm(r): return r["arm"]+("" if r["arm"] in "AD" else f"(lam={r['lam']})")
def fin(r): return r["snaps"][-1] if r["snaps"] else None
def dg(r):
    f=r["fitsnap"]; 
    if not f or r["efit"]>30000: return False
    z=fin(r); return f["extra20"]<0.5 and z["demo_exact"] and z["extra20"]>=0.5 and z["extra20"]>=f["extra20"]+0.3
def fm(k,n): lo,hi=wilson(k,n); return f"{k}/{n} [{lo:.2f},{hi:.2f}]"
out=[]
for scope,filt in (("LOCAL tasks",lambda r:r["ctrl"]==0),("CONTROL tasks",lambda r:r["ctrl"]==1)):
    out.append(f"\n### {scope}\n")
    out.append("| arm | runs | demo fit ever | fit at final | median e_fit | DG runs | final extra20 exact mean | final cellacc | extra20 at e_fit (A/B/D: fitted runs) | nnz final | L1 final | extra20 at e_fit, C-equivalent |")
    out.append("|"+"---|"*12)
    arms=collections.OrderedDict()
    for r in R:
        if filt(r): arms.setdefault(arm(r),[]).append(r)
    for a,rs in sorted(arms.items()):
        n=len(rs); fit=[r for r in rs if r["efit"]]; fz=[r for r in rs if fin(r)["demo_exact"]]
        me=np.median([r["efit"] for r in fit]) if fit else float("nan")
        ex=np.mean([fin(r)["extra20"] for r in rs]); ca=np.mean([fin(r)["cellacc"] for r in rs])
        atfit=np.mean([r["fitsnap"]["extra20"] for r in fit]) if fit else float("nan")
        out.append(f"| {a} | {n} | {fm(len(fit),n)} | {fm(len(fz),n)} | {me:.0f} | {fm(sum(dg(r) for r in rs),n)} | {ex:.3f} | {ca:.3f} | {atfit:.3f} | {np.mean([fin(r)['nnz'] for r in rs]):.0f} | {np.mean([fin(r)['L1'] for r in rs]):.1f} | {atfit:.3f} |")
    out.append("\nHeld-out extra20 exact (mean over runs) vs eval count, by arm:\n")
    out.append("| arm | 1e3 | 3e3 | 1e4 | 3e4 | 1e5 |"); out.append("|---|---|---|---|---|---|")
    for a,rs in sorted(arms.items()):
        out.append(f"| {a} | "+" | ".join(f"{np.mean([s['extra20'] for r in rs for s in r['snaps'] if s['check']==c]):.3f}" for c in (1000,3000,10000,30000,100000))+" |")
    out.append("\nMean extra20 cell accuracy vs eval count:\n")
    out.append("| arm | 1e3 | 3e3 | 1e4 | 3e4 | 1e5 |"); out.append("|---|---|---|---|---|---|")
    for a,rs in sorted(arms.items()):
        out.append(f"| {a} | "+" | ".join(f"{np.mean([s['cellacc'] for r in rs for s in r['snaps'] if s['check']==c]):.3f}" for c in (1000,3000,10000,30000,100000))+" |")
out.append("\n### Per task (local), final extra20 exact / demo fit ever, per arm (pooled over seeds)\n")
tasks=sorted({r["task"] for r in R}); arms=sorted({arm(r) for r in R})
out.append("| task | "+" | ".join(arms)+" |"); out.append("|"+"---|"*(len(arms)+1))
for t in tasks:
    row=[]
    for a in arms:
        rs=[r for r in R if r["task"]==t and arm(r)==a]
        row.append(f"{np.mean([fin(r)['extra20'] for r in rs]):.2f} / {sum(1 for r in rs if r['efit'])}of{len(rs)}")
    out.append(f"| {t} | "+" | ".join(row)+" |")
out.append("\n### Runs meeting DG criterion\n")
for r in R:
    if dg(r): out.append(f"- {r['task']} seed{r['seed']} {arm(r)}: e_fit={r['efit']} extra20 at fit {r['fitsnap']['extra20']:.2f} -> final {fin(r)['extra20']:.2f}")
out.append("\n### Fitted runs where held-out rose after fit by >=0.15 (weaker signal), extra20 fit->final\n")
for r in R:
    if r["fitsnap"] and fin(r)["extra20"]-r["fitsnap"]["extra20"]>=0.15 and not dg(r): out.append(f"- {r['task']} s{r['seed']} {arm(r)}: e_fit={r['efit']} {r['fitsnap']['extra20']:.2f}->{fin(r)['extra20']:.2f}, demo_exact_final={fin(r)['demo_exact']}")
print("\n".join(out))
