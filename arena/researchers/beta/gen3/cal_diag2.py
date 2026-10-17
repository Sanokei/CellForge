#!/usr/bin/env python3
"""Diag2: (1) LDA/diag-LDA on per-decision mean residuals (trajectory means);
(2) behavioral looping stats (cell repetition runs) dev vs tx."""
import os, sys
import torch
import torch.nn.functional as F
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
WD = "/srv/cellforge-arena/researchers/beta/gen3"
data = torch.load(os.path.join(WD, "cal_data.pt"), map_location="cpu",
                  weights_only=False)

def ln(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5)

# per-decision mean residual (ln1 space), label
M, Y, L = [], [], []
for lab, dd in data.items():
    tx = lab.startswith("TX")
    for d in dd["decs"]:
        M.append(ln(d["X"]).mean(0))
        Y.append(tx); L.append(lab)
M = torch.stack(M); Y = torch.tensor(Y).bool()
Md, Mt = M[~Y], M[Y]
print(f"decisions dev={Md.shape[0]} tx={Mt.shape[0]}", flush=True)
dmu = Mt.mean(0) - Md.mean(0)

def sep_report(w, name):
    w = w / (w.norm() + 1e-12)
    pd, pt = Md @ w, Mt @ w
    import math
    def qq(v, q_):
        return float(torch.quantile(v.float(), q_))
    print(f"{name}: dev q10/med/q90 = {qq(pd,0.1):.3f}/{qq(pd,0.5):.3f}/{qq(pd,0.9):.3f} | "
          f"tx q10/med/q90 = {qq(pt,0.1):.3f}/{qq(pt,0.5):.3f}/{qq(pt,0.9):.3f}", flush=True)
    # threshold at dev 99%: tx recall
    tau = qq(pd, 0.99)
    rec = float((pt > tau).float().mean())
    # threshold at tx 1%: dev false-fire
    tau2 = qq(pt, 0.01)
    ff = float((pd > tau2).float().mean())
    print(f"   tau=dev99% -> tx recall {rec:.3f}; tau=tx1% -> dev fire {ff:.3f}", flush=True)

# (a) raw mean-diff direction
sep_report(dmu, "mean-diff")
# (b) diagonal LDA (standardized)
var = (Md.var(0) + Mt.var(0)) / 2 + 1e-6
sep_report(dmu / var.sqrt(), "diag-LDA")
# (c) full LDA with shrinkage
Cd = Md - Md.mean(0, keepdim=True)
Ct = Mt - Mt.mean(0, keepdim=True)
Sw = (Cd.t() @ Cd) / Md.shape[0] + (Ct.t() @ Ct) / Mt.shape[0]
lam = 1.0
w_lda = torch.linalg.solve(Sw + lam * torch.eye(128), dmu)
sep_report(w_lda, "full-LDA(lam=1)")
lam = 10.0
w_lda = torch.linalg.solve(Sw + lam * torch.eye(128), dmu)
sep_report(w_lda, "full-LDA(lam=10)")

# ---- behavioral looping: runs of identical emitted cells ----
for lab in sorted(data.keys()):
    cells = data[lab]["cells"]
    runs = 1 if cells else 0
    maxrun, cur = 1, 1
    for a, b in zip(cells, cells[1:]):
        if a == b: cur += 1; maxrun = max(maxrun, cur)
        else: cur = 1
    cnt = Counter(cells)
    print(f"{lab}: cells={dict(cnt)} maxrun={maxrun}", flush=True)
print("DIAG2 DONE")
