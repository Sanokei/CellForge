#!/usr/bin/env python3
"""Diag9: calibrate epsilon-greedy gate. r = fixed random unit vector.
Measure r.z' at true suffix positions; find threshold for ~2.5% fire rate.
Also verify r.z' is ~independent of (d_act, e_dot) [pseudo-random]."""
import os, sys
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))

WD = "/srv/cellforge-arena/researchers/beta/gen3"
data = torch.load(os.path.join(WD, "cal_data.pt"), map_location="cpu",
                  weights_only=False)
cal = torch.load(os.path.join(WD, "cal_dirs2.pt"), map_location="cpu",
                 weights_only=False)
d_act = cal["d_act"].float(); d_act /= d_act.norm()
e_dot = cal["e_dot"].float(); e_dot /= e_dot.norm()

torch.manual_seed(42)
r = torch.randn(128); r /= r.norm()

def ln(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5)

def q(v):
    v = v.float()
    return {f"{qq:.3f}": round(float(torch.quantile(v, qq)), 3)
            for qq in (0.5, 0.9, 0.95, 0.975, 0.99)}

vals, acts, dots = [], [], []
for lab, dd in data.items():
    for d in dd["decs"]:
        z = ln(d["X"]); ids = d["ids"]
        dot = torch.nonzero(ids == 14).flatten()
        if len(dot) == 0:
            continue
        prev = ids[torch.clamp(dot - 1, min=0)]
        is_true = (prev == 34)
        vals.append(z[dot[is_true]] @ r)
        acts.append(z[dot[is_true]] @ d_act)
        dots.append(z[dot[is_true]] @ e_dot)
vals = torch.cat(vals)
print("r.z' at true suffix:", q(vals), flush=True)
for eps in (0.01, 0.025, 0.05):
    tau = float(torch.quantile(vals, 1 - eps))
    print(f"  eps={eps}: tau={tau:.3f}", flush=True)
# independence: correlation of r.z' with d_act.z', e_dot.z'
acts = torch.cat(acts); dots = torch.cat(dots)
print(f"corr(r, d_act)={float(torch.corrcoef(torch.stack([vals, acts]))[0,1]):.3f}",
      flush=True)
print(f"corr(r, e_dot)={float(torch.corrcoef(torch.stack([vals, dots]))[0,1]):.3f}",
      flush=True)
torch.save({"r": r}, os.path.join(WD, "cal_dirs3.pt"))
print("DIAG9 DONE")
