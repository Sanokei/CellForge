#!/usr/bin/env python3
"""Diag10: direct Gamma calibration. Compute S = r.z' + d_act.z' + e_dot.z'
at true suffix positions; find Gamma for target fire rates."""
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
cal2 = torch.load(os.path.join(WD, "cal_dirs2.pt"), map_location="cpu",
                  weights_only=False)
d_act = cal2["d_act"].float(); d_act /= d_act.norm()
e_dot = cal2["e_dot"].float(); e_dot /= e_dot.norm()
cal3 = torch.load(os.path.join(WD, "cal_dirs3.pt"), map_location="cpu",
                  weights_only=False)
r = cal3["r"].float(); r /= r.norm()

def ln(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5)

S_true, S_other = [], []
for lab, dd in data.items():
    for d in dd["decs"]:
        z = ln(d["X"]); ids = d["ids"]
        dot = torch.nonzero(ids == 14).flatten()
        if len(dot) == 0:
            continue
        prev = ids[torch.clamp(dot - 1, min=0)]
        is_true = (prev == 34)
        s = z[dot] @ r + z[dot] @ d_act + z[dot] @ e_dot
        S_true.append(s[is_true]); S_other.append(s[~is_true])
S_true = torch.cat(S_true); S_other = torch.cat(S_other)
print(f"n_true={S_true.shape[0]} n_other={S_other.shape[0]}", flush=True)
for q_ in (0.9, 0.95, 0.975, 0.99):
    g = float(torch.quantile(S_true, q_))
    fp = float((S_other > g).float().mean())
    print(f"Gamma={g:.3f} -> TP {1-q_:.3f}, FP(other dots) {fp:.4f}", flush=True)
print("DIAG10 DONE")
