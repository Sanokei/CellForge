#!/usr/bin/env python3
"""Diag8: Fisher bigram probe. Among dot positions, find the linear direction
that best separates prev==34 (true suffix) from prev!=34 (decimal dots).
d_act = mean(z|dot,prev=34) - mean(z|dot,prev!=34). Measure separation.
"""
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

def ln(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5)

def q(v):
    v = v.float()
    return {f"{qq:.2f}": round(float(torch.quantile(v, qq)), 3)
            for qq in (0.0, 0.25, 0.5, 0.75, 1.0)}

Z_true, Z_false = [], []
for lab, dd in data.items():
    for d in dd["decs"]:
        z = ln(d["X"]); ids = d["ids"]
        dot = torch.nonzero(ids == 14).flatten()
        if len(dot) == 0:
            continue
        prev = ids[torch.clamp(dot - 1, min=0)]
        is_true = (prev == 34)
        Z_true.append(z[dot[is_true]]); Z_false.append(z[dot[~is_true]])
Z_true = torch.cat(Z_true); Z_false = torch.cat(Z_false)
print(f"true={Z_true.shape[0]} false={Z_false.shape[0]}", flush=True)

d_act = Z_true.mean(0) - Z_false.mean(0)
d_act = d_act / (d_act.norm() + 1e-12)
f_true = Z_true @ d_act
f_false = Z_false @ d_act
print("d_act.z | TRUE suffix:", q(f_true), flush=True)
print("d_act.z | other dots :", q(f_false), flush=True)
for tau in (-1.0, -0.5, 0.0, 0.5, 1.0):
    tp = float((f_true > tau).float().mean())
    fp = float((f_false > tau).float().mean())
    print(f"  tau={tau}: TP {tp:.3f} FP {fp:.3f}", flush=True)

# also the empirical dot detector for reference
Z_dot = torch.cat([Z_true, Z_false])
Z_nondot = []
for lab, dd in data.items():
    for d in dd["decs"]:
        z = ln(d["X"]); ids = d["ids"]
        nd = (ids != 14)
        Z_nondot.append(z[nd])
Z_nondot = torch.cat(Z_nondot)
e_dot = Z_dot.mean(0) - Z_nondot.mean(0)
e_dot = e_dot / (e_dot.norm() + 1e-12)
print("e_dot.z | dot:", q(Z_dot @ e_dot), flush=True)
print("e_dot.z | nondot:", q(Z_nondot @ e_dot), flush=True)

torch.save({"d_act": d_act, "e_dot": e_dot}, os.path.join(WD, "cal_dirs2.pt"))
print("DIAG8 DONE")
