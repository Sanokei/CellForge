#!/usr/bin/env python3
"""Diag7: bigram detector test. At dot positions (token 14), is
'e_act . z' (wte[34]='act' direction) elevated when prev token == 34
vs prev != 34? If yes, we have a hand-authorable (act,dot) bigram gate.
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
CK = torch.load("/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt", map_location="cpu",
                weights_only=False)
wte = CK["state"]["wte.weight"].float()
e_act = wte[34] / (wte[34].norm() + 1e-12)
e_dot = wte[14] / (wte[14].norm() + 1e-12)

def ln(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5)

def q(v):
    v = v.float()
    return {f"{qq:.2f}": round(float(torch.quantile(v, qq)), 3)
            for qq in (0.0, 0.25, 0.5, 0.75, 1.0)}

true_suf, false_dot = [], []  # true suffix (prev=act) vs other dots
for lab, dd in data.items():
    for d in dd["decs"]:
        z = ln(d["X"]); ids = d["ids"]
        dot = torch.nonzero(ids == 14).flatten()
        if len(dot) == 0:
            continue
        prev = ids[torch.clamp(dot - 1, min=0)]
        f_act = z[dot] @ e_act
        f_dot = z[dot] @ e_dot
        is_true = (prev == 34)
        true_suf.append(torch.stack([f_act[is_true], f_dot[is_true]], 1))
        false_dot.append(torch.stack([f_act[~is_true], f_dot[~is_true]], 1))
true_suf = torch.cat(true_suf); false_dot = torch.cat(false_dot)
print(f"true suffix positions: {true_suf.shape[0]}, other dots: {false_dot.shape[0]}",
      flush=True)
print("e_act.z | TRUE suffix :", q(true_suf[:, 0]), flush=True)
print("e_act.z | other dots  :", q(false_dot[:, 0]), flush=True)
print("e_dot.z | TRUE suffix :", q(true_suf[:, 1]), flush=True)
print("e_dot.z | other dots  :", q(false_dot[:, 1]), flush=True)
# joint: how well does (e_act, e_dot) separate? simple threshold on e_act
for tau in (0.0, 0.5, 1.0, 1.5, 2.0):
    tp = float((true_suf[:, 0] > tau).float().mean())
    fp = float((false_dot[:, 0] > tau).float().mean())
    print(f"  tau_act={tau}: true-positive {tp:.3f}, false-positive {fp:.3f}", flush=True)
print("DIAG7 DONE")
