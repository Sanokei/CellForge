#!/usr/bin/env python3
"""Diag6: is absolute position (early vs late in context) linearly decodable
from block-3 output residuals? Tests the 'explore-first' gating feature.

Feature: P_dir . z_t where P_dir separates early vs late wpe.
If separable, we can gate exploration on (early-in-context AND dot-position)
without any novelty detection.
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
wpe = CK["state"]["wpe.weight"].float()  # 192 x 128

def ln(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5)

# direction: early positions vs late positions (in wpe space)
P_dir = wpe[0:60].mean(0) - wpe[120:192].mean(0)
P_dir = P_dir / (P_dir.norm() + 1e-12)

def q(v):
    v = v.float()
    return {f"{qq:.2f}": round(float(torch.quantile(v, qq)), 3)
            for qq in (0.0, 0.25, 0.5, 0.75, 1.0)}

# measure P_dir.z_t on real residuals, binned by absolute position t
early, mid, late = [], [], []
for lab, dd in data.items():
    for d in dd["decs"]:
        z = ln(d["X"])
        T = z.shape[0]
        f = z @ P_dir
        # absolute position t within the 192-window: capture was out[:,-192:],
        # so position = (192 - T) + arange(T)  [right-aligned]
        t = (192 - T) + torch.arange(T).float()
        early.append(f[t < 80]); mid.append(f[(t >= 80) & (t < 130)])
        late.append(f[t >= 130])
early = torch.cat(early); mid = torch.cat(mid); late = torch.cat(late)
print("P_dir.z | t<80   :", q(early))
print("P_dir.z | 80<=t<130:", q(mid))
print("P_dir.z | t>=130 :", q(late))
# also: where are dot positions?
print("DIAG6 DONE")
