#!/usr/bin/env python3
"""Diagnose why pooled novelty doesn't separate: inspect p_t directly."""
import os, sys
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
WD = "/srv/cellforge-arena/researchers/beta/gen3"
data = torch.load(os.path.join(WD, "cal_data.pt"), map_location="cpu",
                  weights_only=False)
cal = torch.load(os.path.join(WD, "cal_dirs.pt"), map_location="cpu",
                 weights_only=False)
n_hat = cal["n_hat"].float()

def ln(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5)

def q(v):
    v = v.float()
    return {f"{qq:.2f}": round(float(torch.quantile(v, qq)), 4)
            for qq in (0.0, 0.25, 0.5, 0.75, 1.0)}

# per-position s_j separation first
s_dev, s_tx = [], []
p_dot_dev, p_dot_tx = [], []
p_early_dev, p_early_tx = [], []   # p_t at t~10 (early context)
p_late_dev, p_late_tx = [], []     # p_t at last position
for lab, dd in data.items():
    tx = lab.startswith("TX")
    for d in dd["decs"]:
        z = ln(d["X"]); ids = d["ids"]
        s = z @ n_hat
        (s_tx if tx else s_dev).append(s)
        t = torch.arange(1, s.shape[0] + 1).float()
        p = torch.cumsum(s, 0) / t
        dot = (ids == 14)
        (p_dot_tx if tx else p_dot_dev).append(p[dot])
        (p_early_tx if tx else p_early_dev).append(p[min(9, len(p)-1):min(10, len(p))])
        (p_late_tx if tx else p_late_dev).append(p[-1:])
s_dev = torch.cat(s_dev); s_tx = torch.cat(s_tx)
print("per-position s: DEV mean/std", round(float(s_dev.mean()),4), round(float(s_dev.std()),4))
print("per-position s: TX  mean/std", round(float(s_tx.mean()),4), round(float(s_tx.std()),4))
p_dot_dev = torch.cat(p_dot_dev); p_dot_tx = torch.cat(p_dot_tx)
print("p@dot: DEV", q(p_dot_dev))
print("p@dot: TX ", q(p_dot_tx))
pe_dev = torch.cat(p_early_dev); pe_tx = torch.cat(p_early_tx)
print("p@early(t=10): DEV", q(pe_dev))
print("p@early(t=10): TX ", q(pe_tx))
pl_dev = torch.cat(p_late_dev); pl_tx = torch.cat(p_late_tx)
print("p@late: DEV", q(pl_dev))
print("p@late: TX ", q(pl_tx))
# where in the context does the tx/dev s difference live? split prompt vs generated:
# prompt = positions before the last 64 (generated); approximate
s_prompt_dev, s_prompt_tx, s_gen_dev, s_gen_tx = [], [], [], []
for lab, dd in data.items():
    tx = lab.startswith("TX")
    for d in dd["decs"]:
        z = ln(d["X"]); s = z @ n_hat; T = s.shape[0]
        gp = s[:max(0, T-64)]; gg = s[max(0, T-64):]
        if tx: s_prompt_tx.append(gp); s_gen_tx.append(gg)
        else: s_prompt_dev.append(gp); s_gen_dev.append(gg)
for name, v in [("prompt_dev", s_prompt_dev), ("prompt_tx", s_prompt_tx),
                ("gen_dev", s_gen_dev), ("gen_tx", s_gen_tx)]:
    v = torch.cat(v)
    print(f"s {name}: mean {float(v.mean()):.4f} std {float(v.std()):.4f} n={v.numel()}")
print("DIAG DONE")
