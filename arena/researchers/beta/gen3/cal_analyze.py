#!/usr/bin/env python3
"""Beta G4 calibration analysis (offline, from cal_data.pt).

Replicates the planned block-4 EXACTLY (pooling head + gate features) and
reports the empirical distributions needed to set:
  c_pool, alpha, beta, Gamma, s_boost  -> cal_params.json (hand-set after review)

Labels DEV/TX only. Transfer = measure-only.
"""
import os, sys, json
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
WD = "/srv/cellforge-arena/researchers/beta/gen3"

data = torch.load(os.path.join(WD, "cal_data.pt"), map_location="cpu",
                  weights_only=False)
labels = sorted(data.keys())
DEV = [l for l in labels if l.startswith("DEV")]
TX = [l for l in labels if l.startswith("TX")]
print(f"labels: {DEV} | {TX}", flush=True)

def ln(x):
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-5)

# per-decision tensors
decs = []
for lab in labels:
    tx = lab.startswith("TX")
    for d in data[lab]["decs"]:
        decs.append({"z": ln(d["X"]), "ids": d["ids"], "tx": tx,
                     "lab": lab})
print(f"decisions: {len(decs)}", flush=True)
Z = torch.cat([d["z"] for d in decs], 0)
is_tx_pos = torch.cat([torch.full((d["z"].shape[0],), d["tx"]) for d in decs]).bool()
is_dot_pos = torch.cat([d["ids"] == 14 for d in decs])

# --- novelty direction (ln1 space) ---
n_hat = Z[is_tx_pos].mean(0) - Z[~is_tx_pos].mean(0)
n_hat = n_hat / (n_hat.norm() + 1e-12)

# --- dot detector: empirical Fisher direction (dev only) ---
z_dev = Z[~is_tx_pos]; dot_dev = is_dot_pos[~is_tx_pos]
e_dot = z_dev[dot_dev].mean(0) - z_dev[~dot_dev].mean(0)
e_dot = e_dot / (e_dot.norm() + 1e-12)

# --- quiet channel: min eigvec of Cov(z_dev), orthogonalized vs ones ---
mu = z_dev.mean(0, keepdim=True)
C = (z_dev - mu).t() @ (z_dev - mu) / z_dev.shape[0]
evals, evecs = torch.linalg.eigh(C)
ones = torch.ones(128) / (128 ** 0.5)
u0 = evecs[:, 0] - (evecs[:, 0] @ ones) * ones
u0 = u0 / (u0.norm() + 1e-12)
print("eigvals 5 smallest:", [round(v, 6) for v in evals[:5].tolist()], flush=True)
print("sqrt(u0 C u0) =", round(float((u0 @ C @ u0).sqrt()), 5), flush=True)

def q(v):
    v = v.float()
    return {f"{qq:.3f}": round(float(torch.quantile(v, qq)), 4)
            for qq in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)}

# --- replicate block-4 for candidate c_pool ---
for c_pool in (0.25, 0.5, 1.0):
    sn_dev_dot, sn_tx_dot, sn_tx_nondot = [], [], []
    sp_dev_dot, sp_dev_nondot, sp_tx_dot = [], [], []
    for d in decs:
        z, ids, tx = d["z"], d["ids"], d["tx"]
        s = z @ n_hat
        t = torch.arange(1, s.shape[0] + 1).float()
        p = torch.cumsum(s, 0) / t                      # pooling head output /c_pool
        hp = z + (p * c_pool).unsqueeze(1) * u0         # h' = x + p_t*u_out
        zp = ln(hp)                                    # ln2 (normalized part)
        s_nov = zp @ u0
        s_pos = zp @ e_dot
        dot = (ids == 14)
        if tx:
            sn_tx_dot.append(s_nov[dot]); sn_tx_nondot.append(s_nov[~dot])
            sp_tx_dot.append(s_pos[dot])
        else:
            sn_dev_dot.append(s_nov[dot])
            sp_dev_dot.append(s_pos[dot]); sp_dev_nondot.append(s_pos[~dot])
    sn_dev_dot = torch.cat(sn_dev_dot); sn_tx_dot = torch.cat(sn_tx_dot)
    sn_tx_nondot = torch.cat(sn_tx_nondot)
    sp_dev_dot = torch.cat(sp_dev_dot); sp_dev_nondot = torch.cat(sp_dev_nondot)
    sp_tx_dot = torch.cat(sp_tx_dot)
    print(f"--- c_pool={c_pool} ---", flush=True)
    print("  s_nov@dot DEV:", q(sn_dev_dot), flush=True)
    print("  s_nov@dot TX :", q(sn_tx_dot), flush=True)
    tau = float(torch.quantile(sn_dev_dot, 0.999))
    print(f"  tau_nov(dev99.9)={tau:.4f}  TX@dot above: "
          f"{float((sn_tx_dot > tau).float().mean()):.3f}  "
          f"TX@nondot above: {float((sn_tx_nondot > tau).float().mean()):.3f}", flush=True)
    print("  s_pos@dot DEV:", q(sp_dev_dot), flush=True)
    print("  s_pos@nondot DEV:", q(sp_dev_nondot), flush=True)
    print("  s_pos@dot TX :", q(sp_tx_dot), flush=True)
    # joint raw score a_raw = s_nov + s_pos
    a_dev_dot = sn_dev_dot + sp_dev_dot
    a_tx_dot = sn_tx_dot + sp_tx_dot
    Ga = float(torch.quantile(a_dev_dot, 0.999))
    print(f"  joint: Gamma(dev99.9)={Ga:.4f} TX@dot above: "
          f"{float((a_tx_dot > Ga).float().mean()):.3f}", flush=True)

# --- baseline exploratory-call rates + dot context check ---
for lab in labels:
    cells = data[lab]["cells"]
    e = sum(1 for x in cells if x in ("act.sweep", "act.try"))
    print(f"{lab}: explore {e}/{len(cells)}", flush=True)
# how often is a dot NOT preceded by 'act'(34) on dev?
nd, nda = 0, 0
for d in decs:
    if d["tx"]:
        continue
    ids = d["ids"]; dot = torch.nonzero(ids == 14).flatten()
    prev = ids[torch.clamp(dot - 1, min=0)]
    nd += dot.numel(); nda += int((prev != 34).sum())
print(f"dev dot positions with prev!=act: {nda}/{nd} = {nda/max(1,nd):.3f}", flush=True)

torch.save({"n_hat": n_hat, "e_dot": e_dot, "u0": u0}, os.path.join(WD, "cal_dirs.pt"))
print("ANALYSIS DONE", flush=True)
