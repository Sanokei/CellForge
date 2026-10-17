#!/usr/bin/env python3
"""Diag3: logit gaps at dot positions (token 14) on dev vs tx, root model.

At each dot position (cell-suffix decision point), measure:
  gap = logit(top suffix token) - logit(sweep/try token)
for suffix set {sweep:427, try:53, click_at:49, ...}. Also the top-1 token id.
Tells us: how big a boost is needed to flip tx decisions, and how safe dev is.
"""
import os, sys, json
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
from tok import load_vocab, encode
from gpt import GPT

WD = "/srv/cellforge-arena/researchers/beta/gen3"
data = torch.load(os.path.join(WD, "cal_data.pt"), map_location="cpu",
                  weights_only=False)
CKPT = "/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = GPT(**{k: cfg[k] for k in
               ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
model.load_state_dict(ck["state"])
model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))

# suffix token ids for the 7 spellable act.* cells
SUFFIX = {}
for cell in ["act.sweep", "act.try", "act.click_at", "act.pick_drop",
             "act.repeat_click", "act.ghost_follow", "act.reset"]:
    ids = encode(cell, stoi)
    assert ids[:2] == [34, 14], (cell, ids)
    SUFFIX[cell] = ids[2]
print("suffix ids:", SUFFIX, flush=True)
SIDS = torch.tensor(sorted(set(SUFFIX.values())))
EXPLORE = torch.tensor([SUFFIX["act.sweep"], SUFFIX["act.try"]])

def q(v):
    v = v.float()
    return {f"{qq:.2f}": round(float(torch.quantile(v, qq)), 3)
            for qq in (0.0, 0.25, 0.5, 0.75, 1.0)}

@torch.no_grad()
def gaps_for(dec):
    X = dec["X"]  # not needed; rerun forward for logits
    ids = dec["ids"]
    logits, _ = model(ids.unsqueeze(0))
    logits = logits[0]  # T x 2209
    dot = (ids == 14)
    lg = logits[dot][:, SIDS]          # n_dot x 7 (suffix logits)
    topv, topi = lg.max(1)
    exv, _ = lg[:, [0, 1]].max(1) if False else (None, None)
    # map EXPLORE into SIDS order
    eidx = [(SIDS == e).nonzero().item() for e in EXPLORE]
    exv = lg[:, eidx].max(1).values
    gap = topv - exv                    # how far ahead is top vs best explore
    top_is_explore = torch.tensor([SIDS[i].item() in EXPLORE.tolist()
                                   for i in topi.tolist()])
    return gap, top_is_explore, topi

gd, gt = [], []
for lab, dd in data.items():
    tx = lab.startswith("TX")
    for d in dd["decs"]:
        gap, tie, _ = gaps_for(d)
        (gt if tx else gd).append(torch.stack([gap, tie.float()], 1))
gd = torch.cat(gd); gt = torch.cat(gt)
print(f"n_dot dev={gd.shape[0]} tx={gt.shape[0]}", flush=True)
print("gap(top-explore) DEV:", q(gd[:, 0]), " top_is_explore frac:",
      round(float(gd[:, 1].mean()), 4), flush=True)
print("gap(top-explore) TX :", q(gt[:, 0]), " top_is_explore frac:",
      round(float(gt[:, 1].mean()), 4), flush=True)
# what boost flips what fraction? P(gap < b) for b in 1..10
for b in (1, 2, 3, 5, 8, 12):
    print(f"  boost +{b}: dev flip frac={float((gd[:,0]<b).float().mean()):.3f} "
          f"tx flip frac={float((gt[:,0]<b).float().mean()):.3f}", flush=True)
print("DIAG3 DONE")
