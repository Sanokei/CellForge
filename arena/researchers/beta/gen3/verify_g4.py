#!/usr/bin/env python3
"""Verify beta_g4: gate fire rates + actual logit boost on 427/53."""
import os, sys
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
from gpt import GPT

WD = "/srv/cellforge-arena/researchers/beta/gen3"
data = torch.load(os.path.join(WD, "cal_data.pt"), map_location="cpu",
                  weights_only=False)

def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd",
                                  "block_size", "dropout")})
    m.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    m.eval()
    return m

root = load_ckpt("/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt")
g4 = load_ckpt(os.path.join(WD, "beta_g4_t4.pt"))

# hook the gate neuron (block4 mlp c_fc output, neuron 0)
gate_act = {}
def hk(m, i, o):
    gate_act["g"] = F.gelu(o)[0, :, 0].detach()
g4.blocks[4].mlp.c_fc.register_forward_hook(hk)

@torch.no_grad()
def run(m, ids):
    return m(ids.unsqueeze(0))[0][0]

n_true_fire, n_true, n_other_fire, n_other = 0, 0, 0, 0
boosts = []
for lab, dd in list(data.items())[:6]:  # dev only for speed
    for d in dd["decs"][:4]:
        ids = d["ids"]
        lr = run(root, ids)
        lg = run(g4, ids)
        g = gate_act["g"]
        dot = (ids == 14)
        idx = torch.nonzero(dot).flatten()
        prev = ids[torch.clamp(idx - 1, min=0)]
        is_true = (prev == 34)
        fires = (g[idx] > 0.5)
        n_true += int(is_true.sum()); n_true_fire += int((fires & is_true).sum())
        n_other += int((~is_true).sum()); n_other_fire += int((fires & ~is_true).sum())
        dl = (lg - lr)[idx][:, [427, 53]]
        boosts.append(dl[is_true])
boosts = torch.cat(boosts)
print(f"gate: true-suffix fire {n_true_fire}/{n_true} = {n_true_fire/max(1,n_true):.3f}", flush=True)
print(f"gate: other-dot fire {n_other_fire}/{n_other} = {n_other_fire/max(1,n_other):.3f}", flush=True)
print(f"logit boost on 427/53 at true suffix (s_boost=1.0): "
      f"mean {boosts.mean().item():.3f} median {boosts.median().item():.3f}", flush=True)
print("VERIFY DONE")
