#!/usr/bin/env python3
"""Debug: does eps4's gate fire on cal_data? Compare live gate vs diag10."""
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

ck = torch.load(os.path.join(WD, "beta_g4_eps5.pt"), map_location="cpu",
                weights_only=False)
cfg = ck["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd",
                                  "block_size", "dropout")})
model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
model.eval()

gate = {}
def hk(m, i, o):
    gate["pre"] = o[0, :, 0].detach()          # pre-gelu
    gate["post"] = F.gelu(o)[0, :, 0].detach()  # post-gelu
model.blocks[4].mlp.c_fc.register_forward_hook(hk)

@torch.no_grad()
def check(dec):
    ids = dec["ids"]
    model(ids.unsqueeze(0))
    pre, post = gate["pre"], gate["post"]
    dot = (ids == 14)
    idx = torch.nonzero(dot).flatten()
    prev = ids[torch.clamp(idx - 1, min=0)]
    is_true = (prev == 34)
    return pre[idx], post[idx], is_true

n_fire_true, n_true, n_fire_other, n_other = 0, 0, 0, 0
for lab in ["DEV-1", "TX-A"]:
    for d in data[lab]["decs"][:3]:
        pre, post, is_true = check(d)
        fires = post > 0.5
        n_true += int(is_true.sum()); n_fire_true += int((fires & is_true).sum())
        n_other += int((~is_true).sum()); n_fire_other += int((fires & ~is_true).sum())
print(f"live gate on cal_data: true fire {n_fire_true}/{n_true}, "
      f"other fire {n_fire_other}/{n_other}", flush=True)
print("DEBUG DONE")
