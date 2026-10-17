#!/usr/bin/env python3
"""Beta G3 novelty-switch: grow bred root 4->5 layers with a hand-authored
ReLU-gated exploration switch as block 4. Pure DnD, zero grad steps.

Design (adapted from shared-cell-registry threshold micro-design):
- Registry pattern: explicit-bias threshold unit (carry.W / carry.b).
  Adaptation: bias-free transformer has no bias term; the threshold is the
  ReLU zero-crossing on the novelty projection, with the LayerNorm-affine
  shift computed in closed form (see below). Reported as ADAPTED.
- Block 4: attention zeroed (pure MLP switch, no attention risk).
  ln1/ln2 standard. mlp.c_fc rows 0..k-1 = s * (n_dir (*) w_f) read the
  novelty direction; mlp.c_proj cols = (alpha/k) * p_pol write the
  exploration policy vector.

LayerNorm bookkeeping: the switch MLP reads ln2(x3) with ln2 = standard
(w=1,b=0), while n_dir was measured in ln_f-output space. With
ln_f(x) = w_f * ((x-mu)/sigma) + b_f and ln2(x) = (x-mu)/sigma:
  w_row = s * (n_dir (*) w_f)  =>  pre = s * (n_dir . h - n_dir . b_f)
where h = ln_f(x3). So the neuron fires iff (n_dir.h) > tau with
tau = n_dir.b_f (computed and printed). Dev center is at -D/2, novel at
+D/2 (D = |tx_mean - dev_mean|); tau ~= 0 lands near the midpoint.

p_pol (policy): direction in residual space that the lm_head maps to
higher explore-token (sweep/try) and lower macro-token logits:
  p_pol = normalize(mean(lm_head[427],lm_head[53]) - mean(lm_head[1170,1519,145]))

Usage: build_switch.py --base beta_g3_iter1.pt --s 4 --alpha 3 --k 16 --tag switch1
"""
import argparse
import os
import sys

import torch

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
from gpt import GPT  # noqa: E402

WD = "/srv/cellforge-arena/researchers/beta/gen2g3"

EXPLORE_IDS = [427, 53]            # sweep, try (reachable)
MACRO_IDS = [1170, 1519, 145]      # repeat_click, ghost_follow, pick_drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="beta_g3_iter1.pt")
    ap.add_argument("--s", type=float, default=4.0)      # detector sharpness
    ap.add_argument("--alpha", type=float, default=3.0)  # policy strength
    ap.add_argument("--k", type=int, default=16)         # switch neurons
    ap.add_argument("--t", type=float, default=0.0)      # threshold shift via m_dir
    ap.add_argument("--tag", default="switch1")
    a = ap.parse_args()

    base = torch.load(os.path.join(WD, a.base), map_location="cpu",
                      weights_only=False)
    probe = torch.load(os.path.join(WD, "probe_dirs.pt"), map_location="cpu",
                       weights_only=False)
    n_dir = probe["n_dir"].float()          # ln_f-space novelty direction
    m_dir = probe["m_dir"].float()          # ln_f-space mean direction
    cfg = dict(base["cfg"])
    assert cfg["n_layer"] == 4
    cfg["n_layer"] = 5

    st = base["state"]
    w_f = st["ln_f.weight"].float()
    b_f = st["ln_f.bias"].float()

    # threshold analysis (printed for the record)
    tau = float(n_dir @ b_f)
    # D estimated from probe cosines: recompute quickly from saved dirs is not
    # possible; report tau only. Selectivity verified empirically via dev eval.
    print(f"tau = n_dir.b_f = {tau:.4f} (fires iff n_dir.h > tau)", flush=True)

    # detector row in ln2-input geometry: w = s * ((n_dir - t*m_dir) (*) w_f)
    read = (n_dir - a.t * m_dir) * w_f
    read = a.s * read

    # policy vector from lm_head rows
    lm = st["lm_head.weight"].float()
    p_exp = lm[EXPLORE_IDS].mean(0)
    p_mac = lm[MACRO_IDS].mean(0)
    p_pol = p_exp - p_mac
    p_pol = p_pol / (p_pol.norm() + 1e-12)
    print(f"p_pol norm=1; cos(p_pol, lm[427])={float(p_pol @ lm[427] / (lm[427].norm()+1e-12)):.3f}",
          flush=True)

    # build 5-layer model, copy base into blocks 0..3
    model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head",
                                       "n_embd", "block_size", "dropout")})
    new_state = model.state_dict()
    for k, v in st.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v.float()
        else:
            print(f"SKIP {k} {tuple(v.shape)}", flush=True)
    # block 4: ln1/ln2 standard already (init); zero attention; author MLP
    B = model.blocks[4]
    with torch.no_grad():
        B.attn.c_attn.weight.zero_()
        B.attn.c_proj.weight.zero_()
        B.mlp.c_fc.weight.zero_()
        B.mlp.c_proj.weight.zero_()
        for i in range(a.k):
            B.mlp.c_fc.weight[i] = read
            B.mlp.c_proj.weight[:, i] = (a.alpha / a.k) * p_pol
    for k, v in B.state_dict().items():
        new_state[f"blocks.4.{k}"] = v.float()

    # verify: every tensor finite; base blocks bit-identical to input
    for k, v in new_state.items():
        assert torch.isfinite(v).all(), f"non-finite {k}"
    for k in st:
        if k.startswith("blocks.") and int(k.split(".")[1]) < 4:
            assert torch.equal(new_state[k], st[k].float()), f"base modified: {k}"
    n_params = sum(v.numel() for v in new_state.values())
    print(f"params: {n_params} (<=10M: {n_params <= 10_000_000})", flush=True)

    out = os.path.join(WD, f"beta_g3_{a.tag}.pt")
    torch.save({"cfg": cfg, "state": new_state}, out)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
