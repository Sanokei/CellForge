#!/usr/bin/env python3
"""Beta G3 iter2: iter1 + global reachable-action token boost.
Targets the 41%/decision parse-fail waste: raise P(real action) vs
P(<unk> no-op) state-independently. Pure DnD, zero grad steps.
Usage: author_iter2.py --b_act 0.2 --tag iter2
"""
import argparse
import os
import sys

import torch

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))

WD = "/srv/cellforge-arena/researchers/beta/gen2g3"
# reachable action-name tokens: click_at 49, try 53, sweep 427,
# pick_drop 145, reset 176, repeat_click 1170, ghost_follow 1519
ACTION_IDS = [49, 53, 427, 145, 176, 1170, 1519]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="beta_g3_iter1.pt")
    ap.add_argument("--b_act", type=float, default=0.2)
    ap.add_argument("--tag", default="iter2")
    a = ap.parse_args()

    ck = torch.load(os.path.join(WD, a.base), map_location="cpu",
                    weights_only=False)
    state = {k: v.float() for k, v in ck["state"].items()}
    probe = torch.load(os.path.join(WD, "probe_dirs.pt"), map_location="cpu",
                       weights_only=False)
    m_dir = probe["m_dir"].float()

    V = state["lm_head.weight"].shape[0]
    A = torch.zeros(V, 1)
    for t in ACTION_IDS:
        A[t, 0] = a.b_act
    dW = A @ m_dir.unsqueeze(0)
    state["lm_head.weight"] = state["lm_head.weight"] + dW
    print(f"lm_head action boost: rank 1 ||dW||/||W|| = "
          f"{dW.norm().item() / state['lm_head.weight'].norm().item():.4f}",
          flush=True)

    out = os.path.join(WD, f"beta_g3_{a.tag}.pt")
    torch.save({"cfg": ck["cfg"], "state": state}, out)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
