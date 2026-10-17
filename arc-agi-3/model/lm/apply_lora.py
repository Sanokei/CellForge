#!/usr/bin/env python3
"""apply_lora.py — apply authored LoRA deltas to router_v3.

Usage: apply_lora.py --base lm/router_v3.pt --deltas lm/dnd1_deltas.json \
                     --out lm/router_v3+dnd1.pt
W_new = W + A @ B per layer. Verifies shapes, finiteness, and reports
relative delta norms. Output keeps the {cfg, state} checkpoint format so
eval_lm.py / rhae_scorer.py consume it unchanged.
"""
import argparse
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--deltas", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ckpt = torch.load(a.base, map_location="cpu", weights_only=False)
    with open(a.deltas) as f:
        deltas = json.load(f)
    state = ckpt["state"]
    for key, d in deltas.items():
        if key not in state:
            raise KeyError(f"layer {key} not in checkpoint")
        W = state[key].float()
        A = torch.tensor(d["A"], dtype=torch.float32)
        B = torch.tensor(d["B"], dtype=torch.float32)
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"rank mismatch at {key}: {A.shape} vs {B.shape}")
        if tuple(A.shape[0:1]) + tuple(B.shape[1:2]) != tuple(W.shape):
            raise ValueError(
                f"shape mismatch at {key}: A{A.shape} B{B.shape} W{tuple(W.shape)}")
        delta = A @ B
        if not torch.isfinite(delta).all():
            raise ValueError(f"non-finite delta at {key}")
        Wn = W + delta
        if not torch.isfinite(Wn).all():
            raise ValueError(f"non-finite result at {key}")
        state[key] = Wn
        print(f"{key}: ||dW||/||W|| = "
              f"{delta.norm().item() / W.norm().item():.4f} "
              f"(rank {A.shape[1]})", flush=True)
    torch.save({"cfg": ckpt["cfg"], "state": state}, a.out)
    print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
