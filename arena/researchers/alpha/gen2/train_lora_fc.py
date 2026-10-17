#!/usr/bin/env python3
"""train_lora_fc.py — ALPHA-G2 discovery LoRA.

Trains rank-r LoRA adapters on DISCOVERY (fc) traces only, base FROZEN.
Init: A ~ N(0, 0.02), B = 0 -> initial model == base exactly.
Saves deltas JSON (same format as dnd1_deltas.json) for alpha-scaled merging.
LR capped at 1e-4.
"""
import argparse
import json
import os
import sys
import time

LMDIR = "/srv/arc3-claude/v6run/model/lm"
sys.path.insert(0, LMDIR)
from tok import load_vocab, encode, render_trace  # noqa: E402
from gpt import GPT  # noqa: E402

import torch

WORK = "/srv/cellforge-arena/researchers/alpha/gen2"

# same behavior-layer groups dnd1 targeted
TARGETS = [
    "lm_head.weight",
    "blocks.2.attn.c_attn.weight",
    "blocks.3.attn.c_attn.weight",
    "blocks.3.mlp.c_proj.weight",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--init", default=os.path.join(WORK, "base_dnd1.pt"))
    ap.add_argument("--out", default=os.path.join(WORK, "lora_fc_deltas.json"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    assert a.lr <= 1e-4, f"LR cap violated: {a.lr}"

    torch.manual_seed(a.seed)
    vocab, stoi = load_vocab(os.path.join(LMDIR, "vocab.json"))

    docs = []
    with open(os.path.join(LMDIR, "traces_fc.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    eos = stoi["<eos>"]
    data = []
    for d in docs:
        data.extend(encode(render_trace(d), stoi) + [eos])
    data = torch.tensor(data, dtype=torch.long)
    print(f"fc tokens: {len(data)}", flush=True)

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = GPT(**cfg)
    model.load_state_dict(ck["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # LoRA params: delta = A @ B added to W
    loras = {}
    params = []
    for name in TARGETS:
        W = dict(model.named_parameters())[name]
        in_f, out_f = W.shape
        A = torch.randn(in_f, a.rank) * 0.02
        B = torch.zeros(a.rank, out_f)
        A.requires_grad_(True)
        B.requires_grad_(True)
        loras[name] = (A, B)
        params += [A, B]
    print(f"lora params: {sum(p.numel() for p in params)}", flush=True)

    # hook: add deltas into forward by patching the weight tensors' data
    # (simpler: materialize W+delta per step via functional hooks is complex;
    #  instead temporarily add to .data, forward, then subtract)
    named = dict(model.named_parameters())

    opt = torch.optim.AdamW(params, lr=a.lr)
    model.train()  # dropout 0.0 anyway; enables grad flow through patched data
    t0 = time.time()
    for step in range(a.steps):
        # apply deltas
        for name in TARGETS:
            A, B = loras[name]
            named[name].data = named[name].data + A @ B
        ix = torch.randint(0, len(data) - a.block, (a.batch,))
        x = torch.stack([data[i:i + a.block] for i in ix])
        y = torch.stack([data[i + 1:i + a.block + 1] for i in ix])
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        # remove deltas before optimizer step (keep base frozen exactly)
        for name in TARGETS:
            A, B = loras[name]
            named[name].data = named[name].data - (A @ B).detach()
        opt.step()
        if (step + 1) % 50 == 0:
            norms = {n: float((loras[n][0] @ loras[n][1]).norm())
                     for n in TARGETS}
            print(f"step {step+1}/{a.steps} loss {loss.item():.4f} "
                  f"dWnorms={[f'{v:.4f}' for v in norms.values()]} "
                  f"{time.time()-t0:.0f}s", flush=True)

    deltas = {}
    for name in TARGETS:
        A, B = loras[name]
        W = named[name].data
        dW = (A @ B).detach()
        deltas[name] = {"A": A.detach().tolist(), "B": B.detach().tolist(),
                        "rel_norm": float(dW.norm() / W.norm())}
    with open(a.out, "w") as f:
        json.dump(deltas, f)
    print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
