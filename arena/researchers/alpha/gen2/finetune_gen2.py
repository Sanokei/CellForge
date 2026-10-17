#!/usr/bin/env python3
"""finetune_gen2.py — ALPHA-G2 "Rehearsal Fix" finetune.

Adapted from lm/train_lm.py (v3 finetune pattern): same GPT arch, same block/batch,
but (1) inits from base_dnd1.pt (router_v3 + dnd1 LoRA deltas), (2) finetunes on a
MIXED corpus: discovery traces_fc.jsonl + answer rehearsal traces_w2.jsonl +
traces_grammar_x100.jsonl (~60/25/15 by tokens), (3) HARD CAP: lr <= 1e-4.

Writes ONLY to the alpha-g2 researcher workdir. Never touches lm/.
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


def load_stream(name, path, stoi, frac):
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if frac < 1.0:
        import random
        rnd = random.Random(20261016)
        rnd.shuffle(docs)
        n = int(len(docs) * frac)
        docs = docs[:n]
    eos = stoi["<eos>"]
    data = []
    for d in docs:
        data.extend(encode(render_trace(d), stoi) + [eos])
    t = torch.tensor(data, dtype=torch.long)
    print(f"corpus {name}: docs={len(docs)} tokens={len(t)}", flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init", default=os.path.join(WORK, "base_dnd1.pt"))
    ap.add_argument("--out", default=os.path.join(WORK, "alpha_gen2.pt"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert a.lr <= 1e-4, f"LR cap violated: {a.lr}"

    torch.manual_seed(a.seed)
    vocab, stoi = load_vocab(os.path.join(LMDIR, "vocab.json"))
    print(f"vocab: {len(vocab)} tokens", flush=True)

    # w2 fraction: target 297,660 tokens of 3,743,067 -> 0.0795
    MIX = {
        "fc": ("traces_fc.jsonl", 1.0),                 # discovery (full)
        "w2": ("traces_w2.jsonl", 297660 / 3743067),    # answers (~8% sample)
        "grammar": ("traces_grammar_x100.jsonl", 1.0),  # answers (full)
    }
    streams, probs = [], []
    for name, (rel, frac) in MIX.items():
        s = load_stream(name, os.path.join(LMDIR, rel), stoi, frac)
        streams.append(s)
        probs.append(len(s))
    total = sum(probs)
    probs = [p / total for p in probs]
    print(f"mixed totals: {total} tokens, probs="
          f"{[f'{n}={p:.3f}' for n, p in zip(MIX, probs)]}", flush=True)

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert cfg["vocab_size"] == len(vocab) and cfg["n_layer"] == 4 \
        and cfg["n_head"] == 4 and cfg["n_embd"] == 128, f"cfg drift: {cfg}"
    model = GPT(**cfg)
    model.load_state_dict(ck["state"])
    print(f"fine-tuning from {a.init}; params={model.param_count()}; "
          f"lr={a.lr}; steps={a.steps}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    model.train()
    t0 = time.time()
    rng = torch.Generator().manual_seed(a.seed)
    for step in range(a.steps):
        xs, ys = [], []
        for _ in range(a.batch):
            si = torch.multinomial(torch.tensor(probs), 1,
                                   generator=rng).item()
            s = streams[si]
            i = torch.randint(0, len(s) - a.block - 1, (1,),
                              generator=rng).item()
            xs.append(s[i:i + a.block])
            ys.append(s[i + 1:i + a.block + 1])
        x = torch.stack(xs)
        y = torch.stack(ys)
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"step {step+1}/{a.steps} loss {loss.item():.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    torch.save({"cfg": cfg, "state": model.state_dict()}, a.out)
    meta = {"init": a.init, "lr": a.lr, "steps": a.steps,
            "mix": {n: {"path": rel, "probs": p}
                    for (n, (rel, _)), p in zip(MIX.items(), probs)},
            "final_loss": float(loss.item())}
    with open(a.out.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
