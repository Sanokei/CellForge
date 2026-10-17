#!/usr/bin/env python3
"""finetune_gen2b.py — ALPHA-G2 "Rehearsal Fix" v2.

Fixes for v1 forgetting (v1: dev-6 0.215 < 0.31 guardrail; s5i5 L1 lost, parse
fails up): (1) FREEZE the 4 dnd1-touched param groups exactly — dnd1's
decisiveness/parse-reliability deltas stay load-bearing and untouched; discovery
behavior is learned only by the remaining ~99% of params. (2) More rehearsal:
mix fc 35 / w2 50 / grammar 15 by tokens. (3) LR stays at cap 1e-4, 300 steps.
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

# dnd1 delta keys: keep these EXACTLY as authored (decisiveness + parse reliability)
FROZEN = {
    "lm_head.weight",
    "blocks.2.attn.c_attn.weight",
    "blocks.3.attn.c_attn.weight",
    "blocks.3.mlp.c_proj.weight",
}


def tokenize(docs, stoi):
    eos = stoi["<eos>"]
    data = []
    for d in docs:
        data.extend(encode(render_trace(d), stoi) + [eos])
    return torch.tensor(data, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init", default=os.path.join(WORK, "base_dnd1.pt"))
    ap.add_argument("--out", default=os.path.join(WORK, "alpha_gen2b.pt"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    assert a.lr <= 1e-4, f"LR cap violated: {a.lr}"

    torch.manual_seed(a.seed)
    vocab, stoi = load_vocab(os.path.join(LMDIR, "vocab.json"))

    def load_docs(name):
        docs = []
        with open(os.path.join(LMDIR, name)) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs

    fc = load_docs("traces_fc.jsonl")                       # 714,385 tok
    w2 = load_docs("traces_w2.jsonl")                       # 3,743,067 tok
    gr = load_docs("traces_grammar_x100.jsonl")             # 173,000 tok

    import random
    rnd = random.Random(20261016)
    rnd.shuffle(w2)
    # target mix by tokens: fc 35% / w2 50% / grammar 15%, anchored on full fc
    t_fc = 714385
    T = t_fc / 0.35
    t_w2 = int(0.50 * T)
    t_gr = int(0.15 * T)

    s_fc = tokenize(fc, stoi)
    s_w2_all = tokenize(w2[: int(len(w2) * (t_w2 / 3743067)) + 1], stoi)
    s_gr_all = tokenize(gr, stoi)

    def tile(s, target):
        reps = (target + len(s) - 1) // len(s)
        return s.repeat(reps)[:target]

    streams = [tile(s_fc, t_fc), tile(s_w2_all, t_w2), tile(s_gr_all, t_gr)]
    probs = [0.35, 0.50, 0.15]
    print(f"mixed totals: {sum(len(s) for s in streams)} tokens, "
          f"probs={probs}", flush=True)

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert cfg["vocab_size"] == len(vocab) and cfg["n_layer"] == 4 \
        and cfg["n_head"] == 4 and cfg["n_embd"] == 128, f"cfg drift: {cfg}"
    model = GPT(**cfg)
    model.load_state_dict(ck["state"])

    # freeze dnd1's authored deltas: they stay exactly as authored
    frozen_params = 0
    for name, p in model.named_parameters():
        if name in FROZEN:
            p.requires_grad_(False)
            frozen_params += p.numel()
    print(f"frozen {len(FROZEN)} param groups ({frozen_params} params); "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}",
          flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr)
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
    with open(a.out.replace(".pt", ".json"), "w") as f:
        json.dump({"init": a.init, "lr": a.lr, "steps": a.steps,
                   "frozen": sorted(FROZEN), "mix_probs": probs,
                   "final_loss": float(loss.item())}, f, indent=1)
    print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
