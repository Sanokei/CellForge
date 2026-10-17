#!/usr/bin/env python3
"""finetune_gen2c.py — ALPHA-G2 ablation finetunes.

Parametrized: --fc_frac (0.0 = answers-only control), --lr, --steps.
Always freezes the 4 dnd1-touched param groups. Inits from base_dnd1.
"""
import argparse
import json
import os
import random
import sys
import time

LMDIR = "/srv/arc3-claude/v6run/model/lm"
sys.path.insert(0, LMDIR)
from tok import load_vocab, encode, render_trace  # noqa: E402
from gpt import GPT  # noqa: E402

import torch

WORK = "/srv/cellforge-arena/researchers/alpha/gen2"

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
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--fc_frac", type=float, default=0.0)
    ap.add_argument("--init", default=os.path.join(WORK, "base_dnd1.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2)
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

    fc = load_docs("traces_fc.jsonl")
    w2 = load_docs("traces_w2.jsonl")
    gr = load_docs("traces_grammar_x100.jsonl")

    rnd = random.Random(20261016)
    rnd.shuffle(w2)

    # anchor: full fc (714,385 tok) when fc_frac > 0; else answers-only budget
    if a.fc_frac > 0:
        t_fc = 714385
        T = int(t_fc / a.fc_frac)
    else:
        T = 1200000  # answers-only corpus budget
        t_fc = 0
    t_w2 = int(0.85 * (T - t_fc))
    t_gr = T - t_fc - t_w2

    s_w2 = tokenize(w2[: int(len(w2) * (t_w2 / 3743067)) + 1], stoi)
    s_gr = tokenize(gr, stoi)

    def tile(s, target):
        reps = (target + len(s) - 1) // len(s)
        return s.repeat(reps)[:target]

    if a.fc_frac > 0:
        s_fc = tokenize(fc, stoi)
        streams = [tile(s_fc, t_fc), tile(s_w2, t_w2), tile(s_gr, t_gr)]
        probs = [t_fc / T, t_w2 / T, t_gr / T]
    else:
        streams = [tile(s_w2, t_w2), tile(s_gr, t_gr)]
        probs = [t_w2 / T, t_gr / T]
    print(f"fc_frac={a.fc_frac} tokens={sum(len(s) for s in streams)} "
          f"probs={[f'{p:.3f}' for p in probs]}", flush=True)

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = GPT(**cfg)
    model.load_state_dict(ck["state"])
    for name, p in model.named_parameters():
        if name in FROZEN:
            p.requires_grad_(False)
    print(f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}",
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
        if (step + 1) % 50 == 0 or step + 1 == a.steps:
            print(f"step {step+1}/{a.steps} loss {loss.item():.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    torch.save({"cfg": cfg, "state": model.state_dict()}, a.out)
    with open(a.out.replace(".pt", ".json"), "w") as f:
        json.dump({"init": a.init, "lr": a.lr, "steps": a.steps,
                   "fc_frac": a.fc_frac, "frozen": sorted(FROZEN),
                   "final_loss": float(loss.item())}, f, indent=1)
    print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
