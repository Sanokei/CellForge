#!/usr/bin/env python3
"""finetune_gen2kl.py — ALPHA-G2 KL-anchored rehearsal finetune.

Anti-forgetting via KL anchor: on ANSWER batches (w2/grammar), loss =
LM + kl_lambda * KL(p_theta || p_base_dnd1). On DISCOVERY batches (fc),
pure LM loss (allowed to learn new behavior). dnd1's 4 authored param groups
stay frozen. LR capped at 1e-4.
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
import torch.nn.functional as F

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
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=5.0)
    ap.add_argument("--fc_frac", type=float, default=0.35)
    ap.add_argument("--init", default=os.path.join(WORK, "base_dnd1.pt"))
    ap.add_argument("--out", default=os.path.join(WORK, "alpha_gen2kl.pt"))
    ap.add_argument("--seed", type=int, default=7)
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

    t_fc = 714385
    T = int(t_fc / a.fc_frac)
    t_w2 = int(0.50 * T)
    t_gr = T - t_fc - t_w2

    s_fc = tokenize(fc, stoi)
    s_w2 = tokenize(w2[: int(len(w2) * (t_w2 / 3743067)) + 1], stoi)
    s_gr = tokenize(gr, stoi)

    def tile(s, target):
        reps = (target + len(s) - 1) // len(s)
        return s.repeat(reps)[:target]

    streams = [tile(s_fc, t_fc), tile(s_w2, t_w2), tile(s_gr, t_gr)]
    # is_answer flags: fc=False, w2=True, grammar=True
    is_ans = [False, True, True]
    probs = [len(s) / sum(len(s) for s in streams) for s in streams]
    print(f"tokens={sum(len(s) for s in streams)} "
          f"probs={[f'{p:.3f}' for p in probs]} kl={a.kl}", flush=True)

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = GPT(**cfg)
    model.load_state_dict(ck["state"])
    base = GPT(**cfg)
    base.load_state_dict(ck["state"])
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
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
        xs, ys, ans = [], [], []
        for _ in range(a.batch):
            si = torch.multinomial(torch.tensor(probs), 1,
                                   generator=rng).item()
            s = streams[si]
            i = torch.randint(0, len(s) - a.block - 1, (1,),
                              generator=rng).item()
            xs.append(s[i:i + a.block])
            ys.append(s[i + 1:i + a.block + 1])
            ans.append(is_ans[si])
        x = torch.stack(xs)
        y = torch.stack(ys)
        ans = torch.tensor(ans, dtype=torch.bool)
        opt.zero_grad()
        logits, loss_lm = model(x, y)
        loss = loss_lm
        if ans.any() and a.kl > 0:
            with torch.no_grad():
                base_logits, _ = base(x[ans])
            cur = logits[ans]
            kl = F.kl_div(
                F.log_softmax(cur, dim=-1),
                F.softmax(base_logits, dim=-1),
                reduction="batchmean",
            )
            loss = loss + a.kl * kl
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"step {step+1}/{a.steps} loss {loss.item():.4f} "
                  f"(lm {loss_lm.item():.4f}) {time.time()-t0:.0f}s", flush=True)

    torch.save({"cfg": cfg, "state": model.state_dict()}, a.out)
    with open(a.out.replace(".pt", ".json"), "w") as f:
        json.dump({"init": a.init, "lr": a.lr, "steps": a.steps,
                   "kl": a.kl, "fc_frac": a.fc_frac,
                   "frozen": sorted(FROZEN),
                   "final_loss": float(loss.item())}, f, indent=1)
    print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
