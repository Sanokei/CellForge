#!/usr/bin/env python3
"""beta-g2 v6: RMT + MEMORY-FORCING, transformer FROZEN.

v5 post-mortem: v5 (full CE, probe 0.1, mem-reg, frozen dnd1) STILL 0.00%.
Every RMT training attempt (gen-1, v1-v5) breaks dev-6; the untrained
warm-start gets 0.46%. The transformer blocks cannot survive RMT
finetuning on this data -- the text path is too brittle.

v6: FREEZE THE ENTIRE TRANSFORMER. Train ONLY mem_init (the learnable
initial memory) + the isolated probe. The text path becomes BIT-IDENTICAL
to the untrained warm-start (dev-6 ~0.46% preserved by construction), while
mem_init adapts via the probe to seed a more predictive memory. The frozen
text path already attends heavily to memory (0.57 in block 0), so a more
informative mem_init can actually be USED without any text-path changes.

Loss = 0.5*probe(mem_in -> obs) + 0.05*mem_reg + 0.2*text_CE(anchor).
Corpus: fc discovery chains only (probe needs chains). 200 steps.
"""
import argparse
import json
import os
import random
import sys
import time

LM = "/srv/arc3-claude/v6run/model/lm"
sys.path.insert(0, LM)
from gpt_rmt import RMTGPT
from tok import load_vocab, encode

import torch
import torch.nn.functional as F

ROW_TOK = 184
N_MEM = 8
SEG = 56
FORBIDDEN = {"ar25", "cd82", "lp85", "m0r0", "r11l", "sb26", "sp80", "su15",
             "tn36", "tr87", "vc33", "wa30"}


def game_prefix(epid):
    return epid.split("_")[0]


def render_row(d):
    return (f"STATE {d['state']}\nTHINK {d['think']}\n"
            f"CALL {json.dumps(d['call'], separators=(',', ':'))}\n")


def encode_row(d, stoi):
    ids = encode(render_row(d), stoi)
    if len(ids) > ROW_TOK + 1:
        ids = ids[:ROW_TOK + 1]
    if len(ids) < 2:
        return None
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    return x, y


def encode_obs(d, stoi, k=N_MEM):
    ids = encode(d.get("obs", ""), stoi)[:k]
    pad = stoi["<pad>"]
    ids = ids + [pad] * (k - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def chunk(ep, seg=SEG):
    return [ep[i:i + seg] for i in range(0, len(ep), seg)]


def load_fc():
    groups, order = {}, []
    with open(os.path.join(LM, "traces_fc.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ep = d.get("episode") or "ep?"
            assert not ({game_prefix(ep)} & FORBIDDEN), "FORBIDDEN in fc"
            if ep not in groups:
                groups[ep] = []
                order.append(ep)
            groups[ep].append(d)
    eps = []
    for ep in order:
        rows = groups[ep]
        if 4 <= len(rows) <= 4000:
            eps.extend(chunk(rows))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--force-w", type=float, default=0.5)
    ap.add_argument("--mreg-w", type=float, default=0.05)
    ap.add_argument("--text-w", type=float, default=0.2)
    ap.add_argument("--warm", default="router_v3+dnd1.pt")
    ap.add_argument("--out", default="router_beta_g2f")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    assert a.lr <= 1e-4 and a.force_w <= 1.0

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    fc_eps = load_fc()
    print(f"fc chains: {len(fc_eps)} segs, {a.steps} steps, lr={a.lr}",
          flush=True)

    vocab, stoi = load_vocab(os.path.join(LM, "vocab.json"))
    assert len(vocab) == 2209
    pad_id = stoi["<pad>"]
    cfg = {"vocab_size": len(vocab), "n_layer": 4, "n_head": 4,
           "n_embd": 128, "block_size": 192, "dropout": 0.0, "n_mem": N_MEM}
    model = RMTGPT(**cfg)
    ck = torch.load(os.path.join(LM, a.warm), map_location="cpu",
                    weights_only=False)
    model.warm_start(ck["state"])

    # v6: freeze EVERYTHING except mem_init.
    for n, p in model.named_parameters():
        if n != "mem_init":
            p.requires_grad_(False)
    print("frozen: all except mem_init", flush=True)
    mem_init_0 = model.mem_init.detach().clone()

    probe = torch.nn.Sequential(torch.nn.LayerNorm(128),
                                torch.nn.Linear(128, len(vocab), bias=False))
    torch.nn.init.normal_(probe[1].weight, std=0.02)

    # encode
    enc = []
    for seg in fc_eps:
        rows = [encode_row(d, stoi) for d in seg]
        rows = [r for r in rows if r is not None]
        if len(rows) < 4:
            continue
        obs = [encode_obs(d, stoi) for d in seg]
        enc.append((rows, obs))
    print(f"usable segs: {len(enc)}", flush=True)

    opt = torch.optim.AdamW([model.mem_init] + list(probe.parameters()),
                            lr=a.lr)
    model.train()
    t0 = time.time()
    step = 0
    tot, ftot, mt, ct, cnt = 0.0, 0.0, 0.0, 0.0, 0
    while step < a.steps:
        random.shuffle(enc)
        for rows, obs in enc:
            if step >= a.steps:
                break
            opt.zero_grad()
            mem = None
            tl_sum, fl_sum, mr_sum = 0.0, 0.0, 0.0
            n_fl = 0
            for ri, (x, y) in enumerate(rows):
                mem_in = mem
                logits, _, mem_out = model(x, mem_in)
                # text CE anchor (frozen transformer; keeps mem_init compatible)
                tl_sum = tl_sum + F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                mr_sum = mr_sum + mem_out.pow(2).mean()
                if ri >= 1:
                    pl = probe(mem_in)
                    fl_sum = fl_sum + F.cross_entropy(
                        pl.reshape(-1, pl.size(-1)), obs[ri].reshape(-1),
                        ignore_index=pad_id)
                    n_fl += 1
                mem = mem_out
            n_rows = len(rows)
            loss = (a.text_w * tl_sum / n_rows +
                    a.mreg_w * mr_sum / n_rows +
                    a.force_w * fl_sum / max(1, n_fl))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [model.mem_init] + list(probe.parameters()), 1.0)
            opt.step()
            tot += (tl_sum / n_rows).item()
            ftot += (fl_sum / max(1, n_fl)).item()
            mt += (mr_sum / n_rows).item()
            cnt += 1
            step += 1
            if step % 25 == 0:
                drift = (model.mem_init.detach() - mem_init_0).norm().item()
                print(f"step {step}/{a.steps} text {tot/cnt:.4f} "
                      f"force {ftot/cnt:.4f} mreg {mt/cnt:.4f} "
                      f"mem_init_drift {drift:.4f} {time.time()-t0:.0f}s",
                      flush=True)
    print(f"DONE text {tot/max(1,cnt):.4f} force {ftot/max(1,cnt):.4f} "
          f"mreg {mt/max(1,cnt):.4f}", flush=True)

    outdir = "/srv/cellforge-arena/researchers/beta/gen2"
    out_pt = os.path.join(outdir, a.out + ".pt")
    torch.save({"cfg": cfg, "state": model.state_dict()}, out_pt)
    meta = {"recipe": "beta-g2 v6: transformer frozen, mem_init + probe only",
            "steps": step, "lr": a.lr, "force_w": a.force_w,
            "mreg_w": a.mreg_w, "text_w": a.text_w, "warm_start": a.warm,
            "seed": a.seed, "params": model.param_count(),
            "mem_init_drift": (model.mem_init.detach() - mem_init_0).norm().item(),
            "final_train_text": tot / max(1, cnt),
            "final_train_force": ftot / max(1, cnt)}
    json.dump(meta, open(os.path.join(outdir, a.out + ".json"), "w"), indent=1)
    print(f"saved {out_pt}", flush=True)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
