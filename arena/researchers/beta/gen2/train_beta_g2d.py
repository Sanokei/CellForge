#!/usr/bin/env python3
"""beta-g2 v4: RMT + MEMORY-FORCING, dnd1 machinery FROZEN.

v3 post-mortem: v3's finetuning (394 steps, lr 1e-4, probe 0.5x) WASHED OUT
dnd1's decisiveness machinery -- dev-6 0.00% vs 0.46% for the UNTRAINED
warm-start from the same checkpoint (which solved s5i5 L1). dnd1's deltas
live in lm_head.weight + 3 block tensors; v4 freezes all four. The memory
write/read path learns through every other parameter.

Kept from v3: eval-format rows (no OBS in input -- matches the runner's
memory-update format), masked CE on discovery rows (think+call from
STATE+MEMORY = read pressure), isolated train-time probe (mem_in -> OBS,
write pressure, discarded before save). Probe weight lowered 0.5 -> 0.25
to further protect the answer machinery.
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
from tok import load_vocab, encode, tokenize

import torch
import torch.nn.functional as F

ROW_TOK = 184
N_MEM = 8
SEG = 56
FORCE_W = 0.5
FORBIDDEN = {"ar25", "cd82", "lp85", "m0r0", "r11l", "sb26", "sp80", "su15",
             "tn36", "tr87", "vc33", "wa30"}


def game_prefix(epid):
    return epid.split("_")[0]


def render_row(d):
    """Eval-format triple: NO obs line (matches runner hist entries)."""
    return (f"STATE {d['state']}\nTHINK {d['think']}\n"
            f"CALL {json.dumps(d['call'], separators=(',', ':'))}\n")


def render_prefix(d):
    """Eval prompt prefix: exactly what the runner feeds the model."""
    return f"STATE {d['state']}\nTHINK "


def encode_row_masked(d, stoi, discovery):
    """Returns (x, y, loss_mask) with x/y truncated to ROW_TOK.
    discovery=True -> CE only on think+call positions (read pressure)."""
    ids = encode(render_row(d), stoi)
    if len(ids) > ROW_TOK + 1:
        ids = ids[:ROW_TOK + 1]
    if len(ids) < 2:
        return None
    pre = len(encode(render_prefix(d), stoi))  # think content starts here
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    if discovery:
        # target position i (predicting ids[i+1]) counts iff ids[i+1] is at
        # or after the think content start.
        keep = torch.tensor([ (i + 1) >= pre for i in range(len(ids) - 1) ],
                             dtype=torch.bool)
        if not keep.any():
            return None
    else:
        keep = torch.ones(len(ids) - 1, dtype=torch.bool)
    return x, y, keep


def encode_obs(d, stoi, k=N_MEM):
    ids = encode(d.get("obs", ""), stoi)[:k]
    pad = stoi["<pad>"]
    ids = ids + [pad] * (k - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def chunk(ep, seg=SEG):
    return [ep[i:i + seg] for i in range(0, len(ep), seg)]


def load_fc():
    groups, order = {}, []
    n = 0
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
            n += 1
    eps = []
    for ep in order:
        rows = groups[ep]
        if not (4 <= len(rows) <= 4000):
            continue
        for seg in chunk(rows):
            eps.append({"rows": seg, "discovery": True})
    return eps, n


def load_chunks(path, seg, n_chunks, seed):
    rows = []
    with open(os.path.join(LM, path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ep = d.get("episode") or ""
            assert not ({game_prefix(ep)} & FORBIDDEN), f"FORBIDDEN in {path}"
            rows.append(d)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_chunks = min(n_chunks, len(rows) // seg)
    return [{"rows": rows[i * seg:(i + 1) * seg], "discovery": False}
            for i in range(n_chunks)], len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--w2-chunks", type=int, default=64)
    ap.add_argument("--gram-chunks", type=int, default=28)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--force-w", type=float, default=0.25)
    ap.add_argument("--warm", default="router_v3+dnd1.pt")
    ap.add_argument("--out", default="router_beta_g2d")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    assert a.lr <= 1e-4 and a.force_w <= 1.0

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    fc_eps, fc_rows = load_fc()
    w2_eps, w2_rows = load_chunks("traces_w2.jsonl", 16, a.w2_chunks, a.seed + 1)
    g_eps, g_rows = load_chunks("traces_grammar_x100.jsonl", 12, a.gram_chunks,
                                a.seed + 2)
    corpus = fc_eps + w2_eps + g_eps
    print(f"corpus: fc {len(fc_eps)} discovery segs ({fc_rows} rows) + "
          f"w2 {len(w2_eps)}x16 + gram {len(g_eps)}x12; "
          f"{len(corpus)*a.epochs} steps lr={a.lr} force_w={a.force_w}",
          flush=True)

    vocab, stoi = load_vocab(os.path.join(LM, "vocab.json"))
    assert len(vocab) == 2209
    pad_id = stoi["<pad>"]
    cfg = {"vocab_size": len(vocab), "n_layer": 4, "n_head": 4,
           "n_embd": 128, "block_size": 192, "dropout": 0.0, "n_mem": N_MEM}
    model = RMTGPT(**cfg)
    ck = torch.load(os.path.join(LM, a.warm), map_location="cpu",
                    weights_only=False)
    skipped = model.warm_start(ck["state"])
    print(f"warm-start {a.warm} skipped={skipped} params={model.param_count()}",
          flush=True)

    # v4: FREEZE dnd1's authored tensors. v3's finetuning washed out the
    # dnd1 decisiveness machinery (dev-6 0.00% vs 0.46% for the untrained
    # warm-start). The memory write path learns through all OTHER params.
    FROZEN_DND1 = {"lm_head.weight", "blocks.2.attn.c_attn.weight",
                   "blocks.3.attn.c_attn.weight", "blocks.3.mlp.c_proj.weight"}
    for n, p in model.named_parameters():
        if n in FROZEN_DND1:
            p.requires_grad_(False)
    print(f"frozen dnd1 tensors: {sorted(FROZEN_DND1)}", flush=True)

    # isolated train-time probe (discarded before save)
    probe = torch.nn.Sequential(torch.nn.LayerNorm(128),
                                torch.nn.Linear(128, len(vocab), bias=False))
    torch.nn.init.normal_(probe[1].weight, std=0.02)

    enc = []
    for ep in corpus:
        rows = [encode_row_masked(d, stoi, ep["discovery"]) for d in ep["rows"]]
        rows = [r for r in rows if r is not None]
        if len(rows) < 4:
            continue
        obs = [encode_obs(d, stoi) for d in ep["rows"]] if ep["discovery"] \
            else None
        enc.append({"rows": rows, "obs": obs, "discovery": ep["discovery"]})
    corpus = enc
    print(f"usable episodes: {len(corpus)}", flush=True)

    n_val = max(1, len(corpus) // 10)
    train_eps, val_eps = corpus[:-n_val], corpus[-n_val:]
    print(f"train {len(train_eps)} / val {len(val_eps)}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable + list(probe.parameters()), lr=a.lr)
    n_tr = sum(p.numel() for p in trainable)
    print(f"trainable model params: {n_tr} / {model.param_count()}",
          flush=True)
    model.train()
    t0 = time.time()
    for epoch in range(a.epochs):
        random.shuffle(train_eps)
        tot, ftot, dt, cnt = 0.0, 0.0, 0.0, 0
        for ei, ep in enumerate(train_eps):
            opt.zero_grad()
            mem = None
            text_losses, force_losses, dec_losses = [], [], []
            for ri, (x, y, keep) in enumerate(ep["rows"]):
                mem_in = mem
                logits, _, mem_out = model(x, mem_in)
                # text CE on kept positions (discovery: think+call only)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     y.reshape(-1), reduction="none")
                keep_f = keep.to(ce.device)
                tl = ce[keep_f].mean()
                text_losses.append(tl)
                if ep["discovery"]:
                    dec_losses.append(tl)
                    if ri >= 1:
                        pl = probe(mem_in)
                        fl = F.cross_entropy(
                            pl.reshape(-1, pl.size(-1)),
                            ep["obs"][ri].reshape(-1), ignore_index=pad_id)
                        force_losses.append(fl)
                mem = mem_out
            ep_text = torch.stack(text_losses).mean()
            ep_loss = ep_text
            ef = torch.tensor(0.0)
            if force_losses:
                ef = torch.stack(force_losses).mean()
                ep_loss = ep_loss + a.force_w * ef
            ep_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable + list(probe.parameters()), 1.0)
            opt.step()
            tot += ep_text.item()
            ftot += ef.item()
            dt += (torch.stack(dec_losses).mean().item()
                   if dec_losses else 0.0)
            cnt += 1
            if (ei + 1) % 25 == 0:
                print(f"epoch {epoch+1} ep {ei+1}/{len(train_eps)} "
                      f"text {tot/cnt:.4f} dec {dt/cnt:.4f} force {ftot/cnt:.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
        model.eval()
        vtot, vcnt = 0.0, 0
        with torch.no_grad():
            for ep in val_eps:
                mem = None
                for x, y, keep in ep["rows"]:
                    logits, _, mem = model(x, mem)
                    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                         y.reshape(-1), reduction="none")
                    vtot += ce[keep].mean().item()
                    vcnt += 1
        model.train()
        print(f"epoch {epoch+1} DONE text {tot/max(1,cnt):.4f} "
              f"dec {dt/max(1,cnt):.4f} force {ftot/max(1,cnt):.4f} "
              f"val {vtot/max(1,vcnt):.4f} {time.time()-t0:.0f}s", flush=True)

    outdir = "/srv/cellforge-arena/researchers/beta/gen2"
    out_pt = os.path.join(outdir, a.out + ".pt")
    torch.save({"cfg": cfg, "state": model.state_dict()}, out_pt)
    meta = {"recipe": "beta-g2 v3: eval-format rows, masked CE (read pressure)"
                      " + isolated probe (write pressure)",
            "steps": len(train_eps) * a.epochs, "epochs": a.epochs,
            "lr": a.lr, "force_w": a.force_w, "warm_start": a.warm,
            "seed": a.seed, "params": model.param_count(),
            "final_train_text": tot / max(1, cnt),
            "final_train_force": ftot / max(1, cnt),
            "final_val": vtot / max(1, vcnt)}
    json.dump(meta, open(os.path.join(outdir, a.out + ".json"), "w"), indent=1)
    print(f"saved {out_pt}", flush=True)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
