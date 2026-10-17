"""Train RMTGPT with episode-grouped (truncated) BPTT.

Corpus priority: lm/traces_fc.jsonl (Track 1, rows carry an "episode" field)
else lm/traces_bb.jsonl (fallback: rows grouped into fake episodes of 8
consecutive rows -- smoke-test quality only, see docstring note).

For each episode: mem starts at mem_init; for each row: render+encode,
truncate row to <= ROW_TOK text tokens, forward, CE loss on text, keep the
mem graph ALIVE across rows (no detach) so the WRITE path gets gradient;
optimizer step per episode (optionally every --step-every episodes).

Warm-start: transformer weights from lm/router_v5.pt if present else
lm/router_v3.pt (RMTGPT.warm_start shifts pretrained wpe rows; mem params
are fresh).

Usage: python train_rmt.py [--episodes N] [--epochs E] [--trunc K] [--out router_rmt]
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_rmt import RMTGPT
from tok import render_trace, load_vocab, encode

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LM = HERE   # train_rmt.py lives in model/lm/ alongside the traces + ckpts
ROW_TOK = 184          # 8 mem + 184 text = 192 <= block_size 192
MAX_EPS = 400
N_MEM = 8
# FORBIDDEN transfer games: measure-only, never train on. Episode ids look
# like "<game>_s<seed>_lvl<l>"; abort if any of these prefixes appear.
FORBIDDEN = {"ar25", "cd82", "lp85", "m0r0", "r11l", "sb26", "sp80", "su15",
             "tn36", "tr87", "vc33", "wa30"}


def load_episodes():
    """Return (episodes, source_note). episodes = list of list[dict]."""
    fc = os.path.join(LM, "traces_fc.jsonl")
    bb = os.path.join(LM, "traces_bb.jsonl")
    if os.path.exists(fc) and os.path.getsize(fc) > 0:
        groups = {}
        order = []
        n = 0
        with open(fc) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ep = d.get("episode") or "ep?"
                if ep not in groups:
                    groups[ep] = []
                    order.append(ep)
                groups[ep].append(d)
                n += 1
        note = f"traces_fc.jsonl ({n} rows)"
    else:
        rows = []
        with open(bb) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        # rows in traces_bb are NOT in episode order; fake episodes of 8
        # consecutive rows are smoke-test quality only.
        groups, order = {}, []
        for i in range(0, len(rows), 8):
            ep = f"fake{i // 8}"
            groups[ep] = rows[i:i + 8]
            order.append(ep)
        note = (f"traces_bb.jsonl FAKE episodes of 8 consecutive rows "
                f"({len(rows)} rows; rows are shuffled, NOT true episodes)")
    episodes = [groups[ep] for ep in order if 4 <= len(groups[ep]) <= 250]
    episodes = episodes[:MAX_EPS]
    # anti-contamination: refuse to train on transfer games
    bad = {ep.split("_")[0] for ep in order} & FORBIDDEN
    assert not bad, f"FORBIDDEN transfer games in corpus: {sorted(bad)}"
    return episodes, note


def encode_row(d, stoi):
    ids = encode(render_trace(d), stoi)
    if len(ids) > ROW_TOK + 1:
        ids = ids[:ROW_TOK + 1]          # truncate tail
    if len(ids) < 2:
        return None
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0,
                    help="cap episodes (0 = all loaded)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--trunc", type=int, default=20,
                    help="0 = full-episode BPTT; K>0 = detach mem every K rows "
                         "(no-op for episodes shorter than K)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="router_rmt")
    ap.add_argument("--warm", default="router_v3.pt",
                    help="warm-start checkpoint (default router_v3.pt: v5 "
                         "regressed on dev per 2026-10-16 coordinator note)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    episodes, note = load_episodes()
    if a.episodes:
        episodes = episodes[:a.episodes]
    print(f"corpus: {note}", flush=True)
    print(f"episodes: {len(episodes)} "
          f"(rows {sum(len(e) for e in episodes)})", flush=True)
    if not episodes:
        print("no usable episodes; exiting (train on nothing is not a run)",
              flush=True)
        return

    vocab, stoi = load_vocab(os.path.join(LM, "vocab.json"))
    cfg = {"vocab_size": len(vocab), "n_layer": 4, "n_head": 4,
           "n_embd": 128, "block_size": 192, "dropout": 0.0, "n_mem": N_MEM}
    model = RMTGPT(**cfg)

    warm_src = a.warm
    assert os.path.exists(os.path.join(LM, warm_src)), \
        f"warm-start checkpoint missing: {warm_src}"
    ck = torch.load(os.path.join(LM, warm_src), map_location="cpu",
                    weights_only=False)
    model.warm_start(ck["state"])
    print(f"warm-started transformer weights from {warm_src}; "
          f"mem params fresh. params: {model.param_count()}", flush=True)

    # encode once
    enc = []
    for ep in episodes:
        rows = [encode_row(d, stoi) for d in ep]
        rows = [r for r in rows if r is not None]
        if len(rows) >= 4:
            enc.append(rows)
    episodes = enc
    print(f"usable episodes after encoding: {len(episodes)}", flush=True)

    n_val = max(1, len(episodes) // 10)
    train_eps, val_eps = episodes[:-n_val], episodes[-n_val:]
    print(f"train {len(train_eps)} / val {len(val_eps)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    model.train()
    t0 = time.time()
    for epoch in range(a.epochs):
        random.shuffle(train_eps)
        tot, cnt = 0.0, 0
        for ei, ep in enumerate(train_eps):
            opt.zero_grad()
            mem = None
            losses = []
            for ri, (x, y) in enumerate(ep):
                _, loss, mem_out = model(x, mem, y)
                if a.trunc and ri and ri % a.trunc == 0:
                    mem_out = mem_out.detach()   # truncated BPTT boundary
                mem = mem_out
                losses.append(loss)
            # ONE backward per episode: full BPTT through the memory chain
            # (per-row backward would free earlier steps' graphs).
            ep_loss = torch.stack(losses).mean()
            ep_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += ep_loss.item()
            cnt += 1
            if (ei + 1) % 20 == 0:
                print(f"epoch {epoch+1} ep {ei+1}/{len(train_eps)} "
                      f"mean_row_loss {tot/cnt:.4f} {time.time()-t0:.0f}s",
                      flush=True)
        # val loss (no grad, memory still carried per episode)
        model.eval()
        vtot, vcnt = 0.0, 0
        with torch.no_grad():
            for ep in val_eps:
                mem = None
                for x, y in ep:
                    _, loss, mem = model(x, mem, y)
                    vtot += loss.item()
                    vcnt += 1
        model.train()
        tr = f"{tot/cnt:.4f}" if cnt else "n/a"
        print(f"epoch {epoch+1} DONE train_loss {tr} "
              f"val_loss {vtot/max(1,vcnt):.4f} {time.time()-t0:.0f}s",
              flush=True)

    out_pt = os.path.join(LM, a.out + ".pt")
    torch.save({"cfg": cfg, "state": model.state_dict()}, out_pt)
    meta = {"params": model.param_count(), "episodes": len(train_eps),
            "epochs": a.epochs, "trunc": a.trunc, "lr": a.lr,
            "warm_start": warm_src, "corpus": note,
            "final_train_loss": tot / cnt if cnt else None,
            "final_val_loss": vtot / max(1, vcnt)}
    json.dump(meta, open(os.path.join(LM, a.out + ".json"), "w"), indent=1)
    print(f"saved {out_pt}", flush=True)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
