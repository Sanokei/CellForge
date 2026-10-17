#!/usr/bin/env python3
"""beta-g2: RMT + MEMORY-FORCING loss on a mixed corpus.

"Memory That Predicts". The gen-1 RMT attempt failed because memory had no
pressure to be used (memory dependence 0.0). The fix: an auxiliary loss that
forces the model to predict the CURRENT row's OBS tokens FROM the memory
tokens entering that row, through the existing lm_head (no new parameters,
so the checkpoint stays a pure fixed-arch RMT {cfg, state}).

Mechanism: at row r>=1 of a true-chain episode, mem_in_r (= mem_out_{r-1},
carrying the write path's autograd graph) is passed through ln_f + lm_head
and trained with CE against the first 8 tokens of row r's OBS string.
Gradient flows into the write path of row r-1: to minimize the loss, the
model must WRITE a predictive world state into memory and READ it back
to anticipate the next observation. Memory becomes load-bearing by
construction: zeroing it at eval destroys the observation-prediction
channel that text-only CE never built.

Hard constraints (from the gen-2 charter + task):
- LR <= 1e-4 (full-LR 3e-4 caused catastrophic forgetting last night).
- Mixed corpus mandatory: discovery chains (fc, forcing ON) + answer
  rehearsal (w2, grammar, forcing OFF / text-CE only to prevent forgetting).
- World-model pressure lives ONLY in the memory probe (weight 0.5 <= 1x);
  there is no OBS-prediction loss on the text path (3x on text distorted
  call generation last night -- never again).
- Transfer-12 games are never opened/trained on: FORBIDDEN prefixes are
  asserted against every episode id and every corpus row.

Corpus: fc true episodes (chunked to <= SEG rows for safe full-episode
BPTT) + w2 16-row chunks + grammar 12-row chunks, shuffled each epoch.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter

LM = "/srv/arc3-claude/v6run/model/lm"
sys.path.insert(0, LM)
from gpt_rmt import RMTGPT
from tok import load_vocab, encode

import torch
import torch.nn.functional as F

ROW_TOK = 184            # 8 mem + 184 text = 192 <= block_size 192
N_MEM = 8
SEG = 56                 # max rows per episode-segment (full BPTT budget)
FORCE_W = 0.5            # memory-forcing loss weight (<= 1x per charter)
FORBIDDEN = {"ar25", "cd82", "lp85", "m0r0", "r11l", "sb26", "sp80", "su15",
             "tn36", "tr87", "vc33", "wa30"}
FC_EP_FILTER = (4, 4000)


def game_prefix(epid):
    return epid.split("_")[0]


def chunk(ep, seg=SEG):
    return [ep[i:i + seg] for i in range(0, len(ep), seg)]


def load_fc():
    """True-chain discovery episodes from traces_fc.jsonl."""
    groups, order = {}, []
    n = 0
    with open(os.path.join(LM, "traces_fc.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ep = d.get("episode") or "ep?"
            bad = {game_prefix(ep)} & FORBIDDEN
            assert not bad, f"FORBIDDEN transfer game in fc corpus: {sorted(bad)}"
            if ep not in groups:
                groups[ep] = []
                order.append(ep)
            groups[ep].append(d)
            n += 1
    eps = []
    for ep in order:
        rows = groups[ep]
        if not (FC_EP_FILTER[0] <= len(rows) <= FC_EP_FILTER[1]):
            continue
        for seg in chunk(rows):
            eps.append({"rows": seg, "force": True, "src": "fc"})
    return eps, n


def load_chunks(path, seg, n_chunks, src, seed):
    """Row-level rehearsal corpora (no episode structure): shuffle once,
    cut into fixed chunks. Forcing OFF -- pure rehearsal against forgetting."""
    rows = []
    with open(os.path.join(LM, path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ep = d.get("episode") or ""
            bad = {game_prefix(ep)} & FORBIDDEN
            assert not bad, f"FORBIDDEN transfer game in {path}: {sorted(bad)}"
            rows.append(d)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_chunks = min(n_chunks, len(rows) // seg)
    eps = [{"rows": rows[i * seg:(i + 1) * seg], "force": False, "src": src}
           for i in range(n_chunks)]
    return eps, len(rows)


def encode_row(d, stoi):
    ids = encode(f"STATE {d['state']}\nTHINK {d['think']}\n"
                 f"CALL {json.dumps(d['call'], separators=(',', ':'))}\n"
                 f"OBS {d['obs']}\n", stoi)
    if len(ids) > ROW_TOK + 1:
        ids = ids[:ROW_TOK + 1]
    if len(ids) < 2:
        return None
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    return x, y


def encode_obs(d, stoi, k=N_MEM):
    """First k token ids of the OBS string, padded with <pad>."""
    ids = encode(d.get("obs", ""), stoi)[:k]
    pad = stoi["<pad>"]
    ids = ids + [pad] * (k - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--w2-chunks", type=int, default=64,
                    help="16-row w2 rehearsal chunks per epoch")
    ap.add_argument("--gram-chunks", type=int, default=28,
                    help="12-row grammar rehearsal chunks per epoch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--force-w", type=float, default=FORCE_W)
    ap.add_argument("--warm", default="router_v3+dnd1.pt",
                    help="warm-start ckpt (dnd1 deltas: decisiveness, parse)")
    ap.add_argument("--out", default="router_beta_g2")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    assert a.lr <= 1e-4, f"LR hard cap 1e-4 violated: {a.lr}"
    assert a.force_w <= 1.0, f"forcing weight cap 1x violated: {a.force_w}"

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    fc_eps, fc_rows = load_fc()
    w2_eps, w2_rows = load_chunks("traces_w2.jsonl", 16, a.w2_chunks, "w2", a.seed + 1)
    g_eps, g_rows = load_chunks("traces_grammar_x100.jsonl", 12, a.gram_chunks,
                                "gram", a.seed + 2)
    corpus = fc_eps + w2_eps + g_eps
    print(f"corpus: fc {len(fc_eps)} segs ({fc_rows} rows) | "
          f"w2 {len(w2_eps)} chunks ({w2_rows} rows avail) | "
          f"gram {len(g_eps)} chunks ({g_rows} rows avail)", flush=True)
    print(f"{len(corpus)} episodes/epoch x {a.epochs} epochs = "
          f"{len(corpus) * a.epochs} steps; lr={a.lr} force_w={a.force_w}",
          flush=True)

    vocab, stoi = load_vocab(os.path.join(LM, "vocab.json"))
    assert len(vocab) == 2209, f"vocab drift: {len(vocab)}"
    pad_id = stoi["<pad>"]
    cfg = {"vocab_size": len(vocab), "n_layer": 4, "n_head": 4,
           "n_embd": 128, "block_size": 192, "dropout": 0.0, "n_mem": N_MEM}
    model = RMTGPT(**cfg)

    warm_path = os.path.join(LM, a.warm)
    assert os.path.exists(warm_path), f"warm-start ckpt missing: {warm_path}"
    ck = torch.load(warm_path, map_location="cpu", weights_only=False)
    skipped = model.warm_start(ck["state"])
    print(f"warm-started from {a.warm} (skipped={skipped}); "
          f"mem_init/wpe-mem fresh. params: {model.param_count()}", flush=True)

    # encode once
    enc = []
    for ep in corpus:
        rows = [encode_row(d, stoi) for d in ep["rows"]]
        rows = [r for r in rows if r is not None]
        if len(rows) < 4:
            continue
        if ep["force"]:
            obs = [encode_obs(d, stoi) for d in ep["rows"]]
            enc.append({"xy": rows, "obs": obs, "force": True,
                        "src": ep["src"]})
        else:
            enc.append({"xy": rows, "obs": None, "force": False,
                        "src": ep["src"]})
    corpus = enc
    print(f"usable episodes after encoding: {len(corpus)}", flush=True)

    n_val = max(1, len(corpus) // 10)
    train_eps, val_eps = corpus[:-n_val], corpus[-n_val:]
    print(f"train {len(train_eps)} / val {len(val_eps)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    model.train()
    t0 = time.time()
    total_steps = 0
    for epoch in range(a.epochs):
        random.shuffle(train_eps)
        tot, ftot, cnt = 0.0, 0.0, 0
        for ei, ep in enumerate(train_eps):
            opt.zero_grad()
            mem = None
            text_losses, force_losses = [], []
            for ri, (x, y) in enumerate(ep["xy"]):
                mem_in = mem                      # graph alive -> write path
                logits, loss, mem_out = model(x, mem_in, y)
                text_losses.append(loss)
                if ep["force"] and ri >= 1:
                    # MEMORY-FORCING: predict this row's OBS from the
                    # incoming memory tokens alone, through the shared head.
                    probe = model.lm_head(model.ln_f(mem_in))
                    fl = F.cross_entropy(probe.reshape(-1, probe.size(-1)),
                                         ep["obs"][ri].reshape(-1),
                                         ignore_index=pad_id)
                    force_losses.append(fl)
                mem = mem_out                     # NO detach: full BPTT
            ep_text = torch.stack(text_losses).mean()
            if force_losses:
                ep_force = torch.stack(force_losses).mean()
                ep_loss = ep_text + a.force_w * ep_force
            else:
                ep_force = torch.tensor(0.0)
                ep_loss = ep_text
            ep_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += ep_text.item()
            ftot += ep_force.item()
            cnt += 1
            total_steps += 1
            if (ei + 1) % 25 == 0:
                print(f"epoch {epoch+1} ep {ei+1}/{len(train_eps)} "
                      f"text {tot/cnt:.4f} force {ftot/cnt:.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
        # val: text CE only (forcing is a train pressure, not a metric)
        model.eval()
        vtot, vcnt = 0.0, 0
        with torch.no_grad():
            for ep in val_eps:
                mem = None
                for x, y in ep["xy"]:
                    _, loss, mem = model(x, mem, y)
                    vtot += loss.item()
                    vcnt += 1
        model.train()
        print(f"epoch {epoch+1} DONE train_text {tot/max(1,cnt):.4f} "
              f"train_force {ftot/max(1,cnt):.4f} "
              f"val_text {vtot/max(1,vcnt):.4f} {time.time()-t0:.0f}s",
              flush=True)

    outdir = "/srv/cellforge-arena/researchers/beta/gen2"
    out_pt = os.path.join(outdir, a.out + ".pt")
    torch.save({"cfg": cfg, "state": model.state_dict()}, out_pt)
    meta = {"recipe": "beta-g2 memory-forcing RMT", "steps": total_steps,
            "epochs": a.epochs, "lr": a.lr, "force_w": a.force_w,
            "warm_start": a.warm, "seg": SEG, "seed": a.seed,
            "params": model.param_count(),
            "corpus": f"fc {len(fc_eps)} segs + w2 {len(w2_eps)}x16 + "
                      f"gram {len(g_eps)}x12",
            "final_train_text": tot / max(1, cnt),
            "final_train_force": ftot / max(1, cnt),
            "final_val_text": vtot / max(1, vcnt)}
    json.dump(meta, open(os.path.join(outdir, a.out + ".json"), "w"), indent=1)
    print(f"saved {out_pt}", flush=True)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
