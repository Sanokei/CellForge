"""Train the tiny router LM on the trace corpus.

Usage: python train_lm.py [--steps N] [--batch B]
Saves lm/router_v1.pt + lm/vocab.json + lm/router_v1.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok import build_vocab, load_vocab, encode, render_trace, SPECIAL
from gpt import GPT

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--corpus", default="traces_w2.jsonl")
    ap.add_argument("--out", default="router_v2")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    a = ap.parse_args()
    torch.manual_seed(0)
    trace_path = os.path.join(HERE, a.corpus)
    vocab_path = os.path.join(HERE, "vocab.json")

    vocab_path = os.path.join(HERE, "vocab.json")
    if a.init and os.path.exists(vocab_path):
        vocab, stoi = load_vocab(vocab_path)
        print(f"reusing vocab ({len(vocab)} tokens) for fine-tune")
    else:
        vocab, stoi = build_vocab(trace_path, vocab_path)
    docs = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    data = []
    eos = stoi["<eos>"]
    for d in docs:
        data.extend(encode(render_trace(d), stoi) + [eos])
    data = torch.tensor(data, dtype=torch.long)
    print(f"tokens: {len(data)}")

    cfg = {"vocab_size": len(vocab), "n_layer": 4, "n_head": 4,
           "n_embd": 128, "block_size": a.block, "dropout": 0.0}
    model = GPT(**cfg)
    if a.init:
        ck = torch.load(os.path.join(HERE, a.init), map_location="cpu",
                        weights_only=False)
        model.load_state_dict(ck["state"])
        print(f"fine-tuning from {a.init}")
    print(f"params: {model.param_count()}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    t0 = time.time()
    for step in range(a.steps):
        ix = torch.randint(0, len(data) - a.block, (a.batch,))
        x = torch.stack([data[i:i + a.block] for i in ix])
        y = torch.stack([data[i + 1:i + a.block + 1] for i in ix])
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{a.steps} loss {loss.item():.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"cfg": cfg, "state": model.state_dict()},
               os.path.join(HERE, a.out + ".pt"))
    with open(os.path.join(HERE, a.out + ".json"), "w") as f:
        json.dump({"params": model.param_count(), "steps": a.steps,
                   "tokens": len(data), "vocab": len(vocab),
                   "corpus": a.corpus,
                   "final_loss": float(loss.item())}, f, indent=1)
    print(f"saved {a.out}.pt")


if __name__ == "__main__":
    main()
