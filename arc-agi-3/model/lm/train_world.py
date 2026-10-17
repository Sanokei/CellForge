"""Train the tiny router LM with a decoder-only WORLD-MODEL auxiliary objective.

The "Modeling" pillar (Ha & Schmidhuber world-models lineage): same decoder-only
weights, no separate encoder. Next-token CE is computed per-position and loss on
tokens in the OBS span (the observation the environment returns after the CALL)
is upweighted --obs_weight x, so the weights learn action -> state-change
prediction. Everything else (data pipeline, GPT arch, AdamW 3e-4, batch 24,
block 192) is identical to train_lm.py.

OBS span localization: render_trace(d) =
    "STATE ...\\nTHINK ...\\nCALL {...}\\nOBS {obs}\\n"
The marker is always "\\nOBS " (the CALL line precedes it). Encode the text
before the marker; its length is exactly the token index where the OBS token
starts. Weight positions [obs_start, doc_end) (excluding the trailing <eos>)
at obs_weight; 1.0 elsewhere.

Target/weight alignment: with x = data[i:i+B], y = data[i+1:i+B+1], the loss
at position t predicts token data[i+1+t], so the batch weight mask is
weights[i+1:i+B+1] -- the 3x weight applies to *predicting* OBS tokens.

Usage: python train_world.py [--init ckpt] --corpus traces_fc.jsonl
                             --out router_v5world [--steps 500]
                             [--obs_weight 3.0] [--batch 24] [--block 192]

init/corpus paths resolve relative to cwd first, then relative to this script
dir, so the script works whether launched from model/ or from lm/.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok import load_vocab, encode, render_trace  # noqa: E402
from gpt import GPT  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OBS_MARKER = "\nOBS "


def resolve(p):
    """Resolve a path relative to cwd first, then to this script's dir."""
    if p is None:
        return None
    if os.path.exists(p):
        return p
    cand = os.path.join(HERE, p)
    if os.path.exists(cand):
        return cand
    return p  # let the caller hit the error with the original name


def obs_start_index(full_text, stoi):
    """Token index of the OBS marker token inside the encoded doc."""
    idx = full_text.find(OBS_MARKER)
    assert idx != -1, "rendered doc missing '\\nOBS ' marker"
    prefix = full_text[:idx + 1]  # keep the newline; tokenizer skips ws
    return len(encode(prefix, stoi))


def build_data_and_weights(docs, stoi, obs_weight):
    """Concatenate encoded docs (+ <eos> each) and a parallel loss-weight
    tensor: obs_weight on OBS-span tokens, 1.0 elsewhere."""
    data = []
    wts = []
    eos = stoi["<eos>"]
    n_obs = 0
    n_tok = 0
    for d in docs:
        text = render_trace(d)
        enc = encode(text, stoi) + [eos]
        start = obs_start_index(text, stoi)
        assert 0 < start < len(enc), f"bad obs span {start} for doc len {len(enc)}"
        w = [1.0] * len(enc)
        for j in range(start, len(enc) - 1):  # up to but excluding <eos>
            w[j] = obs_weight
        data.extend(enc)
        wts.extend(w)
        n_obs += len(enc) - 1 - start
        n_tok += len(enc)
    print(f"obs-span tokens: {n_obs}/{n_tok} ({100.0*n_obs/n_tok:.1f}%) "
          f"at {obs_weight}x")
    return torch.tensor(data, dtype=torch.long), torch.tensor(wts, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--corpus", default="traces_fc.jsonl",
                    help="trace corpus, or comma-separated list for a mixed diet")
    ap.add_argument("--out", default="router_v5world")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--obs_weight", type=float, default=3.0,
                    help="loss multiplier on OBS-span tokens")
    ap.add_argument("--log_every", type=int, default=100)
    a = ap.parse_args()
    torch.manual_seed(0)

    trace_paths = [resolve(c.strip()) for c in a.corpus.split(",")]
    vocab_path = os.path.join(HERE, "vocab.json")
    vocab, stoi = load_vocab(vocab_path)
    print(f"reusing vocab ({len(vocab)} tokens)")

    docs = []
    for trace_path in trace_paths:
        n0 = len(docs)
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        print(f"  {trace_path}: {len(docs)-n0} docs")
    print(f"docs: {len(docs)}")
    data, weights = build_data_and_weights(docs, stoi, a.obs_weight)
    print(f"tokens: {len(data)}")

    cfg = {"vocab_size": len(vocab), "n_layer": 4, "n_head": 4,
           "n_embd": 128, "block_size": a.block, "dropout": 0.0}
    assert cfg["vocab_size"] == 2209, \
        f"vocab_size {cfg['vocab_size']} != 2209 (fixed_runner requirement)"
    model = GPT(**cfg)
    if a.init:
        init_path = resolve(a.init)
        ck = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
        print(f"fine-tuning from {init_path}")
    print(f"params: {model.param_count()}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    t0 = time.time()
    uloss = None
    for step in range(a.steps):
        ix = torch.randint(0, len(data) - a.block, (a.batch,))
        x = torch.stack([data[i:i + a.block] for i in ix])
        y = torch.stack([data[i + 1:i + a.block + 1] for i in ix])
        wb = torch.stack([weights[i + 1:i + a.block + 1] for i in ix])
        opt.zero_grad()
        logits, _ = model(x)
        per = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                              reduction="none")
        uloss = per.mean()
        wloss = (per * wb.view(-1)).mean()
        wloss.backward()
        opt.step()
        if (step + 1) % a.log_every == 0 or a.steps <= a.log_every:
            print(f"step {step+1}/{a.steps} weighted {wloss.item():.4f} "
                  f"unweighted {uloss.item():.4f} {time.time()-t0:.0f}s",
                  flush=True)

    out_pt = os.path.join(HERE, a.out + ".pt")
    torch.save({"cfg": cfg, "state": model.state_dict()}, out_pt)
    with open(os.path.join(HERE, a.out + ".json"), "w") as f:
        json.dump({"params": model.param_count(), "steps": a.steps,
                   "tokens": len(data), "vocab": len(vocab),
                   "corpus": a.corpus, "init": a.init,
                   "obs_weight": a.obs_weight,
                   "final_weighted_loss": float(wloss.item()),
                   "final_unweighted_loss": float(uloss.item())}, f, indent=1)
    print(f"saved {out_pt}")


if __name__ == "__main__":
    main()
