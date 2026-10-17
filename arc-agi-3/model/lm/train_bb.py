"""Train the blackboard router LM from scratch.

Corpus: traces_bb.jsonl (STATE+INTAKE -> THINK -> CALL(act.*) -> OBS) plus a
small sample of realtext/valid.txt as an auxiliary English fluency objective
(THINK quality). Vocab rebuilt from scratch (v3's vocab lacked perceive.*
tokens entirely; the blackboard LM only emits act.* CALLs).

Usage: python train_bb.py [--steps N] [--out router_v4]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok import build_vocab, load_vocab, encode, SPECIAL
from gpt import GPT

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REALTEXT = "/srv/cellforge-private/realtext/valid.txt"


def render_bb(d):
    return (f"STATE {d['state']}\n{d['intake']}\nTHINK {d['think']}\n"
            f"CALL {json.dumps(d['call'], separators=(',', ':'))}\n"
            f"OBS {d['obs']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--corpus", default="traces_bb.jsonl")
    ap.add_argument("--realtext-n", type=int, default=300)
    ap.add_argument("--out", default="router_v4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    docs = []
    with open(os.path.join(HERE, a.corpus)) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(render_bb(json.loads(line)))
    rng = random.Random(1234)
    rt_lines = [l.rstrip("\n") for l in open(REALTEXT) if l.strip()]
    rng.shuffle(rt_lines)
    for l in rt_lines[: a.realtext_n]:
        docs.append(l + "\n")
    n_trace = 0
    with open(os.path.join(HERE, a.corpus)) as f:
        for line in f:
            if line.strip():
                n_trace += 1
    print(f"trace docs: {n_trace} + {a.realtext_n} realtext")

    vocab_path = os.path.join(HERE, "vocab_bb.json")
    # Build vocab from the ACTUAL training render (STATE+INTAKE+THINK+CALL+OBS),
    # not tok.render_trace (which omits intake lines). Then force-include
    # mandatory grammar tokens so the constrained decoder never hits <unk>.
    from collections import Counter
    from tok import tokenize
    cnt = Counter()
    for d in docs:
        cnt.update(tokenize(d))
    vocab = list(SPECIAL)
    for tok, c in cnt.most_common():
        if c < 2 or len(vocab) >= 4000:
            break
        if tok not in vocab:
            vocab.append(tok)
    have = set(vocab)
    mandatory = []
    for cell in ["act.try", "act.sweep", "act.click_at", "act.pick_drop",
                 "act.repeat_click", "act.ghost_follow", "act.reset"]:
        mandatory.extend(tokenize(cell))
    mandatory.extend(str(i) for i in range(64))
    mandatory.extend(tokenize('{"cell":"x","args":{}}'))
    mandatory.extend(["CALL", "INTAKE", "THINK", "STATE", "OBS"])
    added_mand = 0
    for tok in mandatory:
        if tok not in have and len(vocab) < 4000:
            vocab.append(tok)
            have.add(tok)
            added_mand += 1
    # realtext fluency tokens
    added_rt = 0
    rt_cnt = Counter()
    for l in rt_lines[: a.realtext_n]:
        rt_cnt.update(tokenize(l))
    for tok, c in rt_cnt.most_common():
        if len(vocab) >= 4000:
            break
        if tok not in have and c >= 2:
            vocab.append(tok)
            have.add(tok)
            added_rt += 1
    stoi = {t: i for i, t in enumerate(vocab)}
    with open(vocab_path, "w") as f:
        json.dump({"vocab": vocab, "n_docs": len(docs)}, f)
    print(f"vocab {len(vocab)} (+{added_mand} mandatory, +{added_rt} realtext)")

    data = []
    eos = stoi["<eos>"]
    for d in docs:
        data.extend(encode(d, stoi) + [eos])
    data = torch.tensor(data, dtype=torch.long)
    print(f"tokens: {len(data)}")

    cfg = {"vocab_size": len(vocab), "n_layer": 4, "n_head": 4,
           "n_embd": 128, "block_size": a.block, "dropout": 0.0}
    model = GPT(**cfg)
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
                   "corpus": a.corpus, "realtext_n": a.realtext_n,
                   "final_loss": float(loss.item())}, f, indent=1)
    print(f"saved {a.out}.pt")


if __name__ == "__main__":
    main()
