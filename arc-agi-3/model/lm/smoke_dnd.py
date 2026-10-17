#!/usr/bin/env python3
"""smoke_dnd.py — compare baseline vs LoRA'd router on fixed prompts.

Measures: parseable-CALL fraction, macro-cell fraction, degenerate gens.
Usage: smoke_dnd.py <model_a.pt> <model_b.pt>
"""
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tok import load_vocab, encode, decode  # noqa: E402
from gpt import GPT  # noqa: E402
import eval_lm as ev  # noqa: E402

LM_DIR = HERE
N_GEN = 20


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = GPT(**ckpt["cfg"])
    m.load_state_dict(ckpt["state"])
    m.eval()
    return m


def degenerate(gen):
    toks = gen.split()
    if len(toks) < 8:
        return True
    run = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            run += 1
            if run >= 10:
                return True
        else:
            run = 1
    return False


def main():
    torch.manual_seed(7)
    vocab, stoi = load_vocab(os.path.join(LM_DIR, "vocab.json"))
    prompts = []
    with open(os.path.join(LM_DIR, "traces_grammar.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            prompts.append(f"STATE {d['state']}\nTHINK {d['think']}")
            if len(prompts) == 3:
                break
    for name, path in [("BASE", sys.argv[1]), ("DND1", sys.argv[2])]:
        model = load_model(path)
        parse_ok = macro = degen = 0
        total = 0
        for p in prompts:
            idx = torch.tensor([encode(p, stoi)], dtype=torch.long)
            if idx.shape[1] > 160:
                idx = idx[:, -160:]
            for _ in range(N_GEN):
                with torch.no_grad():
                    out = model.generate(idx, max_new_tokens=64,
                                         temperature=0.7, top_k=40)
                gen = decode(out[0].tolist()[len(idx[0]):], vocab)
                total += 1
                if degenerate(gen):
                    degen += 1
                nm, ag = ev.parse_call(gen)
                if nm is not None:
                    parse_ok += 1
                    if nm in ("act.repeat_click", "act.ghost_follow"):
                        macro += 1
        print(f"{name}: parse_ok={parse_ok}/{total} "
              f"macro={macro}/{max(parse_ok,1)} degen={degen}/{total}",
              flush=True)


if __name__ == "__main__":
    main()
