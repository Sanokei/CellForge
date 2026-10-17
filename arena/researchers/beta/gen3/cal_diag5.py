#!/usr/bin/env python3
"""Diag5: LVL progression per decision. Does 'stuck at same LVL' separate
solving (dev) from non-solving (tx + unsolved dev) trajectories?

LVL{k} appears in STATE text. If LVL hasn't increased, no level completed.
"""
import os, sys, re
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
from tok import load_vocab, encode, decode

WD = "/srv/cellforge-arena/researchers/beta/gen3"
data = torch.load(os.path.join(WD, "cal_data.pt"), map_location="cpu",
                  weights_only=False)
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))

# find LVL token ids
lvl_ids = {}
for i, tok in enumerate(vocab):
    if tok and re.fullmatch(r"LVL\d+", tok):
        lvl_ids[tok] = i
print("LVL tokens:", lvl_ids, flush=True)

def cur_lvl(ids):
    # current STATE span = after last STATE(30) token; find first LVL token in it
    idx = (ids == 30).nonzero()
    if len(idx) == 0:
        return None
    span = ids[idx[-1].item():]
    for tok, tid in lvl_ids.items():
        if (span == tid).any():
            return int(tok[3:])
    return None

for lab in sorted(data.keys()):
    lvls = []
    for d in data[lab]["decs"]:
        lv = cur_lvl(d["ids"])
        lvls.append(lv)
    cells = data[lab]["cells"]
    print(f"{lab}: LVL per decision: {lvls}", flush=True)
print("DIAG5 DONE")
