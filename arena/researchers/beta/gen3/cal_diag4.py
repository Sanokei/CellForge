#!/usr/bin/env python3
"""Diag4: measure 'stuck' signal = current STATE span lacks chg token.
chg{N} (N>0, single token) appears in STATE text iff last action changed pixels.
Absent => action had no observable effect => stuck.
"""
import os, sys, re
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
from tok import load_vocab, encode

WD = "/srv/cellforge-arena/researchers/beta/gen3"
data = torch.load(os.path.join(WD, "cal_data.pt"), map_location="cpu",
                  weights_only=False)
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))

# all chgN single tokens in vocab
chg_ids = set()
for i, tok in enumerate(vocab):
    if re.fullmatch(r"chg\d+", tok or ""):
        chg_ids.add(i)
print(f"chgN tokens in vocab: {len(chg_ids)}", flush=True)
STATE_id = encode("STATE", stoi)
print("STATE token ids:", STATE_id, flush=True)
state_id = STATE_id[0] if len(STATE_id) == 1 else None

def cur_state_has_chg(ids):
    # current STATE span = after last STATE token
    idx = (ids == state_id).nonzero()
    if len(idx) == 0 or state_id is None:
        return None
    span = ids[idx[-1].item():]
    return any(int(t) in chg_ids for t in span)

for lab in sorted(data.keys()):
    dd = data[lab]["decs"]
    n_stuck, n_ok, n_none = 0, 0, 0
    for d in dd:
        r = cur_state_has_chg(d["ids"])
        if r is None: n_none += 1
        elif r: n_ok += 1
        else: n_stuck += 1
    tot = n_stuck + n_ok
    print(f"{lab}: stuck(no chg in cur STATE)={n_stuck}/{tot} "
          f"= {n_stuck/max(1,tot):.2f}  (none={n_none})", flush=True)
print("DIAG4 DONE")
