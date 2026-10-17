#!/usr/bin/env python3
"""Diagnose token-39 ('Probing') logit rank at true cell-suffix positions.

Runs the ROOT (no boost) on a game, and at each generation step where the
gate WOULD fire (prev=='act', curr=='.'), records logit[39] and its rank
among 2209. Tells us how much boost is needed to crack top_k=40.
"""
import os, sys, json
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
import logging
import cells
from cells import GameState, call_cell
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from arcengine import GameState as GState

CKPT = "/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt"
GAME = sys.argv[1] if len(sys.argv) > 1 else "s5i5"
NDEC = int(sys.argv[2]) if len(sys.argv) > 2 else 30

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd",
                                  "block_size", "dropout")})
model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
ID_ACT = stoi["act"]; ID_DOT = stoi["."]
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")

info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == GAME)
card = arc.create_scorecard()
env = arc.make(info.game_id, seed=0, scorecard_id=card)
st = GameState(env, 3000)
hist = []
ranks, logits = [], []
for ndec in range(NDEC):
    state_txt = call_cell(st, "perceive.describe", {})
    prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
    if idx.shape[1] > 192:
        idx = idx[:, -192:]
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
    gen_ids = out[0].tolist()[len(idx[0]):]
    # find suffix positions in the GENERATED ids: curr=='.' and prev=='act'
    for i, tid in enumerate(gen_ids):
        if tid == ID_DOT and i > 0 and gen_ids[i - 1] == ID_ACT:
            # recompute logits at this position: feed prefix through model
            prefix = idx[0].tolist() + gen_ids[:i]
            with torch.no_grad():
                lg_out = model(torch.tensor([prefix[-192:]]))[0, -1]
            l39 = lg_out[39].item()
            rank = int((lg_out > l39).sum().item()) + 1
            ranks.append(rank); logits.append(l39)
    gen = decode(gen_ids, vocab)
    from fixed_runner import parse_call
    name, args = parse_call(gen)
    if name is None:
        name, args = "perceive.describe", {}
    call_cell(st, name, args)
    think_txt = gen.split("CALL")[0].strip()
    hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                "\nCALL " + json.dumps({"cell": name, "args": args}))
    if st.resp.state == GState.WIN:
        break

import statistics
print(f"game={GAME} suffix_positions={len(ranks)}", flush=True)
if ranks:
    print(f"token39 rank: median={statistics.median(ranks)} min={min(ranks)} "
          f"max={max(ranks)}", flush=True)
    print(f"token39 logit: median={statistics.median(logits):.2f} "
          f"max={max(logits):.2f}", flush=True)
    # what logit does rank-40 have at these positions? (need full dist; approx)
    print(f"positions with rank<=40: {sum(1 for r in ranks if r <= 40)}/{len(ranks)}",
          flush=True)
