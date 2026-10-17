#!/usr/bin/env python3
"""Diagnose lf52 parse fails: dump generations, marking 'Probing' occurrences.
Merges act.Probing privately (like verify_g5.py)."""
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
from fixed_runner import parse_call
from arc_agi import Arcade
from arcengine import GameState as GState

WD = "/srv/cellforge-arena/researchers/beta/gen2g5"
import hashlib
import numpy as np
ns = dict(vars(cells))
ns["hashlib"] = hashlib
ns["np"] = np
src = open(os.path.join(WD, "cell_act_probing_proposal.py")).read()
exec(compile(src, "cell_act_probing_proposal.py", "exec"), ns)
cells.REGISTRY["act.Probing"] = ns["c_probing"]

CKPT = sys.argv[1]
N = 20

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd",
                                  "block_size", "dropout")})
model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")
info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == "lf52")
card = arc.create_scorecard()
env = arc.make(info.game_id, seed=0, scorecard_id=card)
st = GameState(env, 3000)
hist = []
for ndec in range(N):
    state_txt = call_cell(st, "perceive.describe", {})
    prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
    if idx.shape[1] > 192:
        idx = idx[:, -192:]
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
    gen = decode(out[0].tolist()[len(idx[0]):], vocab)
    name, args = parse_call(gen)
    think, _, callpart = gen.partition("CALL")
    probing_in_think = "Probing" in think
    tag = f"parsed={name} probing_in_THINK={probing_in_think}"
    print(f"--- dec{ndec} {tag}", flush=True)
    # print THINK part and CALL part separately, truncated
    print("THINK: " + think[:200].replace("\n", "\\n"), flush=True)
    print("CALLP: " + callpart[:200].replace("\n", "\\n"), flush=True)
    if name is None:
        name, args = "perceive.describe", {}
    call_cell(st, name, args)
    think_txt = think.strip()
    hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                "\nCALL " + json.dumps({"cell": name, "args": args}))
    if st.resp.state == GState.WIN:
        break
