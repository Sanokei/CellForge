#!/usr/bin/env python3
"""Beta G4 CALIBRATION probe (measure-only, no authoring).

Captures per-POSITION block-3 output residuals (what a new block-4 would see)
over dev + transfer trajectories, plus emitted cell per decision.

Transfer = MEASURE-ONLY. Labels DEV-1..6 / TX-A..L only; game codes are the
established runner input lists, never used as features.
"""
import json, os, sys, time
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
import logging
from cells import GameState, call_cell
from tok import load_vocab, encode, decode
from gpt import GPT
from fixed_runner import parse_call
from arc_agi import Arcade
from arcengine import GameState as GState

WD = "/srv/cellforge-arena/researchers/beta/gen3"
DEV = ["lf52", "s5i5", "ls20", "g50t", "tu93", "bp35"]
DEV_LABELS = [f"DEV-{i+1}" for i in range(6)]
TX = json.load(open("/srv/cellforge-arena/gen2/judge_transfer_beta.json"))
TX = [g["game"] for g in TX["results"]]
TX_LABELS = [f"TX-{chr(65+i)}" for i in range(len(TX))]
MAXDEC = 10

CKPT = "/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = GPT(**{k: cfg[k] for k in
               ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
model.load_state_dict(ck["state"])
model.eval()

cap = {}
def _mkmid():
    def h(module, inp, out):
        cap["b3"] = out.detach().cpu()   # block-3 output, ALL positions
    return h
model.blocks[3].register_forward_hook(_mkmid())

vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")

@torch.no_grad()
def run_game(game, label, seed=0, budget=3000, maxdec=MAXDEC):
    torch.manual_seed(seed)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, budget)
    hist, ndec = [], 0
    decs = []  # per-decision dicts (pooling is per-decision-context)
    cells = []
    while ndec < maxdec and st.steps_used < budget:
        if st.resp.state == GState.WIN:
            break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:])
        prompt += "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > 192:
            idx = idx[:, -192:]
        out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        if name is None:
            name, args = "perceive.describe", {}
        # capture pass: full final context, all positions, block-3 output
        idx_f = out[:, -192:]
        model(idx_f)
        decs.append({"X": cap["b3"][0].clone(), "ids": idx_f[0].clone()})
        cells.append(name)
        call_cell(st, name, args)
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    return {"decs": decs, "cells": cells, "ndec": ndec}

t0 = time.time()
data = {}
pairs = list(zip(DEV_LABELS, DEV)) + list(zip(TX_LABELS, TX))
for label, game in pairs:
    r = run_game(game, label)
    data[label] = r
    npos = sum(d["X"].shape[0] for d in r["decs"])
    exp = sum(1 for c in r["cells"] if c in ("act.sweep", "act.try"))
    print(f"{label}: ndec={r['ndec']} pos={npos} explore_cells={exp}/{r['ndec']} "
          f"t={time.time()-t0:.0f}s", flush=True)

torch.save(data, os.path.join(WD, "cal_data.pt"))
print("CAL PROBE DONE", flush=True)
