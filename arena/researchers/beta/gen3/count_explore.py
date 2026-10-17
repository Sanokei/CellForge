#!/usr/bin/env python3
"""Count exploratory-cell emissions (act.sweep/act.try) per decision.
Measure-only. Usage: count_explore.py <checkpoint> <maxdec>"""
import os, sys, json, time
import torch

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

CKPT, MAXDEC = sys.argv[1], int(sys.argv[2])
TX = [g["game"] for g in
      json.load(open("/srv/cellforge-arena/gen2/judge_transfer_beta.json"))["results"]]
DEV = ["lf52", "s5i5", "ls20", "g50t", "tu93", "bp35"]

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

def run_game(game, seed=0, budget=3000):
    torch.manual_seed(seed)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, budget)
    hist, ndec, exp = [], 0, 0
    while ndec < MAXDEC and st.steps_used < budget:
        if st.resp.state == GState.WIN:
            break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:])
        prompt += "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > 192:
            idx = idx[:, -192:]
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        if name is None:
            name, args = "perceive.describe", {}
        if name in ("act.sweep", "act.try"):
            exp += 1
        call_cell(st, name, args)
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    return exp, ndec

t0 = time.time()
tot_e, tot_n = 0, 0
for lab, games in (("DEV", DEV), ("TX", TX)):
    e, n = 0, 0
    for g in games:
        ge, gn = run_game(g)
        e += ge; n += gn
    tot_e += e; tot_n += n
    print(f"{lab}: exploratory {e}/{n} = {e/max(1,n):.3f}", flush=True)
print(f"TOTAL: {tot_e}/{tot_n} = {tot_e/max(1,tot_n):.4f} t={time.time()-t0:.0f}s", flush=True)
