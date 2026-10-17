#!/usr/bin/env python3
"""Beta G5 prototype: forced-probe stress test for act.Probing.

Loads the REAL cells.py, merges the proposed c_probing privately
(monkey-patch; the shared cells.py is NOT touched), then rolls out the
elitism root with FORCED act.Probing calls every K decisions.

Test A (dev safety): lf52, s5i5 -- want ZERO explores during L1 solves
   (stuck-gate false positives must be ~0) and levels banked as usual.
Test B (transfer utility): transfer games (measure-only) -- want the
   probe to explore when the model flails.

Usage: probe_stress.py <checkpoint> <K> <dev|tx>
"""
import os, sys, json, time
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

# ---- private merge of the proposed cell (prototype only) ----
import hashlib
import numpy as np
ns = dict(vars(cells))
ns["hashlib"] = hashlib
ns["np"] = np
src = open(os.path.join(WD, "cell_act_probing_proposal.py")).read()
exec(compile(src, "cell_act_probing_proposal.py", "exec"), ns)
c_probing = ns["c_probing"]
cells.REGISTRY["act.Probing"] = c_probing
assert "act.Probing" in cells.REGISTRY
print("cell merged privately: act.Probing", flush=True)

CKPT, K = sys.argv[1], int(sys.argv[2])
MODE = sys.argv[3]
TX = [g["game"] for g in
      json.load(open("/srv/cellforge-arena/gen2/judge_transfer_beta.json"))["results"]]
DEV = ["lf52", "s5i5", "ls20", "g50t", "tu93", "bp35"]
GAMES = DEV if MODE == "dev" else TX

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


def run_game(game, seed=0, budget=3000, maxdec=60):
    torch.manual_seed(seed)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, budget)
    hist, ndec = [], 0
    n_probe = n_explore = n_noop = 0
    log = []
    while ndec < maxdec and st.steps_used < budget:
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
        forced = (ndec % K == 0)
        if forced:
            name, args = "act.Probing", {}
        obs = call_cell(st, name, args)
        if forced:
            n_probe += 1
            explored = obs.startswith("PROBE explored")
            n_explore += explored
            n_noop += (not explored)
            log.append((ndec, st.steps_used, int(st.resp.levels_completed),
                        "EXPLORE" if explored else "noop"))
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    return {"game": game, "levels": int(st.resp.levels_completed),
            "steps": st.steps_used, "ndec": ndec,
            "probe_calls": n_probe, "explores": n_explore, "noops": n_noop,
            "log": log}


t0 = time.time()
tot = {"probe_calls": 0, "explores": 0, "noops": 0}
for g in GAMES:
    r = run_game(g)
    tot["probe_calls"] += r["probe_calls"]
    tot["explores"] += r["explores"]
    tot["noops"] += r["noops"]
    print(f"{g}: levels={r['levels']} steps={r['steps']} ndec={r['ndec']} "
          f"probe={r['probe_calls']} explore={r['explores']} noop={r['noops']}",
          flush=True)
    for ndec, steps, lvl, kind in r["log"]:
        if kind == "EXPLORE":
            print(f"    EXPLORE dec{ndec} step{steps} lvl{lvl}", flush=True)
print(f"TOTAL probe={tot['probe_calls']} explore={tot['explores']} "
      f"noop={tot['noops']} t={time.time()-t0:.0f}s", flush=True)
