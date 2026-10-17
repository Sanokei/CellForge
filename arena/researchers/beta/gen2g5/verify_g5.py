#!/usr/bin/env python3
"""Beta G5 verification: emission rate + dev-6 RHAE + transfer (measure-only).

Private merge of act.Probing (does NOT touch shared cells.py). Replicates
fixed_runner.run_game faithfully (same prompt, generate params, parse,
per_level bookkeeping, RHAE via rhae_scorer), so numbers are comparable to
judge runs.

Usage: verify_g5.py <checkpoint> [--force K] [--games g1,g2] [--maxdec N]
         [--seeds 0] [--emit-only]
  --force K : force act.Probing every K decisions (stress test; 0=off)
  --emit-only: just count act.Probing emissions, skip RHAE detail
"""
import os, sys, json, time, argparse
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
from rhae_scorer import environment_score
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
cells.REGISTRY["act.Probing"] = ns["c_probing"]

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--force", type=int, default=0)
ap.add_argument("--games", default="lf52,s5i5,ls20,g50t,tu93,bp35")
ap.add_argument("--maxdec", type=int, default=60)
ap.add_argument("--seeds", default="0")
ap.add_argument("--budget", type=int, default=3000)
a = ap.parse_args()

ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd",
                                  "block_size", "dropout")})
sd = {k: v.float() for k, v in ck["state"].items()}
model.load_state_dict(sd)
model.eval()
n_params = sum(v.numel() for v in sd.values())
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")


def run_game(game, seed, budget, maxdec):
    torch.manual_seed(seed)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    baselines = [int(x) for x in info.baseline_actions]
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, budget)
    hist, ndec = [], 0
    per_level, prev_lvl, prev_steps = {}, 0, 0
    failures, n_emit, n_explore = {}, 0, 0
    while ndec < maxdec and st.steps_used < budget:
        if st.resp.state == GState.WIN:
            break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:])
        prompt += "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > model.block_size:
            idx = idx[:, -model.block_size:]
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        if name is None:
            failures["parse_fail"] = failures.get("parse_fail", 0) + 1
            name, args = "perceive.describe", {}
        if a.force and ndec % a.force == 0:
            name, args = "act.Probing", {}
        obs = call_cell(st, name, args)
        if name == "act.Probing":
            n_emit += 1
            n_explore += obs.startswith("PROBE explored")
        lvl = int(st.resp.levels_completed)
        if lvl > prev_lvl:
            for L in range(prev_lvl + 1, lvl + 1):
                per_level[L] = st.steps_used - prev_steps
            prev_lvl, prev_steps = lvl, st.steps_used
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    env_score, level_scores = environment_score(baselines, per_level)
    return {"game": game, "seed": seed, "rhae_pct": round(env_score * 100, 4),
            "levels": prev_lvl, "steps": st.steps_used, "ndec": ndec,
            "emit": n_emit, "explore": n_explore, "failures": failures}


games = [g.strip() for g in a.games.split(",")]
seeds = [int(s) for s in a.seeds.split(",")]
results = []
t0 = time.time()
for game in games:
    for seed in seeds:
        r = run_game(game, seed, a.budget, a.maxdec)
        results.append(r)
        er = r["emit"] / max(1, r["ndec"])
        print(f"{game} s{seed}: RHAE={r['rhae_pct']} lv={r['levels']} "
              f"steps={r['steps']} emit={r['emit']}/{r['ndec']} ({er:.3f}) "
              f"explore={r['explore']} fail={r['failures']}", flush=True)
mean_rhae = sum(r["rhae_pct"] for r in results) / max(1, len(results))
tot_emit = sum(r["emit"] for r in results)
tot_dec = sum(r["ndec"] for r in results)
tot_explore = sum(r["explore"] for r in results)
print(f"MEAN_RHAE={mean_rhae:.4f} EMIT={tot_emit}/{tot_dec} "
      f"({tot_emit/max(1,tot_dec):.4f}) EXPLORE_CALLS={tot_explore} "
      f"params={n_params} t={time.time()-t0:.0f}s", flush=True)
json.dump({"checkpoint": a.checkpoint, "force": a.force, "games": games,
           "seeds": seeds, "mean_rhae": round(mean_rhae, 4),
           "emit": tot_emit, "decisions": tot_dec, "explore": tot_explore,
           "results": results},
          open(os.path.join(WD, f"verify_{os.path.basename(a.checkpoint)}.json"), "w"),
          indent=1)
