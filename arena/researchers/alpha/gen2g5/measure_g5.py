"""ALPHA-GEN4 dev eval: run dev games with prototype cells injected into REGISTRY
(mirrors the post-merge shared library state). Reports dev RHAE per game,
exploratory-call rate (Probing+scanning invocations per decision), parse fails,
and per-cell emission counts. Also supports --no-cells for the graceful-
degradation check (unmerged registry: new-leaf emissions -> parse_fail).

Pure dev-games. No transfer. No grad.
"""
import sys, os, json, re
import numpy as np
import torch

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
os.environ["OMP_NUM_THREADS"] = "1"

import cells as CELLMOD
WANT_CELLS = "--no-cells" not in sys.argv
if WANT_CELLS:
    inj = dict(vars(CELLMOD)); inj["__name__"] = "proposals"
    exec(compile(open("/srv/cellforge-arena/researchers/alpha/gen2g5/cell_proposals_gen4.py").read(),
                 "proposals", "exec"), inj)
    CELLMOD.REGISTRY["act.Clicking"] = inj["c_Clicking"]
    CELLMOD.REGISTRY["act.scanning"] = inj["c_scanning"]
    print("injected: act.Clicking, act.scanning", flush=True)
else:
    print("no-cells mode: prototype cells NOT in registry", flush=True)

from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from arcengine import GameState as GState
from rhae_scorer import environment_score
import logging

CKPT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else \
    "/srv/cellforge-arena/researchers/alpha/gen3/alpha_g4.pt"
rest = [a for a in sys.argv[1:] if not a.startswith("--")][1:]
GAMES = rest[0].split(",") if rest else ["lf52", "s5i5", "ls20", "g50t", "tu93", "bp35"]
MAXDEC = 60
BUDGET = 3000
SEED = 0
DISCOVERY = {"act.Clicking", "act.scanning", "act.sweep", "act.target"}

torch.manual_seed(12345)
sd = torch.load(CKPT, map_location="cpu", weights_only=True)["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg); model.load_state_dict(sd); model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))

lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")

CALL_RE = re.compile(r"CALL\s*(\{)")
def extract_balanced(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return text[start:i+1]
    return None
def parse_call(text):
    m = CALL_RE.search(text)
    if not m: return None, None
    blob = extract_balanced(text, m.start(1))
    if not blob: return None, None
    try:
        d = json.loads(re.sub(r"\s+", "", blob))
        name = d.get("cell")
        if name not in REGISTRY: return None, None
        return name, d.get("args", {})
    except Exception:
        return None, None

from collections import Counter
tot_cells = Counter(); tot_disc = 0; tot_dec = 0; tot_parse = 0
rhaes = {}
for game in GAMES:
    torch.manual_seed(SEED)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    baselines = [int(x) for x in info.baseline_actions]
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=SEED, scorecard_id=card)
    st = GameState(env, BUDGET)
    hist, ndec = [], 0
    per_level, prev_lvl, prev_steps = {}, 0, 0
    fails = {}
    while ndec < MAXDEC and st.steps_used < BUDGET:
        if st.resp.state == GState.WIN: break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > model.block_size: idx = idx[:, -model.block_size:]
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        if name is None:
            fails["parse_fail"] = fails.get("parse_fail", 0) + 1
            name, args = "perceive.describe", {}
        call_cell(st, name, args)
        tot_cells[name] += 1; tot_dec += 1
        if name in DISCOVERY: tot_disc += 1
        lvl = int(st.resp.levels_completed)
        if lvl > prev_lvl:
            for L in range(prev_lvl + 1, lvl + 1): per_level[L] = st.steps_used - prev_steps
            prev_lvl, prev_steps = lvl, st.steps_used
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    env_score, _ = environment_score(baselines, per_level)
    rhaes[game] = round(env_score * 100, 4)
    print(f"{game}: rhae={rhaes[game]} levels={sorted(per_level)} fails={fails}", flush=True)

dev6 = round(sum(rhaes.values()) / len(rhaes), 4)
print("\n=== SUMMARY ===", flush=True)
print("per-game RHAE:", rhaes, flush=True)
print(f"dev-{len(GAMES)} RHAE: {dev6} (guardrail >=0.31: {'PASS' if dev6 >= 0.31 else 'FAIL'})", flush=True)
print(f"decisions: {tot_dec}, exploratory {tot_disc} -> rate {tot_disc/max(1,tot_dec):.4f}", flush=True)
print("cell counts:", dict(tot_cells), flush=True)
print("discovery cells:", {c: tot_cells[c] for c in DISCOVERY if tot_cells[c]}, flush=True)
