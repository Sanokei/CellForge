"""Test the grounded-ghost_follow upgrade hypothesis.

Monkeypatch c_ghost_follow with a version that snaps (x,y) to the nearest
perceived object center (within 32px; else use as-is). Run dev-2 with the
BASELINE alpha_g4 (no weight edits). If dev holds, the upgrade is safe and
directly grounds the model's 17.8% transfer ghost_follow emissions.

This tests: does geometric grounding break memorized solves?
"""
import sys, os, json, re
import numpy as np
import torch

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "lm"))
os.environ["OMP_NUM_THREADS"] = "1"

import cells as CELLMOD
from cells import GameState, call_cell, REGISTRY, _objs, _after, _new_small_comp
from arcengine import GameAction, GameState as GState

# grounded upgrade: snap (x,y) to nearest object center
_orig_ghost = CELLMOD.c_ghost_follow
def c_ghost_grounded(st, args):
    x, y = int(args.get("x", 32)), int(args.get("y", 32))
    f = st.frame()
    objs = _objs(f)
    if objs:
        best, bd = None, 1e9
        for o in objs:
            cx = (o["x0"] + o["x1"]) // 2; cy = (o["y0"] + o["y1"]) // 2
            d = abs(cx - x) + abs(cy - y)
            if d < bd: bd, best = d, (cx, cy)
        if best is not None and bd <= 64:  # Manhattan snap radius
            x, y = best
    before = st.frame().copy()
    note = st._step(GameAction.ACTION6, data={"x": x, "y": y})
    after = st.frame()
    g = _new_small_comp(before, after, near=(x, y))
    base = _after(st, note + f"GHOSTLINK click({x},{y})", 6)
    if g:
        gx, gy, size = g
        return base + f" -> ghost marker at ({gx},{gy}) size{size}"
    return base + " -> no ghost marker appeared"

CELLMOD.c_ghost_follow = c_ghost_grounded
CELLMOD.REGISTRY["act.ghost_follow"] = c_ghost_grounded
print("patched: act.ghost_follow now snaps to nearest object center", flush=True)

from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from rhae_scorer import environment_score
import logging

CKPT = "/srv/cellforge-arena/researchers/alpha/gen3/alpha_g4.pt"
GAMES = ["lf52", "s5i5"]
MAXDEC = 60; BUDGET = 3000; SEED = 0

sd = torch.load(CKPT, map_location="cpu", weights_only=True)["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg); model.load_state_dict(sd); model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")

CALL_RE = re.compile(r"CALL\s*(\{)")
def parse_call(text):
    m = CALL_RE.search(text)
    if not m: return None, None
    s = m.start(1); depth = 0
    for i in range(s, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    d = json.loads(re.sub(r"\s+", "", text[s:i+1]))
                    n = d.get("cell")
                    if n in REGISTRY: return n, d.get("args", {})
                except Exception: pass
                return None, None
    return None, None

for game in GAMES:
    torch.manual_seed(SEED)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    baselines = [int(x) for x in info.baseline_actions]
    card = arc.create_scorecard(); env = arc.make(info.game_id, seed=SEED, scorecard_id=card)
    st = GameState(env, BUDGET); hist, ndec = [], 0
    per_level, prev_lvl, prev_steps = {}, 0, 0
    snaps = []
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
        if name is None: name, args = "perceive.describe", {}
        # record snap distance for ghost_follow
        if name == "act.ghost_follow":
            ox, oy = int(args.get("x", 32)), int(args.get("y", 32))
            objs = _objs(st.frame())
            if objs:
                bd = min(abs((o["x0"]+o["x1"])//2 - ox) + abs((o["y0"]+o["y1"])//2 - oy) for o in objs)
                snaps.append(bd)
        call_cell(st, name, args)
        lvl = int(st.resp.levels_completed)
        if lvl > prev_lvl:
            for L in range(prev_lvl+1, lvl+1): per_level[L] = st.steps_used - prev_steps
            prev_lvl, prev_steps = lvl, st.steps_used
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    env_score, _ = environment_score(baselines, per_level)
    import numpy as np
    print(f"{game}: rhae={env_score*100:.4f} levels={sorted(per_level)} "
          f"snap_dist median={np.median(snaps) if snaps else -1:.1f} n={len(snaps)}", flush=True)
print("done")
