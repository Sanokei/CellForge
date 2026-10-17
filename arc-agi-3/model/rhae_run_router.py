"""First true-RHAE run: router_v3 on dev games, per-level action counts -> RHAE.
Usage: python rhae_run_router.py <model.pt> <game> [--seed S] [--budget B] [--maxdec D]
Prints JSON with per-level actions + RHAE. Uses framework baselines as human h_l."""
import argparse
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm"))
from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT
from rhae_scorer import environment_score
import torch
from arc_agi import Arcade
from arcengine import GameState as GState

BASE = "/srv/arc3-claude"
HERE = os.path.dirname(os.path.abspath(__file__))
CALL_RE = re.compile(r"CALL\s*(\{)")


def extract_balanced(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_call(text):
    m = CALL_RE.search(text)
    if not m:
        return None, None
    blob = extract_balanced(text, m.start(1))
    if not blob:
        return None, None
    compact = re.sub(r"\s+", "", blob)
    try:
        d = json.loads(compact)
        name = d.get("cell")
        if name not in REGISTRY:
            return None, None
        return name, d.get("args", {})
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("game")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--maxdec", type=int, default=60)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
    ckpt = torch.load(a.model, map_location="cpu", weights_only=False)
    model = GPT(**ckpt["cfg"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    lg = logging.getLogger("q")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    arc = Arcade(logger=lg, environments_dir=BASE + "/environment_files",
                 recordings_dir=BASE + "/recordings")
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == a.game)
    baselines = [int(x) for x in info.baseline_actions]
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=a.seed, scorecard_id=card)
    st = GameState(env, a.budget)
    hist = []
    ndec = 0
    parse_fails = 0
    per_level = {}
    prev_lvl = 0
    prev_steps = 0
    t0 = time.time()
    while ndec < a.maxdec and st.steps_used < a.budget:
        if st.resp.state == GState.WIN:
            break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:])
        prompt += "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > 160:
            idx = idx[:, -160:]
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.7,
                                 top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        if name is None:
            parse_fails += 1
            name, args = "perceive.describe", {}
        obs = call_cell(st, name, args)
        lvl = int(st.resp.levels_completed)
        if lvl > prev_lvl:
            for L in range(prev_lvl + 1, lvl + 1):
                per_level[L] = st.steps_used - prev_steps
            prev_lvl = lvl
            prev_steps = st.steps_used
        think_txt = gen.split("CALL")[0].strip()
        call_json = json.dumps({"cell": name, "args": args})
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + call_json + "\nOBS " + obs[:160])
        ndec += 1
    env_score, level_scores = environment_score(baselines, per_level)
    print("RHAE_RESULT " + json.dumps({
        "agent": os.path.basename(a.model), "game": a.game, "seed": a.seed,
        "baselines": baselines, "per_level_actions": per_level,
        "level_scores": {str(k): round(v, 4) for k, v in level_scores.items()},
        "rhae_pct": round(env_score * 100, 2),
        "levels_completed": prev_lvl, "steps_used": st.steps_used,
        "parse_fails": parse_fails, "secs": round(time.time() - t0, 1)}),
        flush=True)


if __name__ == "__main__":
    main()
