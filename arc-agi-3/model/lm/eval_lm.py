"""Eval: tiny LM routes cells to play ARC-AGI-3 games end-to-end.

Loop: STATE -> LM generates THINK + CALL -> execute cell -> OBS -> repeat.
Usage: python eval_lm.py <model.pt> <game1> [game2 ...] [--seed S] [--budget B] [--maxdec D]
"""
import argparse
import json
import logging
import os
import re
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT

import torch
from arc_agi import Arcade
from arcengine import GameState as GState

BASE = "/srv/arc3-claude"
HERE = os.path.dirname(os.path.abspath(__file__))
CALL_RE = re.compile(r"CALL\s*(\{)")


def extract_balanced(text, start):
    """Return the {...} substring with balanced braces starting at start."""
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
    # the word tokenizer spaces out punctuation: strip ALL whitespace
    # (cell names / args never contain meaningful spaces)
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
    ap.add_argument("games", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--maxdec", type=int, default=60)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    vocab, stoi = load_vocab(os.path.join(HERE, "vocab.json"))
    ckpt = torch.load(a.model, map_location="cpu", weights_only=False)
    model = GPT(**ckpt["cfg"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"router params: {model.param_count()}", flush=True)

    lg = logging.getLogger("q")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    logging.getLogger("arc_agi").setLevel(logging.WARNING)

    results = []
    for game in a.games:
        arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                     recordings_dir=f"{BASE}/recordings")
        info = next(e for e in arc.get_environments()
                    if e.game_id.split("-")[0] == game)
        card = arc.create_scorecard()
        env = arc.make(info.game_id, seed=a.seed, scorecard_id=card)
        st = GameState(env, a.budget)
        t0 = time.time()
        hist = []
        ndec = 0
        parse_fails = 0
        while ndec < a.maxdec and st.steps_used < a.budget:
            if st.resp.state == GState.WIN:
                break
            state_txt = call_cell(st, "perceive.describe", {})
            # prompt in exact training format: recent blocks + current STATE
            blocks = hist[-2:]  # each: "STATE.. \nTHINK .. \nCALL .. \nOBS .."
            prompt = "".join(b + "\n" for b in blocks)
            prompt += f"STATE {state_txt}\nTHINK "
            idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
            # keep within the model's block size
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
            # store the full block in training format for the next prompt
            think_txt = gen.split("CALL")[0].strip()
            hist.append(f"STATE {state_txt}\nTHINK {think_txt}\n"
                        f"CALL {json.dumps({'cell': name, 'args': args})}\n"
                        f"OBS {obs[:160]}")
            ndec += 1
        dt = time.time() - t0
        sc = arc.close_scorecard(card)
        run = sc.environments[0].runs[-1].model_dump() if sc and sc.environments else {}
        out = {"agent": f"lmrouter:{os.path.basename(a.model)}", "game": game,
               "seed": a.seed, "budget": a.budget, "used": st.steps_used,
               "decisions": ndec, "parse_fails": parse_fails,
               "levels": run.get("levels_completed"),
               "n_levels": len(info.baseline_actions),
               "score": run.get("score"), "actions": run.get("actions"),
               "state": str(run.get("state")), "secs": round(dt, 1),
               "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               "tags": info.tags}
        results.append(out)
        print("RESULT " + json.dumps(out), flush=True)

    tot_l = sum(r["levels"] or 0 for r in results)
    mean_s = float(sum(r["score"] or 0 for r in results) / max(len(results), 1))
    print(f"SUMMARY games={len(results)} levels={tot_l} mean_score={mean_s:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
