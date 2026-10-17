"""Blackboard eval: tiny LM router with passive intake, act.* CALLs only.

Loop per decision: observers run (zero env actions) -> top-k INTAKE lines ->
prompt = prev CALL/OBS + STATE + INTAKE + THINK -> free THINK gen ->
constrained CALL gen (always valid) -> execute act cell -> OBS.

Usage: python eval_bb.py <model.pt> <game1> [game2 ...] [--seed S]
       [--budget B] [--maxdec D] [--drop obs1,obs2] [--scramble obs3]
       [--bids bids.json] [--log-intake path] [--rhae]
--drop/--scramble: load-bearing ablations. --rhae: print paper-spec RHAE.
"""
import argparse
import json
import logging
import os
import re
import resource
import sys
import time

import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "cells_intake"))
# NOTE: appended (not prepended): v6run/model/intake.py belongs to the
# parallel scripted-agent track; ours lives in cells_intake/
sys.path.append("/srv/arc3-claude/v6run/model")

# lm cells under distinct name (v6run/cells.py must keep its name for blitz)
_spec = importlib.util.spec_from_file_location(
    "lm_cells", os.path.join(HERE, "cells.py"))
lm_cells = importlib.util.module_from_spec(_spec)
sys.modules["lm_cells"] = lm_cells
_spec.loader.exec_module(lm_cells)
GameState, call_cell = lm_cells.GameState, lm_cells.call_cell

# our blackboard intake under a distinct name: v6run/model/intake.py belongs
# to the parallel scripted-agent track and must not shadow ours
_spec2 = importlib.util.spec_from_file_location(
    "bb_intake", os.path.join(HERE, "cells_intake", "intake.py"))
bb_intake = importlib.util.module_from_spec(_spec2)
sys.modules["bb_intake"] = bb_intake
_spec2.loader.exec_module(bb_intake)
new_ctx = bb_intake.new_ctx
collect_intake = bb_intake.collect_intake
format_intake = bb_intake.format_intake
format_state = bb_intake.format_state
BidLogger = bb_intake.BidLogger
from callgen import CallGrammar, generate_think_call
from tok import load_vocab, encode
from gpt import GPT

import torch
from arc_agi import Arcade
from arcengine import GameState as GState

BASE = "/srv/arc3-claude"
NOVELTY_RE = re.compile(r"new c\d+@\((\d+),(\d+)\)")


def novelty_xy(top):
    for n, t, _ in top:
        if n == "novelty":
            m = NOVELTY_RE.search(t)
            if m:
                return (int(m.group(1)), int(m.group(2)))
    return None


def run_game(model, vocab, stoi, grammar, arc, info, seed, budget, maxdec,
             drop, scramble, bids, bid_logger, temperature=0.7):
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, budget)
    ctx = new_ctx()
    prev_frame = None
    ndec, t0 = 0, time.time()
    gen_tokens = 0
    while ndec < maxdec and st.steps_used < budget:
        if st.resp.state == GState.WIN:
            break
        frame = st.frame()
        top = collect_intake(frame, prev_frame, ctx, drop=drop,
                             scramble=scramble, bids=bids)
        if bid_logger:
            bid_logger.log_step(st.steps_used, [n for n, _, _ in top])
        # pure blackboard: STATE + INTAKE only, no history (matches training)
        prompt = format_state(st.level, st.steps_used,
                              st.resp.available_actions)
        prompt += format_intake(top)
        prompt += "THINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > 160:
            idx = idx[:, -160:]
        n0 = idx.shape[1]
        copy_xy = novelty_xy(top)
        idx, think_text, name, args, ngen = generate_think_call(
            model, idx, vocab, stoi, grammar, temperature=temperature,
            copy_xy=copy_xy)
        gen_tokens += ngen
        obs = call_cell(st, name, args)
        if name in ("act.click_at", "act.ghost_follow", "act.repeat_click",
                    "act.pick_drop") and "x" in args and "y" in args:
            ctx["last_click"] = (int(args["x"]), int(args["y"]))
        if bid_logger and st.level > ctx["level"]:
            bid_logger.log_levelup(st.steps_used, st.level)
        ctx["level"] = st.level
        prev_frame = frame
        ndec += 1
    dt = time.time() - t0
    run = {}
    sc = arc.close_scorecard(card)
    if sc and sc.environments:
        runs = sc.environments[0].runs
        if runs:
            run = runs[-1].model_dump()
    if bid_logger:
        bid_logger.log_episode(info.game_id.split("-")[0], seed,
                               run.get("levels_completed", 0),
                               run.get("actions", 0))
    return {"game": info.game_id.split("-")[0], "seed": seed,
            "used": st.steps_used, "decisions": ndec,
            "gen_tokens": gen_tokens,
            "levels": run.get("levels_completed"),
            "n_levels": len(info.baseline_actions or []),
            "score": run.get("score"), "actions": run.get("actions"),
            "level_actions": run.get("level_actions"),
            "level_baselines": list(info.baseline_actions or []),
            "state": str(run.get("state")), "secs": round(dt, 1),
            "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("games", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--maxdec", type=int, default=60)
    ap.add_argument("--drop", default="")
    ap.add_argument("--scramble", default="")
    ap.add_argument("--bids", default=None)
    ap.add_argument("--log-intake", default=None)
    ap.add_argument("--rhae", action="store_true")
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    ckpt = torch.load(a.model, map_location="cpu", weights_only=False)
    need_vs = ckpt["cfg"].get("vocab_size")
    vocab, stoi = load_vocab(os.path.join(HERE, "vocab_bb.json"))
    if need_vs is not None and need_vs != len(vocab):
        # Checkpoint trained with a different vocab (e.g. router_v3.pt ->
        # 2209-token vocab.json). Decoding its samples against vocab_bb
        # (988) raises IndexError in generate_think_call; select the
        # matching vocab instead.
        alt = None
        for cand in ("vocab.json", "vocab_cur.json"):
            cp = os.path.join(HERE, cand)
            if os.path.exists(cp):
                v2, _ = load_vocab(cp)
                if len(v2) == need_vs:
                    alt = cp
                    break
        if alt is None:
            raise SystemExit(
                f"no vocab of size {need_vs} for checkpoint {a.model}")
        print(f"vocab: {os.path.basename(a.model)} needs size {need_vs}; "
              f"using {os.path.basename(alt)}", flush=True)
        vocab, stoi = load_vocab(alt)
    model = GPT(**ckpt["cfg"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"router params: {model.param_count()}", flush=True)
    grammar = CallGrammar(vocab, stoi)
    print(f"grammar: {len(grammar.trie)} first-token options, "
          f"{len(grammar.num_ids)} number tokens", flush=True)

    drop = [d for d in a.drop.split(",") if d]
    scramble = [d for d in a.scramble.split(",") if d]
    bids = None
    if a.bids and os.path.exists(a.bids):
        bids = json.load(open(a.bids))["bids"]
        print(f"loaded bid multipliers: {bids}", flush=True)
    bid_logger = BidLogger(a.log_intake) if a.log_intake else None

    lg = logging.getLogger("q")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                 recordings_dir=f"{BASE}/recordings")

    results = []
    for game in a.games:
        info = next(e for e in arc.get_environments()
                    if e.game_id.split("-")[0] == game)
        r = run_game(model, vocab, stoi, grammar, arc, info, a.seed,
                     a.budget, a.maxdec, drop, scramble, bids, bid_logger,
                     temperature=a.temp)
        r["agent"] = f"bbrouter:{os.path.basename(a.model)}"
        if drop:
            r["drop"] = drop
        if scramble:
            r["scramble"] = scramble
        if a.rhae:
            from rhae_scorer import environment_score
            la = r["level_actions"] or []
            lc = r["levels"] or 0
            per = {l + 1: la[l] for l in range(min(lc, len(la)))
                   if la[l] and la[l] > 0}
            env_rhae, lvl = environment_score(r["level_baselines"], per)
            r["rhae"] = round(env_rhae * 100, 4)
            r["level_scores"] = [round(lvl.get(l + 1, 0.0), 6)
                                 for l in range(len(r["level_baselines"]))]
        results.append(r)
        print("RESULT " + json.dumps(r), flush=True)
    if bid_logger:
        bid_logger.close()
    if a.rhae:
        mean_r = sum(r.get("rhae", 0) for r in results) / max(len(results), 1)
        print(f"SUMMARY games={len(results)} mean_RHAE={mean_r:.2f} "
              f"levels={sum(r['levels'] or 0 for r in results)}", flush=True)


if __name__ == "__main__":
    main()
