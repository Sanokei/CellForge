"""Agent API for the CellForge AUTHOR->COMPETE->SELECT->BREED loop.

Researchers author researchers/<name>/agent.py with:
    NAME = "..."
    WEIGHTS = {"param": value, ...}   # every numeric knob the policy uses
    CELLS = [...]                      # subset of registry cells used
    def decide(state, history, rng) -> (think, cell, args)

state: dict with level, steps_used, budget, n_objects, objects[{c,x0,y0,x1,y1,area}],
       changed_px, changed_bbox, avail[list of int], hud (str)
history: list of (think, cell, args, obs_short) for this game
The harness runs the decide->cell->env loop and scores via the scorecard.
"""
import json
import os
import re
import sys

import numpy as np

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text):
    """Harness-side reasoning meter. Counts word/punct tokens in think text."""
    return len(TOKEN_RE.findall(text or ""))

sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
sys.path.insert(0, "/srv/arc3-claude/v6run/model")
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "lm_cells", "/srv/arc3-claude/v6run/model/lm/cells.py")
lm_cells = importlib.util.module_from_spec(_spec)
sys.modules["lm_cells"] = lm_cells
_spec.loader.exec_module(lm_cells)

from arcengine import GameState as GState


def state_dict(st):
    f = st.frame()
    objs = lm_cells._objs(f)[:8]
    chg_px, chg_bbox = 0, None
    if st.prev_frame is not None:
        d = f != st.prev_frame
        chg_px = int(d.sum())
        if chg_px:
            ys, xs = np.nonzero(d)
            chg_bbox = (int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max()))
    return {
        "level": st.level,
        "steps_used": st.steps_used,
        "budget": st.budget,
        "n_objects": len(objs),
        "objects": [{"c": o["c"], "x0": o["x0"], "y0": o["y0"],
                     "x1": o["x1"], "y1": o["y1"], "area": o["area"],
                     "cx": (o["x0"] + o["x1"]) // 2,
                     "cy": (o["y0"] + o["y1"]) // 2} for o in objs],
        "changed_px": chg_px,
        "changed_bbox": chg_bbox,
        "avail": sorted(int(x) for x in st.resp.available_actions),
        "won": st.resp.state == GState.WIN,
        "game_over": st.resp.state == GState.GAME_OVER,
    }


def load_agent(path):
    spec = importlib.util.spec_from_file_location("agent_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "decide") and hasattr(mod, "WEIGHTS"), \
        f"{path} must define decide() and WEIGHTS"
    return mod


def run_agent_game(agent_mod, game, seed, budget, max_decisions, arc,
                   info, card, env, token_budget=None):
    """Play one game. Returns (result_dict, decision_trace).
    token_budget: max think-tokens for the game (harness-metered). Exceeding
    the cap ends the run structurally — the agent acts on what it concluded.
    """
    import random
    import time
    rng = random.Random(seed * 7919 + 13)
    st = lm_cells.GameState(env, budget)
    t0 = time.time()
    history = []
    trace = []
    ndec = 0
    parse_fails = 0
    think_tokens = 0
    capped = False
    while ndec < max_decisions and st.steps_used < budget:
        if st.resp.state == GState.WIN:
            break
        s = state_dict(st)
        try:
            think, cell, args = agent_mod.decide(s, history, rng)
        except Exception as e:
            think, cell, args = f"decide() crashed: {e}", "perceive.describe", {}
            parse_fails += 1
        think_tokens += count_tokens(think)
        if token_budget is not None and think_tokens > token_budget:
            capped = True
            break  # harness cuts reasoning off; run ends on current score
        if cell not in lm_cells.REGISTRY:
            think, cell, args = f"unknown cell {cell}", "perceive.describe", {}
            parse_fails += 1
        obs = lm_cells.call_cell(st, cell, args or {})
        history.append((str(think)[:200], cell, args, obs[:160]))
        trace.append((cell, json.dumps(args, sort_keys=True)))
        ndec += 1
    dt = time.time() - t0
    return {
        "decisions": ndec, "secs": round(dt, 2), "parse_fails": parse_fails,
        "steps": st.steps_used, "think_tokens": think_tokens,
        "capped": capped,
    }, trace


def purity_check(agent_path, probe_game="ft09", steps=300):
    """Perturb each WEIGHTS entry; the agent's decision trace must change.
    Returns {weight: True(load-bearing)/False(decorative)/'unperturbable'}."""
    import copy
    mod = load_agent(agent_path)
    base = _quick_trace(mod, probe_game, steps)
    out = {}
    for k, v in mod.WEIGHTS.items():
        if isinstance(v, bool):
            pv = not v
        elif isinstance(v, int):
            pv = v + 1
        elif isinstance(v, float):
            pv = v + 0.5 if v != 0 else 0.5
        else:
            out[k] = "unperturbable"
            continue
        mod2 = load_agent(agent_path)
        mod2.WEIGHTS = dict(mod.WEIGHTS)
        mod2.WEIGHTS[k] = pv
        t2 = _quick_trace(mod2, probe_game, steps)
        out[k] = (t2 != base)
    return out


def _quick_trace(mod, game, steps):
    # light trace without scorecard: fixed seed, count decisions
    import random
    from arc_agi import Arcade
    import logging
    lg = logging.getLogger("pq")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    arc = Arcade(logger=lg,
                 environments_dir="/srv/arc3-claude/environment_files",
                 recordings_dir="/srv/arc3-claude/recordings")
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=0, scorecard_id=card)
    rng = random.Random(13)
    st = lm_cells.GameState(env, steps)
    trace = []
    ndec = 0
    while ndec < 40 and st.steps_used < steps:
        if st.resp.state == GState.WIN:
            break
        s = state_dict(st)
        try:
            think, cell, args = mod.decide(s, [], rng)
        except Exception:
            break
        if cell not in lm_cells.REGISTRY:
            cell, args = "perceive.describe", {}
        lm_cells.call_cell(st, cell, args or {})
        trace.append((cell, json.dumps(args, sort_keys=True)))
        ndec += 1
    arc.close_scorecard(card)
    return trace
