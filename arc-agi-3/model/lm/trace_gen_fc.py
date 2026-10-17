"""First-contact discovery traces (Track 1, attempt #3).

For each ALLOWED dev game x seed, run the training-free directed
state-action graph explorer (v6run/explorer.py) and record every decision
as a v3-schema trace doc: {"state","think","call","obs","episode"}.

- state/obs use the exact describe() format from trace_gen_w2.py
  ("LVL0 STEP0 OBJ1 c5@0,0-63,62a4032 AVAIL[3, 4, 6, 7]").
- Raw explorer actions map to act.* CALLs:
    keyboard id i        -> {"cell":"act.try","args":{"action":i}}
    click (6,x,y)        -> {"cell":"act.click_at","args":{"x":x,"y":y}}
    RESET                -> {"cell":"act.reset","args":{}}
- THINK is a game-agnostic discovery narrative (no game names anywhere).
  Words restricted to vocab.json's word list so <unk> stays ~0.
- episode = "<game>_s<seed>_lvl<l>" (metadata only; render_trace ignores it).

Forbidden transfer games are NEVER referenced here (allowed list only).
held_v6.jsonl is never touched.
"""
import json
import logging
import os
import sys
import importlib.util

import numpy as np

sys.path.insert(0, "/srv/arc3-claude/v6run")  # explorer's `import cells` -> v6run/cells.py

_spec = importlib.util.spec_from_file_location(
    "lm_cells", "/srv/arc3-claude/v6run/model/lm/cells.py")
lm_cells = importlib.util.module_from_spec(_spec)
sys.modules["lm_cells"] = lm_cells
_spec.loader.exec_module(lm_cells)
_objs = lm_cells._objs

from explorer import Explorer, RESET  # noqa: E402
from arcengine import GameState as GState  # noqa: E402
from arc_agi import Arcade  # noqa: E402

BASE = "/srv/arc3-claude"
OUT = os.path.join(BASE, "v6run", "model", "lm", "traces_fc.jsonl")

# ALLOWED dev games only. Transfer set (ar25 cd82 lp85 m0r0 r11l sb26 sp80
# su15 tn36 tr87 vc33 wa30) is forbidden and never appears here.
GAMES = ["bp35", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
         "ls20", "re86", "s5i5", "sc25", "sk48", "tu93"]
SEEDS = [0]  # env seed does not vary level-0 starts; diversity comes from
SPLIT_AT = 220  # exploration depth instead (steps 0..219 -> _s0, 220.. -> _s1)
BUDGET = 440  # explorer steps per game; ~5.7k docs max

lg = logging.getLogger("q")
lg.addHandler(logging.NullHandler())
lg.propagate = False
logging.getLogger("arc_agi").setLevel(logging.WARNING)


def describe(frame, prev, avail, level, steps):
    """Exact v3 state-text format (copied from trace_gen_w2.py)."""
    f = np.asarray(frame, dtype=np.uint8)
    objs = _objs(f)[:6]
    parts = [f"c{o['c']}@{o['x0']},{o['y0']}-{o['x1']},{o['y1']}a{o['area']}"
             for o in objs]
    chg = ""
    if prev is not None:
        d = f != np.asarray(prev, dtype=np.uint8)
        n = int(d.sum())
        if n:
            ys, xs = np.nonzero(d)
            chg = f" chg{n}@({xs.min()},{ys.min()},{xs.max()},{ys.max()})"
    return (f"LVL{level} STEP{steps} OBJ{len(objs)} " + " ".join(parts) +
            chg + f" AVAIL{sorted(int(x) for x in avail)}")


def count_phrase(n):
    """In-vocab phrasing of the untested-action count (digits 0-9 only)."""
    if n == 0:
        return "No new actions here."
    if 1 <= n <= 9:
        return f"{n} new actions here."
    return "new actions here."


def map_act(act):
    if act == RESET:
        return {"cell": "act.reset", "args": {}}
    if act[0] == 6:
        return {"cell": "act.click_at",
                "args": {"x": int(act[1]), "y": int(act[2])}}
    return {"cell": "act.try", "args": {"action": int(act[0])}}


def run_game(game, seed, budget):
    info = next(e for e in ARC.get_environments()
                if e.game_id.split("-")[0] == game)
    card = ARC.create_scorecard()
    env = ARC.make(info.game_id, seed=seed, scorecard_id=card)
    ex = Explorer(env, budget, seed=seed)
    docs = []
    lvl_start = {ex.level: 0}

    orig_send, orig_step = ex._send, ex.step_once
    info_box = {}

    def send_rec(act):
        r = orig_send(act)
        info_box["act"] = act
        info_box["resp"] = r
        return r

    ex._send = send_rec

    def step_once_rec():
        g = ex.graph
        cur = ex.cur
        over_b = ex.resp.state == GState.GAME_OVER
        plan_b = bool(ex.plan)
        try:
            untested_b = list(g.untested.get(cur, []))
        except Exception:
            untested_b = []
        fb = ex.cur_f.copy()
        lvl_b = ex.level
        av_b = list(ex.avail)
        used_b = ex.used
        orig_step()
        act = info_box.pop("act", None)
        r = info_box.pop("resp", None)
        if act is None or r is None or not r.frame:
            return  # env hiccup: skip recording this step
        fa = ex.cur_f.copy()
        lvl_a = ex.level
        av_a = list(ex.avail)
        win = ex.resp.state == GState.WIN

        if act == RESET:
            if over_b:
                think = "Stuck. Resetting the level to try fresh."
            elif plan_b:
                think = "No new action here. Moving to the next spot."
            else:
                think = "No progress here. Resetting the level to try fresh."
        elif plan_b:
            think = "No new action here. Moving to the next spot."
        else:
            n = len(untested_b)
            cp = count_phrase(n)
            if act[0] == 6:
                think = (f"Probing click at ({act[1]},{act[2]}). "
                         f"This spot is new. I do not know what it do. {cp}")
            else:
                think = (f"Trying action {act[0]}. This action is new. "
                         f"I do not know what it do. {cp}")

        if lvl_a != lvl_b:
            lvl_start[lvl_a] = used_b + 1
            if win:
                think += " This is progress."
            elif act == RESET:
                think += " new level. This is progress."
            elif act[0] == 6:
                think += (f" new level. This is progress. "
                          f"Clicking ({act[1]},{act[2]}).")
            else:
                think += (f" new level. This is progress. "
                          f"Action {act[0]}.")

        docs.append({
            "state": describe(fb, None, av_b, lvl_b, used_b),
            "think": think,
            "call": map_act(act),
            "obs": describe(fa, fb, av_a, lvl_a, used_b + 1),
            "episode": f"{game}_s{0 if len(docs) < SPLIT_AT else 1}_lvl{lvl_b}",
        })

    ex.step_once = step_once_rec
    try:
        ex.run()
    except Exception as e:
        print(f"AGENT ERROR {game} s{seed}: {e}", file=sys.stderr)
    finally:
        ex._send = orig_send
    ARC.close_scorecard(card)
    return docs


def main():
    global ARC
    ARC = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                 recordings_dir=f"{BASE}/recordings")
    n_all = 0
    with open(OUT, "w") as f:
        for g in GAMES:
            for s in SEEDS:
                docs = run_game(g, s, BUDGET)
                kinds = {}
                eplen = {}
                for d in docs:
                    kinds[d["call"]["cell"]] = kinds.get(d["call"]["cell"], 0) + 1
                    ep = d["episode"]
                    eplen[ep] = eplen.get(ep, 0) + 1
                    f.write(json.dumps(d) + "\n")
                n_all += len(docs)
                meplen = (sum(eplen.values()) / len(eplen)) if eplen else 0
                print(json.dumps({"game": g, "seed": s, "docs": len(docs),
                                  "kinds": kinds,
                                  "mean_episode_len": round(meplen, 1)}),
                      flush=True)
    print(f"WROTE {n_all} traces to {OUT}", flush=True)


if __name__ == "__main__":
    main()
