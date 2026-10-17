"""Verify the lf52 ghost-chain grammar with the ghost_follow cell and emit clean
chained traces. Mechanic: click token -> ghost marker appears -> click ghost ->
token teleports -> next ghost. Chain one link per macro call.
Appends to lm/traces_grammar.jsonl
"""
import json
import logging
import os
import re
import sys

sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from cells import GameState, call_cell

from arc_agi import Arcade
from arcengine import GameState as GState

BASE = "/srv/arc3-claude"
OUT = os.path.join(BASE, "v6run", "model", "lm", "traces_grammar.jsonl")
GHOST_RE = re.compile(r"ghost marker at \((\d+),(\d+)\)")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    max_links = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    lg = logging.getLogger("q")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                 recordings_dir=f"{BASE}/recordings")
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == "lf52")
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, 3000)
    traces = []

    def snap():
        return call_cell(st, "perceive.describe", {})

    # link 0: click the most salient small token (skip background) to reveal
    # the first ghost
    s = snap()
    obs = call_cell(st, "perceive.objects", {})
    traces.append({"state": s,
                   "think": "Pick-and-place game. Clicking a token to reveal where it wants to go.",
                   "call": {"cell": "perceive.objects", "args": {}},
                   "obs": obs[:200]})
    # parse small non-background objects: "OBJn: color C bbox (x0,y0)-(x1,y1)"
    toks = []
    for m in re.finditer(r"OBJ\d+: color (\d+) bbox \((\d+),(\d+)\)-\((\d+),(\d+)\)", obs):
        c, x0, y0, x1, y1 = map(int, m.groups())
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if area <= 64:
            toks.append((area, (x0 + x1) // 2, (y0 + y1) // 2))
    if not toks:
        print("no small token found", flush=True)
        return
    toks.sort(reverse=True)
    # try tokens until one reveals a real ghost (skip noise artifacts);
    # rarer colors first (the active token is often the odd one out)
    from collections import Counter
    # (keep area order; just try more candidates)
    cx = cy = None
    for attempt in range(min(10, len(toks))):
        tx, ty = toks[attempt][1], toks[attempt][2]
        s_try = snap()
        obs_try = call_cell(st, "act.ghost_follow", {"x": tx, "y": ty})
        m2 = GHOST_RE.search(obs_try)
        if m2 and not (int(m2.group(1)) == 0 and int(m2.group(2)) == 0):
            cx, cy = tx, ty
            s, obs = s_try, obs_try
            think = f"Clicking the token at ({cx},{cy}) to reveal the ghost."
            traces.append({"state": s, "think": think,
                           "call": {"cell": "act.ghost_follow",
                                    "args": {"x": cx, "y": cy}},
                           "obs": obs})
            print(f"link0: {obs}", flush=True)
            break
        # no ghost: reset to try the next token cleanly
        call_cell(st, "act.reset", {})
    if cx is None:
        print("no token revealed a ghost", flush=True)
        return

    links = 0
    while links < max_links:
        if st.resp.state == GState.WIN or st.level > 0:
            break
        m = GHOST_RE.search(obs)
        if not m:
            print("chain ended: no ghost", flush=True)
            break
        gx, gy = int(m.group(1)), int(m.group(2))
        s = snap()
        think = (f"A ghost marker appeared at ({gx},{gy}). Ghosts mark the next "
                 f"target — following it.")
        obs = call_cell(st, "act.ghost_follow", {"x": gx, "y": gy})
        traces.append({"state": s, "think": think,
                       "call": {"cell": "act.ghost_follow",
                                "args": {"x": gx, "y": gy}},
                       "obs": obs})
        print(f"link{links+1}: {obs[:120]}", flush=True)
        links += 1

    sc = arc.close_scorecard(card)
    run = sc.environments[0].runs[-1].model_dump() if sc and sc.environments else {}
    levels = run.get("levels_completed")
    print(f"lf52 ghost probe: links={links} levels={levels} "
          f"score={run.get('score')}", flush=True)
    if levels and levels > 0:
        with open(OUT, "a") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        print(f"APPENDED {len(traces)} grammar traces", flush=True)
    else:
        print("ghost chain did NOT solve; traces not appended", flush=True)


if __name__ == "__main__":
    main()
