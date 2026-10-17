"""Verify the s5i5 dial grammar with the new cells and emit clean grammar traces.
Wave-2 finding: tokens are dials — click (46,21)x7 then (24,45)x6.
Appends to lm/traces_grammar.jsonl
"""
import json
import logging
import os
import sys

sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from cells import GameState, call_cell

from arc_agi import Arcade
from arcengine import GameState as GState

BASE = "/srv/arc3-claude"
OUT = os.path.join(BASE, "v6run", "model", "lm", "traces_grammar.jsonl")

DIALS = [((46, 21), 7), ((24, 45), 6)]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    lg = logging.getLogger("q")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                 recordings_dir=f"{BASE}/recordings")
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == "s5i5")
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, 3000)
    traces = []

    def snap():
        return call_cell(st, "perceive.describe", {})

    s0 = snap()
    t0 = ("Level 0. Two dial tokens visible. Dials need repeated clicks, "
          "not single toggles.")
    o0 = call_cell(st, "perceive.objects", {})
    traces.append({"state": s0, "think": t0,
                   "call": {"cell": "perceive.objects", "args": {}},
                   "obs": o0[:200]})
    for (x, y), n in DIALS:
        s = snap()
        think = (f"Token at ({x},{y}) is a dial. Turning it with {n} "
                 f"repeated clicks.")
        obs = call_cell(st, "act.repeat_click", {"x": x, "y": y, "n": n})
        traces.append({"state": s, "think": think,
                       "call": {"cell": "act.repeat_click",
                                "args": {"x": x, "y": y, "n": n}},
                       "obs": obs})
        print(f"dial ({x},{y})x{n}: {obs}", flush=True)
        if st.resp.state == GState.WIN or st.level > 0:
            break
    sc = arc.close_scorecard(card)
    run = sc.environments[0].runs[-1].model_dump() if sc and sc.environments else {}
    levels = run.get("levels_completed")
    print(f"s5i5 dial probe: levels={levels} score={run.get('score')}",
          flush=True)
    if levels and levels > 0:
        with open(OUT, "a") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        print(f"APPENDED {len(traces)} grammar traces", flush=True)
    else:
        print("dial grammar did NOT solve; traces not appended", flush=True)


if __name__ == "__main__":
    main()
