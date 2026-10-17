"""probeB_bfs.py: BFS over click(/key) actions from level start, replaying from root.
Finds a winning action sequence for level 0.

Usage: python probeB_bfs.py <game> [--depth 6] [--seed 0] [--keys] [--lattice N]
Actions: component-rep clicks (size>=3); with --keys also keyboard actions; with
--lattice N adds an N-step click lattice.
Prints winning path or frontier stats.
"""
import argparse, hashlib, logging, os, sys
from collections import deque
sys.path.insert(0, "/srv/arc3-claude/v6run")
from arc_agi import Arcade
from arcengine import GameAction, GameState
import numpy as np

BASE = "/srv/arc3-claude"
lg = logging.getLogger("quiet"); lg.addHandler(logging.NullHandler()); lg.propagate = False
logging.getLogger("arc_agi").setLevel(logging.WARNING)
arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files", recordings_dir=f"{BASE}/recordings")
sys.path.insert(0, "/srv/arc3-claude/v6run")
from cells import components, salience_order

def key_of(f):
    return hashlib.blake2b(np.ascontiguousarray(f).tobytes(), digest_size=10).digest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game"); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--keys", action="store_true")
    ap.add_argument("--lattice", type=int, default=0); ap.add_argument("--nodes", type=int, default=3000)
    a = ap.parse_args()
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == a.game)

    def new_env():
        card = arc.create_scorecard()
        return arc.make(info.game_id, seed=a.seed, scorecard_id=card)

    env = new_env()
    r0 = env.observation_space
    f0 = np.asarray(r0.frame[-1], dtype=np.uint8)
    vals, cnts = np.unique(f0, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    acts = []
    seen = set()
    for c in salience_order(components(f0), bg):
        if c["size"] >= 3 and (c["x"], c["y"]) not in seen:
            seen.add((c["x"], c["y"]))
            acts.append(("C", c["x"], c["y"]))
    if a.lattice:
        off = a.lattice // 2
        for y in range(off, 64, a.lattice):
            for x in range(off, 64, a.lattice):
                if (x, y) not in seen:
                    seen.add((x, y)); acts.append(("C", x, y))
    if a.keys:
        for i in list(r0.available_actions):
            if i not in (0, 6):
                acts.append(("K", i))
    print(f"{len(acts)} actions: {acts[:20]}")

    def run_path(path):
        env = new_env()
        r = env.observation_space
        for act in path:
            if act[0] == "C":
                r = env.step(GameAction.ACTION6, data={"x": act[1], "y": act[2]})
            else:
                r = env.step(GameAction.from_id(act[1]))
            if r is None or not r.frame:
                return None, None
        return r, np.asarray(r.frame[-1], dtype=np.uint8)

    root_k = key_of(f0)
    seen_states = {root_k}
    q = deque([()])
    nodes = 0
    while q and nodes < a.nodes:
        path = q.popleft()
        if len(path) >= a.depth:
            continue
        for act in acts:
            p2 = path + (act,)
            r, f = run_path(p2)
            nodes += 1
            if r is None:
                continue
            if r.levels_completed > 0 or r.state == GameState.WIN:
                print("WIN path:", " ".join(f"{x[0]}{x[1:]}" for x in p2))
                return
            k = key_of(f)
            if k not in seen_states:
                seen_states.add(k)
                q.append(p2)
    print(f"no win. nodes={nodes} states={len(seen_states)} depth_reached={a.depth}")

main()
