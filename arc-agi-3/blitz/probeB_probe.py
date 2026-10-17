"""probeB_ probe scripts: figure out level-0 mechanics of ARC-AGI-3 games by direct probing.

Usage:
  python probeB_probe.py info <game>                 # tags, actions, ascii frame, color hist
  python probeB_probe.py comps <game>                # component list in salience order
  python probeB_probe.py clickrep <game> [seed]      # click each comp rep on fresh env; report effect
  python probeB_probe.py lattice <game> <step>       # click lattice points on fresh env; report effect
  python probeB_probe.py keysw <game> [seed]        # sweep each keyboard action; report effect
  python probeB_probe.py seq <game> "<plan>"        # replay action plan on fresh env; report
Plan format: C x y (click), K id (keyboard action), R (reset), e.g. "C 32 32 K 1 K 1"

NEVER reads /srv/arc3-claude/v6run/held_v6.jsonl. Only public games via Arcade.
"""
import argparse, json, logging, os, sys
sys.path.insert(0, "/srv/arc3-claude/v6run")
from arc_agi import Arcade
from arcengine import GameAction, GameState
import numpy as np

BASE = "/srv/arc3-claude"
lg = logging.getLogger("quiet"); lg.addHandler(logging.NullHandler()); lg.propagate = False
logging.getLogger("arc_agi").setLevel(logging.WARNING)
arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files", recordings_dir=f"{BASE}/recordings")

def make(game, seed=0):
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard()
    return arc.make(info.game_id, seed=seed, scorecard_id=card), info

def frame_of(resp):
    return np.asarray(resp.frame[-1], dtype=np.uint8)

CHARS = " .:-=+*#%@abcdefgh"
def ascii_frame(f):
    return "\n".join("".join(CHARS[min(int(v), 15)] for v in row) for row in f)

def hist(f):
    vals, cnts = np.unique(f, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}

def do_info(game):
    env, info = make(game)
    r = env.observation_space
    f = frame_of(r)
    print("game", game, "tags", info.tags)
    print("available_actions", list(r.available_actions))
    print("state", r.state, "levels_completed", r.levels_completed)
    print("n_frames", len(r.frame))
    print("hist", hist(f))
    print(ascii_frame(f))

def do_comps(game):
    env, info = make(game)
    f = frame_of(env.observation_space)
    sys.path.insert(0, "/srv/arc3-claude/v6run")
    from cells import components, salience_order
    vals, cnts = np.unique(f, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    print("bg", bg)
    for c in salience_order(components(f), bg):
        print(f"color={c['color']} size={c['size']} rep=({c['x']},{c['y']}) bbox={c['bbox']}")

def probe_click(game, x, y, seed=0):
    env, _ = make(game, seed)
    r0 = env.observation_space
    f0 = frame_of(r0); l0 = r0.levels_completed
    r = env.step(GameAction.ACTION6, data={"x": x, "y": y})
    f1 = frame_of(r)
    changed = int((f0 != f1).sum())
    return dict(changed=changed, level_up=(r.levels_completed - l0), over=(r.state == GameState.GAME_OVER),
                avail=list(r.available_actions), level=r.levels_completed)

def do_clickrep(game, seed):
    sys.path.insert(0, "/srv/arc3-claude/v6run")
    from cells import components, salience_order
    env, _ = make(game, seed)
    f = frame_of(env.observation_space)
    vals, cnts = np.unique(f, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    for c in salience_order(components(f), bg):
        res = probe_click(game, c["x"], c["y"], seed)
        print(f"click ({c['x']},{c['y']}) color={c['color']} size={c['size']} -> {res}")

def do_lattice(game, step):
    off = step // 2
    pts = [(x, y) for y in range(off, 64, step) for x in range(off, 64, step)]
    for x, y in pts:
        res = probe_click(game, x, y)
        if res["changed"] or res["level_up"] or res["over"]:
            print(f"click ({x},{y}) -> {res}")
    print("done", len(pts), "clicks")

def probe_key(game, aid, seed=0):
    env, _ = make(game, seed)
    r0 = env.observation_space
    f0 = frame_of(r0); l0 = r0.levels_completed
    r = env.step(GameAction.from_id(aid))
    f1 = frame_of(r)
    changed = int((f0 != f1).sum())
    return dict(changed=changed, level_up=(r.levels_completed - l0), over=(r.state == GameState.GAME_OVER),
                avail=list(r.available_actions), level=r.levels_completed)

def do_keysw(game, seed):
    env, _ = make(game, seed)
    for aid in list(env.observation_space.available_actions):
        if aid in (0, 6):
            continue
        res = probe_key(game, aid, seed)
        print(f"key {aid} -> {res}")

def do_seq(game, plan, seed=0):
    env, _ = make(game, seed)
    r = env.observation_space
    f0 = frame_of(r)
    for tok in plan.split(","):
        tok = tok.strip().split()
        if tok[0] == "C":
            r = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        elif tok[0] == "K":
            r = env.step(GameAction.from_id(int(tok[1])))
        elif tok[0] == "R":
            r = env.step(GameAction.RESET)
        print("after", " ".join(tok), "level", r.levels_completed, "state", r.state,
              "changed", int((frame_of(r) != f0).sum()), "avail", list(r.available_actions))
    f1 = frame_of(r)
    print(ascii_frame(f1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("game")
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args()
    {"info": do_info, "comps": do_comps}.get(a.cmd, None) and None
    if a.cmd == "info": do_info(a.game)
    elif a.cmd == "comps": do_comps(a.game)
    elif a.cmd == "clickrep": do_clickrep(a.game, int(a.arg or 0))
    elif a.cmd == "lattice": do_lattice(a.game, int(a.arg or 8))
    elif a.cmd == "keysw": do_keysw(a.game, int(a.arg or 0))
    elif a.cmd == "seq": do_seq(a.game, a.arg, 0)

main()
