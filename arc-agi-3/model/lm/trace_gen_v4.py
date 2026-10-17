"""v4 trace generator: MINIMAL-action play only (efficiency distillation).

Sources (all dev games; held_v6.jsonl never touched):
  A. s5i5 dial grammar, seeds 0..5: perceive.objects + 2 repeat_click = 13 actions
  B. lf52 ghost chain, seeds 0..5: perceive.objects + 8 ghost_follow = 8 actions
  C. ls20 blitz8 L0 (painter-demotion, 64 actions): translated action stream
  D. cell primer: one canonical row per cell (vocab + format coverage)
Efficiency weighting: each episode's rows duplicated round(64/episode_actions)x.
Writes lm/traces_v4.jsonl
"""
import json, logging, os, random, re, sys

sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
sys.path.insert(0, "/srv/arc3-claude/v6run")
sys.path.insert(0, "/srv/arc3-claude/v6run/model")
sys.path.insert(0, "/srv/arc3-claude/v6run/blitz")
from cells import GameState, call_cell
from arc_agi import Arcade
from arcengine import GameState as GState

BASE = "/srv/arc3-claude"
OUT = os.path.join(BASE, "v6run", "model", "lm", "traces_v4.jsonl")
GHOST_RE = re.compile(r"ghost marker at \((\d+),(\d+)\)")
DIALS = [((46, 21), 7), ((24, 45), 6)]

lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
logging.getLogger("arc_agi").setLevel(logging.WARNING)


def make_arc():
    return Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                  recordings_dir="/tmp/v4gen_rec")


def snap(st):
    return call_cell(st, "perceive.describe", {})


def row(state, think, cell, args, obs):
    return {"state": state, "think": think,
            "call": {"cell": cell, "args": args}, "obs": obs[:220]}


# ---------------- A. s5i5 dial grammar ----------------
def gen_s5i5(arc, seeds):
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == "s5i5")
    eps = []
    for seed in seeds:
        card = arc.create_scorecard(); env = arc.make(info.game_id, seed=seed, scorecard_id=card)
        st = GameState(env, 3000); rows = []
        s0 = snap(st)
        o0 = call_cell(st, "perceive.objects", {})
        rows.append(row(s0, "Efficient play: two dial tokens on this level. "
                           "Dials need repeated clicks, not single toggles — no probing wasted.",
                        "perceive.objects", {}, o0))
        ok = True
        for (x, y), n in DIALS:
            s = snap(st)
            obs = call_cell(st, "act.repeat_click", {"x": x, "y": y, "n": n})
            rows.append(row(s, f"Efficient play: turning the dial at ({x},{y}) "
                               f"with {n} repeated clicks in one macro call.",
                            "act.repeat_click", {"x": x, "y": y, "n": n}, obs))
            if st.resp.state == GState.WIN or st.level > 0:
                break
        else:
            ok = False
        if st.level == 0: ok = False
        n_act = st.steps_used
        arc.close_scorecard(card)
        if ok:
            eps.append((rows, n_act)); print(f"s5i5 seed {seed}: {n_act} actions OK", flush=True)
        else:
            print(f"s5i5 seed {seed}: FAILED", flush=True)
    return eps


# ---------------- B. lf52 ghost chain ----------------
def token_candidates(obs):
    toks = []
    for m in re.finditer(r"OBJ\d+: color (\d+) bbox \((\d+),(\d+)\)-\((\d+),(\d+)\)", obs):
        c, x0, y0, x1, y1 = map(int, m.groups())
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if area <= 64:
            toks.append((area, (x0 + x1) // 2, (y0 + y1) // 2))
    toks.sort(reverse=True)
    return toks


def gen_lf52(arc, seeds):
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == "lf52")
    eps = []
    for seed in seeds:
        # find working token on a scratch env (as validated: attempt index 5)
        card = arc.create_scorecard(); env = arc.make(info.game_id, seed=seed, scorecard_id=card)
        st = GameState(env, 3000)
        s0 = snap(st); o0 = call_cell(st, "perceive.objects", {})
        toks = token_candidates(o0)
        arc.close_scorecard(card)
        good = None
        for attempt in range(min(10, len(toks))):
            card2 = arc.create_scorecard(); env2 = arc.make(info.game_id, seed=seed, scorecard_id=card2)
            st2 = GameState(env2, 3000)
            tx, ty = toks[attempt][1], toks[attempt][2]
            o2 = call_cell(st2, "act.ghost_follow", {"x": tx, "y": ty})
            m = GHOST_RE.search(o2)
            if m and not (int(m.group(1)) == 0 and int(m.group(2)) == 0):
                good = (tx, ty)
            arc.close_scorecard(card2)
            if good: break
        if not good:
            print(f"lf52 seed {seed}: no token", flush=True); continue
        # clean minimal run on the known-good token
        card = arc.create_scorecard(); env = arc.make(info.game_id, seed=seed, scorecard_id=card)
        st = GameState(env, 3000); rows = []
        s0 = snap(st); o0 = call_cell(st, "perceive.objects", {})
        rows.append(row(s0, "Efficient play: pick-and-place level. Clicking the token "
                           "once to reveal where it wants to go — no blind probing.",
                        "perceive.objects", {}, o0))
        tx, ty = good
        s = snap(st)
        obs = call_cell(st, "act.ghost_follow", {"x": tx, "y": ty})
        rows.append(row(s, f"Efficient play: clicking the token at ({tx},{ty}) to reveal the ghost.",
                        "act.ghost_follow", {"x": tx, "y": ty}, obs))
        links = 0
        while links < 12 and st.level == 0 and st.resp.state != GState.WIN:
            m = GHOST_RE.search(obs)
            if not m: break
            gx, gy = int(m.group(1)), int(m.group(2))
            s = snap(st)
            obs = call_cell(st, "act.ghost_follow", {"x": gx, "y": gy})
            rows.append(row(s, f"Efficient play: ghost marker at ({gx},{gy}) marks the next "
                               "target — following it in one macro call.",
                            "act.ghost_follow", {"x": gx, "y": gy}, obs))
            links += 1
        n_act = st.steps_used
        arc.close_scorecard(card)
        if st.level > 0:
            eps.append((rows, n_act)); print(f"lf52 seed {seed}: {n_act} actions OK", flush=True)
        else:
            print(f"lf52 seed {seed}: FAILED", flush=True)
    return eps


# ---------------- C. ls20 blitz8 L0 translated ----------------
def gen_ls20(arc):
    from blitz_agent import AGENTS
    import blitz_w2  # noqa: registers blitz8
    import numpy as np
    from trace_gen_w2 import translate
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == "ls20")

    def aid_of(action):
        if action == GameAction.RESET: return 0
        if action == GameAction.ACTION6: return 6
        return int(action.value)
    from arcengine import GameAction
    card = arc.create_scorecard(); env = arc.make(info.game_id, seed=0, scorecard_id=card)
    rec = []
    cur = {"resp": env.observation_space}
    orig_step = env.step
    def rec_step(action, data=None):
        fb = np.asarray(cur["resp"].frame[-1], dtype=np.uint8).copy()
        av = list(cur["resp"].available_actions)
        aid = aid_of(action)
        r = orig_step(action) if data is None else orig_step(action, data=data)
        fa = np.asarray(r.frame[-1], dtype=np.uint8).copy()
        rec.append((fb, aid, -1, -1, fa, int(r.levels_completed), list(r.available_actions)))
        cur["resp"] = r
        return r
    env.step = rec_step
    ex = AGENTS["blitz8"](env, 3000, seed=0, log=lambda *a: None,
                          flags=("pr", "nm", "ck", "gv", "d6"), tags=info.tags)
    try: ex.run()
    except Exception as e: print("blitz8 err", e, flush=True)
    arc.close_scorecard(card)
    # truncate at first L0 completion
    cut = next((i + 1 for i, r in enumerate(rec) if r[5] > 0), len(rec))
    rec = rec[:cut]
    rng = random.Random(7)
    traces = translate(rec, rng)
    # efficiency framing on the translated rows
    for t in traces:
        t["think"] = "Efficient play (keyboard): " + t["think"][0].lower() + t["think"][1:]
    n_act = len(rec)
    print(f"ls20: {len(traces)} trace rows from {n_act} actions", flush=True)
    return [([dict(t) for t in traces], n_act)]


# ---------------- D. cell primer ----------------
PRIMER = [
    ("perceive.describe", {}, "Efficient play: scanning the scene before acting costs nothing."),
    ("perceive.objects", {}, "Efficient play: listing objects to pick exact targets."),
    ("perceive.diff", {}, "Efficient play: checking what changed since the last look."),
    ("perceive.hud", {}, "Efficient play: reading the top rows for counters."),
    ("act.try", {"action": 1}, "Efficient play: trying one keyboard action."),
    ("act.sweep", {}, "Efficient play: mapping all unknown keyboard actions at once instead of guessing one by one."),
    ("act.click_at", {"x": 32, "y": 32}, "Efficient play: single precise click."),
    ("act.pick_drop", {"x1": 10, "y1": 10, "x2": 20, "y2": 20}, "Efficient play: moving a piece in one two-click macro."),
    ("act.repeat_click", {"x": 46, "y": 21, "n": 2}, "Efficient play: dial macro — repeated clicks in one call."),
    ("act.ghost_follow", {"x": 30, "y": 30}, "Efficient play: ghost-chain macro — click and locate the next marker."),
    ("act.reset", {}, "Efficient play: resetting only when truly stuck."),
]


def gen_primer(arc):
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == "s5i5")
    card = arc.create_scorecard(); env = arc.make(info.game_id, seed=0, scorecard_id=card)
    st = GameState(env, 3000)
    rows = []
    for cell, args, think in PRIMER:
        s = snap(st)
        obs = call_cell(st, cell, dict(args))
        rows.append(row(s, think, cell, dict(args), obs))
    arc.close_scorecard(card)
    return [([dict(r) for r in rows], 11)]


def main():
    arc = make_arc()
    seeds = [0, 1, 2, 3, 4, 5]
    eps = gen_s5i5(arc, seeds) + gen_lf52(arc, seeds) + gen_ls20(arc) + gen_primer(arc)
    # efficiency weighting: duplicate rows round(64/episode_actions)x
    out_rows = []
    for rows, n_act in eps:
        dup = max(1, round(64 / max(1, n_act)))
        for _ in range(dup):
            out_rows.extend(rows)
    random.Random(0).shuffle(out_rows)
    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    n_ep = len(eps)
    print(f"WROTE {len(out_rows)} rows from {n_ep} episodes -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
