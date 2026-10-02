"""ARC-AGI-3 game runner for policy cells (server, OFFLINE mode: local game files only, no network, no scorecard upload).

  play(source, params, game, budget)      -> result dict of one policy run (levels, steps per level, stall digest, error)
  controls(source, params, game, budget)  -> the same policy with zeroed tensors, 3 random draws, and BLIND (frame zeroed)
  probe(game)                             -> dynamics of the game from its first frame, for the maker prompt (no policy involved)
  random_agent(game, budget, seed)        -> the seeded random-action baseline at the same budget

The harness, not the policy, issues RESET after GAME_OVER (it costs one action, as in the official harness). An invalid action id, a crash or a timeout ends
the run (levels completed so far still count; the error is reported).
"""
import logging
import os
import random
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arc3cell as C  # noqa: E402

BASE = os.environ.get("ARC3_DIR", "/srv/arc3-claude")
HEX = ".123456789abcdef"
_arc = None


def arcade():
    global _arc
    if _arc is None:
        from arc_agi import Arcade, OperationMode
        quiet = logging.getLogger("arc3-quiet")
        quiet.addHandler(logging.NullHandler())
        quiet.propagate = False
        logging.getLogger("arc_agi").setLevel(logging.WARNING)
        _arc = Arcade(logger=quiet, environments_dir=BASE + "/environment_files", recordings_dir=tempfile.mkdtemp(prefix="arc3rec-"),
                      operation_mode=OperationMode.OFFLINE)
    return _arc


def games():
    return sorted(e.game_id.split("-")[0] for e in arcade().get_environments())


def info_of(game):
    return next(e for e in arcade().get_environments() if e.game_id.split("-")[0] == game)


def new_env(game, seed=0):
    arc = arcade()
    card = arc.create_scorecard()
    return arc, card, arc.make(info_of(game).game_id, seed=seed, scorecard_id=card)


def last_frame(resp):
    return np.asarray(resp.frame[-1], dtype=np.uint8)


def dump(frame):
    return "\n".join("".join(HEX[v] for v in row) for row in np.asarray(frame))


def _do(env, action, x=0, y=0):
    from arcengine import GameAction
    if action == 6:
        return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
    return env.step(GameAction.from_id(int(action)))


def play(source, params, game, budget=1500, seed=0, blind=False, wall=240.0, keep_trace=False):
    from arcengine import GameAction, GameState
    arc, card, env = new_env(game, seed)
    n_levels = len(info_of(game).baseline_actions)
    policy = C.Policy(source, params)
    resp = env.observation_space
    out = {"game": game, "seed": seed, "budget": budget, "n_levels": n_levels, "levels": 0, "used": 0, "resets": 0, "noops": 0, "error": policy.error,
           "level_steps": [], "state": "NOT_FINISHED", "distinct": 0, "action_hist": {}}
    seen, trace, started = set(), [], time.time()
    prev = last_frame(resp)
    zero = np.zeros((64, 64), dtype=np.uint8)
    while out["used"] < budget and resp is not None and resp.state != GameState.WIN and not out["error"]:
        if time.time() - started > wall:
            out["error"] = "wall-clock limit %ds" % wall
            break
        if resp.state == GameState.GAME_OVER:
            resp = env.step(GameAction.RESET)
            out["used"] += 1
            out["resets"] += 1
            if resp is None:
                break
            prev = last_frame(resp)
            continue
        frame = last_frame(resp)
        key = C.frame_key(frame)
        seen.add(key)
        if blind:
            key = C.frame_key(zero)  # blind hides the frame AND its key, otherwise a state-graph explorer could still see the game through the digest
        info = {"levels": int(resp.levels_completed), "resets": out["resets"], "step": out["used"], "key": key}
        avail = [a for a in resp.available_actions if a != 0]
        action, x, y, err = policy.step(zero if blind else frame, avail, info)
        if err:
            out["error"] = err
            break
        if action not in avail:
            out["error"] = "illegal action %r (available %s)" % (action, avail)
            break
        resp = _do(env, action, x, y)
        out["used"] += 1
        out["action_hist"][str(action)] = out["action_hist"].get(str(action), 0) + 1
        if resp is None:
            break
        new = last_frame(resp)
        if np.array_equal(new, prev):
            out["noops"] += 1
        if int(resp.levels_completed) > out["levels"]:
            out["levels"] = int(resp.levels_completed)
            out["level_steps"].append(out["used"])
        if keep_trace and len(trace) < 400:
            trace.append((action, x, y))
        prev = new
    policy.close()
    out["state"] = resp.state.name if resp is not None else "NONE"
    out["distinct"] = len(seen)
    if resp is not None:
        out["final_frame"] = dump(last_frame(resp))
    try:
        card_result = arc.close_scorecard(card)
        run = card_result.environments[0].runs[-1].model_dump() if card_result and card_result.environments else {}
        out["score"] = run.get("score")
    except Exception:  # noqa: BLE001
        out["score"] = None
    if keep_trace:
        out["trace"] = trace
    return out


def controls(source, params, game, budget=1500, seed=0):
    """Matched controls at the same positions and budget: zeroed tensors, three random draws, and the cell run blind."""
    res = {"zero": play(source, C.zeros_like(params), game, budget, seed)}
    res["random"] = [play(source, C.random_like(params, s), game, budget, seed) for s in (11, 12, 13)]
    res["blind"] = play(source, params, game, budget, seed, blind=True)
    return res


def random_agent(game, budget=1500, seed=0):
    from arcengine import GameAction, GameState
    arc, card, env = new_env(game, seed)
    rng = random.Random(seed)
    resp = env.observation_space
    used, levels = 0, 0
    while used < budget and resp is not None and resp.state != GameState.WIN:
        if resp.state == GameState.GAME_OVER:
            resp = env.step(GameAction.RESET)
        else:
            a = rng.choice([i for i in resp.available_actions if i != 0])
            resp = _do(env, a, rng.randrange(64), rng.randrange(64))
        used += 1
        if resp is not None:
            levels = max(levels, int(resp.levels_completed))
    return {"game": game, "levels": levels, "used": used, "n_levels": len(info_of(game).baseline_actions)}


def components(frame, background=None):
    """4-connected same-colour components, largest first: [(colour, area, (y0, x0, y1, x1))]."""
    frame = np.asarray(frame)
    h, w = frame.shape
    seen = np.zeros((h, w), bool)
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y, x]:
                continue
            colour = int(frame[y, x])
            stack, area, y0, x0, y1, x1 = [(y, x)], 0, y, x, y, x
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                area += 1
                y0, x0, y1, x1 = min(y0, cy), min(x0, cx), max(y1, cy), max(x1, cx)
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and frame[ny, nx] == colour:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((colour, area, (y0, x0, y1, x1)))
    return sorted(out, key=lambda c: -c[1])


def diff_summary(a, b):
    d = np.asarray(a) != np.asarray(b)
    if not d.any():
        return {"changed": 0}
    ys, xs = np.nonzero(d)
    fro = sorted({int(v) for v in np.asarray(a)[d]})
    to = sorted({int(v) for v in np.asarray(b)[d]})
    return {"changed": int(d.sum()), "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())], "colours_from": fro, "colours_to": to}


def probe(game, seed=0, max_clicks=10):
    """Dynamics from the first frame, each probe on a fresh env: what every simple action changes, what a click on each of the largest components changes,
    and what repeating each action 3 times does. Nothing here involves a policy or a level solution."""
    from arcengine import GameState
    arc, card, env = new_env(game, seed)
    resp = env.observation_space
    first = last_frame(resp)
    info = info_of(game)
    avail = [a for a in resp.available_actions if a != 0]
    out = {"game": game, "tags": info.tags, "available": avail, "n_levels": len(info.baseline_actions), "human_baseline_actions": list(info.baseline_actions),
           "first_frame": dump(first), "colours": {int(k): int(v) for k, v in zip(*np.unique(first, return_counts=True))}, "actions": {}, "clicks": []}
    for a in avail:
        if a == 6:
            continue
        _, _, e = new_env(game, seed)
        r = e.observation_space
        steps = []
        prev = last_frame(r)
        for _ in range(3):
            r = _do(e, a)
            if r is None:
                break
            cur = last_frame(r)
            steps.append(diff_summary(prev, cur))
            prev = cur
            if r.state == GameState.GAME_OVER:
                steps[-1]["game_over"] = True
                break
        out["actions"][str(a)] = steps
    if 6 in avail:
        comps = [c for c in components(first) if c[1] >= 3][:max_clicks]
        for colour, area, (y0, x0, y1, x1) in comps:
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            _, _, e = new_env(game, seed)
            r = _do(e, 6, cx, cy)
            if r is None:
                continue
            out["clicks"].append({"x": cx, "y": cy, "on_colour": colour, "component_area": area, **diff_summary(first, last_frame(r))})
    return out
