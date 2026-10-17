"""gamma-g0: mechanic-first tiny game-playing agent.

No training: frontier-model game understanding authored directly into a
phase machine. For each token in a click game, in order:
  1. ghost probe -> follow ghost chains (lf52: click token, ghost appears,
     click ghost, token teleports; chain the links)
  2. dial test   -> single clicks; chase while the dial still turns
     (s5i5: tokens are dials, they lock at target; extra clicks no-op)
  3. pick-drop   -> select/drop pairs (ft09, lp85); hot tokens earn bonus
     drop targets
Keyboard games: one sweep, then a novelty walk that demotes painter
actions (never revisit a state) without ever pruning them.
Mixed games: click round first; keyboard iff nothing responded.

Ghost detection is by object diff (no string parsing), so it works even
when the harness passes empty history. Think is terse: every word is
metered against the per-game token budget.
"""

NAME = "gamma-g0"

WEIGHTS = {
    "probe_tokens_max": 4,   # tokens probed per click round
    "ghost_probes": 2,       # ghost_follow probes per token
    "dial_probe_n": 4,       # dial test clicks (chase reuses this bound)
    "pd_targets": 4,         # drop targets per token
    "pd_reselect": False,    # True: select+drop as 2 calls; False: 1 pick_drop
    "hot_bonus_targets": 2,  # extra drop targets for hot tokens
}

CELLS = ["perceive.describe", "act.sweep", "act.try", "act.click_at",
         "act.pick_drop", "act.ghost_follow", "act.reset"]

_S = None

# ---- structural constants (definitions, not tuned knobs) ----
_GHOST_MAX_AREA = 120  # biggest thing still treated as a ghost marker
_GRID_STEP = 16        # fallback click grid spacing
_KB_STUCK = 25         # no-change kb decisions before reset (lenient)
_PAINTER_TRIALS = 10   # trials before an action is judged a painter
_NOOP_TRIALS = 3       # trials before an action is judged a no-op
_MAX_ROUNDS = 4        # click rounds before falling back to grid


def _fresh():
    return {"max_steps": -1, "level": -1, "mode": None, "kb_mode": False,
            "round": 0, "toks": [], "ti": 0, "gpool": 0, "hot_any": False,
            "kb": None, "grid": None, "last": None, "prev_objs": None,
            "_cur_objs": []}


def _new_tok(o):
    return {"x": o["cx"], "y": o["cy"], "phase": "ghost", "gp": 0,
            "chain": False, "seen": set(), "dn": 0,
            "ti": 0, "hot": False, "ntargets": None, "sel": False}


def _key(state):
    return tuple((o["c"], o["x0"], o["y0"], o["x1"], o["y1"])
                 for o in state["objects"])


def _tokens(state, W):
    objs = state["objects"]
    if len(objs) < 2:
        return []
    toks = list(objs[1:])
    toks.sort(key=lambda o: (o["area"], o["cx"], o["cy"]))
    return toks[:W["probe_tokens_max"]]


def _targets(state, s):
    out = []
    for o in state["objects"][1:]:
        if abs(o["cx"] - s["x"]) < 2 and abs(o["cy"] - s["y"]) < 2:
            continue
        out.append(o)
    out.sort(key=lambda o: (o["area"], o["cx"], o["cy"]))
    return [(o["cx"], o["cy"]) for o in out]


def _find_ghost_objs(state, tx, ty):
    prev = _S.get("prev_objs") or []
    pb = set((o["x0"], o["y0"], o["x1"], o["y1"]) for o in prev)
    best = None
    for o in state["objects"]:
        b = (o["x0"], o["y0"], o["x1"], o["y1"])
        if b in pb or o["area"] > _GHOST_MAX_AREA:
            continue
        cx = (o["x0"] + o["x1"]) // 2
        cy = (o["y0"] + o["y1"]) // 2
        d = abs(cx - tx) + abs(cy - ty)
        if best is None or d < best[0]:
            best = (d, cx, cy)
    return (best[1], best[2]) if best else None


def _get_ghost(state, history, tx, ty):
    # Primary: the cell reports the ghost center in its observation.
    # Fallback: object diff (the purity probe passes empty history).
    if history:
        obs = history[-1][3]
        i = obs.find("ghost marker at (")
        if i >= 0:
            j = obs.find(")", i)
            try:
                xs, ys = obs[i + 17:j].split(",")
                return (int(xs), int(ys))
            except Exception:
                pass
        return None
    return _find_ghost_objs(state, tx, ty)


def _do_ghost(lvl, s):
    _S["prev_objs"] = _S["_cur_objs"]
    _S["last"] = {"kind": "ghost"}
    tag = "link" if s["chain"] else "t%d" % _S["ti"]
    return ("l%d ghost %s" % (lvl, tag), "act.ghost_follow",
            {"x": s["x"], "y": s["y"]})


def _next_token(state, W, rng, lvl, kb, history):
    _S["ti"] += 1
    if _S["ti"] >= len(_S["toks"]):
        _S["toks"] = []
    return _click(state, W, rng, lvl, kb, history)


def _click_fallback(state, W, rng, lvl, kb):
    # toks empty: start a new round, or fall back.
    if _S["round"] >= 1 and not _S["hot_any"]:
        if kb:
            _S["kb_mode"] = True
            return _kb(state, W, rng, kb, lvl)
        return _grid(state, W, rng, lvl)
    if _S["round"] >= _MAX_ROUNDS:
        _S["round"] = 0
        _S["hot_any"] = False
        return _grid(state, W, rng, lvl)
    _S["round"] += 1
    toks = _tokens(state, W)
    if not toks:
        return _grid(state, W, rng, lvl)
    _S["toks"] = [{"o": o, "st": _new_tok(o)} for o in toks]
    _S["ti"] = 0
    return None


def _click(state, W, rng, lvl, kb, history):
    if not _S["toks"]:
        fb = _click_fallback(state, W, rng, lvl, kb)
        if fb is not None:
            return fb
    tok = _S["toks"][_S["ti"]]
    s = tok["st"]
    last = _S.get("last")
    if last is not None:
        kind = last["kind"]
        _S["last"] = None
        if kind == "ghost" and s["phase"] == "ghost":
            g = _get_ghost(state, history, s["x"], s["y"])
            if g and g not in s["seen"]:
                s["seen"].add(g)
                s["x"], s["y"] = g
                s["chain"] = True
            else:
                s["chain"] = False
                s["gp"] += 1
                if g:
                    return _next_token(state, W, rng, lvl, kb,
                                       history)  # ghost cycle
        elif kind in ("drop", "pickdrop") and s["phase"] == "pd":
            if state["changed_px"] > 0 and not s["hot"]:
                s["hot"] = True
                _S["hot_any"] = True
                s["ntargets"] = W["pd_targets"] + W["hot_bonus_targets"]
            s["ti"] += 1
            s["sel"] = False
    if s["phase"] == "ghost":
        if s["chain"]:
            if _S["gpool"] > 0:
                _S["gpool"] -= 1
                return _do_ghost(lvl, s)
            s["chain"] = False
            s["phase"] = "dial"
            s["dn"] = 0
        elif s["gp"] < W["ghost_probes"]:
            return _do_ghost(lvl, s)
        else:
            s["phase"] = "dial"
            s["dn"] = 0
    if s["phase"] == "dial":
        if s["dn"] < W["dial_probe_n"] or (
                state["changed_px"] > 0
                and s["dn"] < 2 * W["dial_probe_n"]):
            s["dn"] += 1
            _S["last"] = {"kind": "dial"}
            return ("l%d dial t%d n%d" % (lvl, _S["ti"], s["dn"]), "act.try",
                    {"action": 6, "x": s["x"], "y": s["y"]})
        s["phase"] = "pd"
        s["ti"] = 0
        s["ntargets"] = W["pd_targets"]
        s["sel"] = False
    if s["phase"] == "pd":
        tgts = _targets(state, s)
        if tgts and s["ti"] < s["ntargets"]:
            gx, gy = tgts[s["ti"] % len(tgts)]
            if W["pd_reselect"] and not s["sel"]:
                s["sel"] = True
                _S["last"] = {"kind": "select"}
                return ("l%d select t%d" % (lvl, _S["ti"]), "act.click_at",
                        {"x": s["x"], "y": s["y"]})
            s["sel"] = False
            if W["pd_reselect"]:
                _S["last"] = {"kind": "drop"}
                return ("l%d drop t%d" % (lvl, _S["ti"]), "act.click_at",
                        {"x": gx, "y": gy})
            _S["last"] = {"kind": "pickdrop"}
            return ("l%d drop t%d" % (lvl, _S["ti"]), "act.pick_drop",
                    {"x1": s["x"], "y1": s["y"],
                     "x2": gx, "y2": gy})
    return _next_token(state, W, rng, lvl, kb, history)


def _grid(state, W, rng, lvl):
    g = _S.get("grid")
    if g is None:
        g = {"pts": [(x, y)
                     for y in range(8, 64, _GRID_STEP)
                     for x in range(8, 64, _GRID_STEP)],
             "i": 0, "cand": None, "dn": 0}
        _S["grid"] = g
    last = _S.get("last")
    if last is not None and last.get("kind") == "grid":
        _S["last"] = None
        if state["changed_px"] > 0 and state["changed_bbox"]:
            x0, y0, x1, y1 = state["changed_bbox"]
            g["cand"] = ((x0 + x1) // 2, (y0 + y1) // 2)
            g["dn"] = 0
    if g["cand"]:
        if g["dn"] < W["dial_probe_n"] or (
                state["changed_px"] > 0
                and g["dn"] < 2 * W["dial_probe_n"]):
            g["dn"] += 1
            _S["last"] = {"kind": "grid_dial"}
            return ("l%d dial grid" % lvl, "act.try",
                    {"action": 6, "x": g["cand"][0], "y": g["cand"][1]})
        g["cand"] = None
    if g["i"] >= len(g["pts"]):
        _S["grid"] = None
        _S["round"] = 0
        _S["hot_any"] = False
        return ("l%d rescan" % lvl, "perceive.describe", {})
    x, y = g["pts"][g["i"]]
    g["i"] += 1
    _S["last"] = {"kind": "grid"}
    return ("l%d grid %d" % (lvl, g["i"]), "act.try",
            {"action": 6, "x": x, "y": y})


def _kb(state, W, rng, kb, lvl):
    K = _S.get("kb")
    if K is None:
        K = {"swept": False, "tried": set(), "stats": {}, "visited": set(),
             "stuck": 0, "last": None}
        _S["kb"] = K
    key = _key(state)
    if K["last"] is not None:
        _lk, la = K["last"]
        st = K["stats"].setdefault(la, [0, 0, 0.0])
        st[0] += 1
        if key not in K["visited"]:
            st[1] += 1
        st[2] += (state["changed_px"] - st[2]) / st[0]
        K["last"] = None
    K["visited"].add(key)
    if state["changed_px"] == 0:
        K["stuck"] += 1
    else:
        K["stuck"] = 0
    if K["stuck"] >= _KB_STUCK and state["steps_used"] > 0:
        K["stuck"] = 0
        return ("l%d stuck reset" % lvl, "act.reset", {})
    if not K["swept"]:
        K["swept"] = True
        for a in kb:
            K["tried"].add((key, a))
        return ("l%d sweep keys" % lvl, "act.sweep", {})
    cands = []
    for a in kb:
        st = K["stats"].get(a, [0, 0, 0.0])
        untried = 0 if (key, a) in K["tried"] else 1
        is_painter = (st[0] >= _PAINTER_TRIALS and st[1] == st[0])
        is_noop = (st[0] >= _NOOP_TRIALS and st[2] == 0)
        cands.append(((untried, 1 if is_noop else 0, 1 if is_painter else 0,
                       st[2], a), a))
    cands.sort(key=lambda t: t[0])
    best_rank = cands[0][0]
    best = [a for r, a in cands if r == best_rank]
    a = rng.choice(best) if len(best) > 1 else best[0]
    K["tried"].add((key, a))
    K["last"] = (key, a)
    return ("l%d try a%d" % (lvl, a), "act.try", {"action": a})


def decide(state, history, rng):
    global _S
    W = WEIGHTS
    lvl = state["level"]
    steps = state["steps_used"]
    if _S is None or steps < _S["max_steps"] or (
            steps == 0 and _S["max_steps"] > 0):
        _S = _fresh()
    _S["max_steps"] = steps if steps > _S["max_steps"] else _S["max_steps"]
    _S["_cur_objs"] = state["objects"]
    if state["game_over"]:
        return ("l%d over" % lvl, "act.reset", {})
    if state["won"]:
        return ("l%d won" % lvl, "perceive.describe", {})
    if lvl != _S["level"]:
        _S["level"] = lvl
        _S["round"] = 0
        _S["toks"] = []
        _S["ti"] = 0
        _S["kb"] = None
        _S["kb_mode"] = False
        _S["hot_any"] = False
        _S["grid"] = None
        _S["last"] = None
        _S["gpool"] = W["ghost_probes"] * W["probe_tokens_max"]
    avail = state["avail"]
    kb = [a for a in avail if a not in (0, 6)]
    click = 6 in avail
    if _S["mode"] is None:
        if click and kb:
            _S["mode"] = "mixed"
        elif click:
            _S["mode"] = "click"
        elif kb:
            _S["mode"] = "keys"
        else:
            _S["mode"] = "mixed"
    if _S["mode"] == "keys" or _S["kb_mode"]:
        return _kb(state, W, rng, kb, lvl)
    return _click(state, W, rng, lvl, kb, history)
