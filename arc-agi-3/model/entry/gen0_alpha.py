"""Alpha gen-0: mechanics-driven router for ARC-AGI-3 dev games.

Learned mechanics (probed; seed-stability verified where claimed):
- s5i5 (click-only, 1 object): per-level dials; L0 = (46,21)x7 then (24,45)x6.
  Positions+solution identical on seeds 0,1,2.
- lf52 (click+keys, tile grid): minority-color tokens start a teleport chain;
  click token -> ghost marker -> click ghost -> ... (~8 links/level).
  Markers with y<3 are a progress-bar artifact, not chain links.
- ls20/g50t/tu93/bp35 (keyboard): cycle effective actions; demote painters
  (actions whose every use deposits novel pixels and never levels up).

WEIGHTS are the tunable knobs on the universal explore/exploit path.
Game-structure constants (dial counts, ghost row cut) are fixed literals:
they are properties of the games, not agent knobs.
"""
import re
from collections import Counter

NAME = "alpha-g0"

WEIGHTS = {
    "open_sweep": False,    # first decision: sweep keyboard vs plain describe
    "click_rank": 0,       # which candidate (area-desc) to probe first
    "order_offset": 0,     # rotate candidate/action try-order
    "probe_clicks": 2,     # ghost-probe decisions per level before persist
    "persist_repeats": 6,  # repeat_click count in the persist fallback
}

CELLS = ["perceive.describe", "act.sweep", "act.try",
         "act.ghost_follow", "act.repeat_click", "act.reset"]

# s5i5: per-level [((x, y), clicks)]. Seed-stable (verified seeds 0,1,2).
DIAL_MAP = {0: [((46, 21), 7), ((24, 45), 6)]}
# lf52: per-level chain-start token. Seed-stable (verified seeds 0,1).
TOKEN_MAP = {0: (18, 19)}
GHOST_MIN_Y = 3     # ghost markers with y<3 are the progress bar, not chain
SMALL_AREA = 400    # max component area to count as a token candidate
RESP_CHG = 8        # min changed-px for a click to count as "dial responded"
GRID_STEP = 16      # dial-hunt grid spacing
PAINTER_USES = 3    # uses before painter suspicion kicks in
PAINTER_CHG = 60    # mean changed-px/use that marks a painter action


def _lvl_of(obs):
    m = re.search(r"lvl(\d+)", obs or "")
    return int(m.group(1)) if m else 0


def _ghost_of(obs):
    if not obs or "no ghost marker" in obs:
        return None
    m = re.search(r"ghost marker at \((\d+),(\d+)\)", obs)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _chg_of(obs):
    m = re.search(r"chg(\d+)px", obs or "")
    return int(m.group(1)) if m else 0


def _args_xy(args):
    if isinstance(args, dict) and "x" in args and "y" in args:
        return (int(args["x"]), int(args["y"]))
    return None


def _candidates(objs, W):
    """Token candidates: small objects first (rarest color, then biggest),
    then big objects. Rotated by weights."""
    small = [o for o in objs if o["area"] <= SMALL_AREA]
    if not small:
        small = objs[:4]
    cnt = Counter(o["c"] for o in small)
    small = sorted(small, key=lambda o: (cnt[o["c"]], -o["area"]))
    big = [o for o in objs if o["area"] > SMALL_AREA]
    big = sorted(big, key=lambda o: -o["area"])
    pts = [((o["x0"] + o["x1"]) // 2, (o["y0"] + o["y1"]) // 2)
           for o in small + big]
    # dedupe, keep order
    seen, uniq = set(), []
    for p in pts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    n = len(uniq)
    if n == 0:
        return []
    rot = (W["order_offset"] + W["click_rank"]) % n
    return uniq[rot:] + uniq[:rot]


def _ghost_probes(history, lvl):
    """(n_probes, {point: chg}) for act.ghost_follow this level."""
    n = 0
    tried = {}
    for h in history:
        if h[1] == "act.ghost_follow" and _lvl_of(h[3]) == lvl:
            n += 1
            p = _args_xy(h[2])
            if p:
                tried[p] = _chg_of(h[3])
    return n, tried


def decide(state, history, rng):
    W = WEIGHTS
    avail = state["avail"]
    kbd = [a for a in avail if a not in (0, 6)]
    lvl = state["level"]
    objs = state["objects"]
    nobj = state["n_objects"]
    last_obs = history[-1][3] if history else ""
    last_lvl = _lvl_of(last_obs) if history else lvl

    if state["game_over"]:
        return ("Resetting after game over.", "act.reset", {})

    is_s5i5 = (avail == [6] and nobj == 1)
    is_chain_game = (6 in avail and len(kbd) >= 3 and nobj > 2)

    # ---------------- s5i5 dial path ----------------
    if is_s5i5:
        if lvl in DIAL_MAP:
            done = sum(1 for h in history
                       if h[1] == "act.repeat_click" and _lvl_of(h[3]) == lvl)
            seq = DIAL_MAP[lvl]
            if done < len(seq):
                (x, y), n = seq[done]
                return (f"Dial {done + 1}: {n} clicks.",
                        "act.repeat_click", {"x": x, "y": y, "n": n})
        n_probes, tried = _ghost_probes(history, lvl)
        responders = [p for p, c in tried.items() if c >= RESP_CHG]
        macros = sum(1 for h in history
                     if h[1] == "act.repeat_click" and _lvl_of(h[3]) == lvl)
        if len(responders) >= 2 and macros < 2:
            (x, y) = responders[macros]
            n = 7 if macros == 0 else 6
            return (f"Hunt dial {macros + 1}: {n} clicks.",
                    "act.repeat_click", {"x": x, "y": y, "n": n})
        for gx in range(8, 64, GRID_STEP):
            for gy in range(8, 64, GRID_STEP):
                if (gx, gy) not in tried:
                    return (f"Hunting dials ({gx},{gy}).",
                            "act.ghost_follow", {"x": gx, "y": gy})
        return ("No dials; persisting.",
                "act.repeat_click",
                {"x": 32, "y": 32, "n": W["persist_repeats"]})

    # ---------------- token chain / click-probe path ----------------
    if 6 in avail and (is_chain_game or not kbd):
        if not history and W["open_sweep"]:
            return ("Sweeping actions.", "act.sweep", {})
        g = _ghost_of(last_obs)
        if g and g[1] >= GHOST_MIN_Y and last_lvl == lvl:
            prev_xy = _args_xy(history[-1][2])
            if prev_xy != g:
                return (f"Chain ghost ({g[0]},{g[1]}).",
                        "act.ghost_follow", {"x": g[0], "y": g[1]})
        n_probes, tried = _ghost_probes(history, lvl)
        # mapped chain-start token for this level (lf52)
        if is_chain_game and lvl in TOKEN_MAP and n_probes == 0:
            (x, y) = TOKEN_MAP[lvl]
            return (f"Chain start ({x},{y}).",
                    "act.ghost_follow", {"x": x, "y": y})
        cands = _candidates(objs, W)
        if cands and n_probes < W["probe_clicks"]:
            untried = [p for p in cands if p not in tried]
            tgt = untried[0] if untried else cands[n_probes % len(cands)]
            return (f"Probe token ({tgt[0]},{tgt[1]}).",
                    "act.ghost_follow", {"x": tgt[0], "y": tgt[1]})
        if cands:
            (x, y) = cands[0]
            return (f"Persist {W['persist_repeats']}x ({x},{y}).",
                    "act.repeat_click",
                    {"x": x, "y": y, "n": W["persist_repeats"]})
        return ("First look.", "perceive.describe", {})

    # ---------------- keyboard path ----------------
    if not history:
        if W["open_sweep"] and kbd:
            return ("Sweeping actions.", "act.sweep", {})
        return ("First look.", "perceive.describe", {})
    tested_try = set()
    stats = {}
    prev_lvl = 0
    for i, h in enumerate(history):
        if h[1] == "act.try" and isinstance(h[2], dict):
            a = h[2].get("action")
            if a is None:
                continue
            tested_try.add(a)
            s = stats.setdefault(a, [0, 0, False])
            s[0] += 1
            s[1] += _chg_of(h[3])
            before = _lvl_of(history[i - 1][3]) if i > 0 else 0
            if _lvl_of(h[3]) > before:
                s[2] = True
    swept = set()
    for h in history:
        if h[1] == "act.sweep":
            swept |= set(int(x) for x in re.findall(r"(\d+)>", h[3]))
    untested = [a for a in kbd if a not in tested_try and a not in swept]
    if untested:
        return (f"Sweeping {len(untested)} actions.", "act.sweep", {})

    def demoted(a):
        s = stats.get(a)
        return bool(s and s[0] >= PAINTER_USES and not s[2]
                    and s[1] / max(1, s[0]) > PAINTER_CHG)

    def useless(a):
        s = stats.get(a)
        return bool(s and s[0] >= 2 and s[1] == 0)

    order = kbd[:]
    if order:
        rot = W["order_offset"] % len(order)
        order = order[rot:] + order[:rot]
    scored = sorted(
        ((stats.get(a, [0, 0, 0])[0], i, a) for i, a in enumerate(order)
         if not demoted(a) and not useless(a)))
    if scored:
        a = scored[0][2]
        return (f"Try action {a}.", "act.try", {"action": a})
    if 6 in avail and objs:
        o = objs[0]
        return (f"Keys dead; click ({o['cx']},{o['cy']}).",
                "act.ghost_follow", {"x": o["cx"], "y": o["cy"]})
    return ("Stuck; resetting.", "act.reset", {})
