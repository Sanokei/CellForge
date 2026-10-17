"""Alpha gen-1: RHAE-efficiency-bred champion from gen0_alpha.

Mechanisms (probed on dev seeds; layouts verified seed-invariant):
- s5i5 (avail==[6], 1 object): DIALS are click-responsive 7x7 spots.
  Hunt by response on a grid (no fixed coords). Refine each hit to a
  reliable pixel (center-ish: best of 4 offsets, verified twice), then
  click it until the goal tick (chg 3..8 = bar reached goal). Positions
  AND counts discovered per level, nothing hardcoded.
- lf52 (click + nobj>2): minority-color token starts a teleport chain;
  ghost_follow reports each link; follow until level-up. Keyboard NEVER
  touched (breaks the chain's hidden selector state).
- keyboard games: sweep untested actions, try in rotation, demote
  painters (deposit novel px forever, never level up) and dead actions.

RHAE: per-level score = min(115, 100*(baseline/actions)^2). Think texts
are 1-3 tokens (harness bills think 100:1 vs bankroll).
"""
import re
from collections import Counter

NAME = "alpha-g1"

CELLS = ["perceive.describe", "perceive.objects", "act.click_at",
         "act.ghost_follow", "act.try", "act.sweep", "act.reset"]

WEIGHTS = {
    "hunt_step": 16,    # dial-hunt grid spacing (px)
    "resp_chg": 8,      # min changed-px for a click to count as dial hit
    "refine_dist": 3,   # offset distance for reliable-pixel search
    "tick_lo": 3,       # chg in [tick_lo, tick_hi] -> dial at goal: stop
    "tick_hi": 8,       # (chg<=2: dead pixel)
    "ghost_min_y": 3,   # ghost markers with y<3 are progress bar, not links
    "small_area": 400,  # max area for chain-token candidate
    "probe_clicks": 4,  # token probes per level before giving up chain
    "click_rank": 0,    # which candidate (rarest-color-first) to probe first
    "order_offset": 0,  # rotate candidate/action try-order
    "max_dial_clicks": 20,  # cap per dial (safety)
    "painter_uses": 3,  # act.try uses before painter suspicion
    "painter_chg": 60,  # mean changed-px/use marking a painter action
}

# refine offsets to try around a hunt hit (center, +x, -x, +y, -y)
def _refine_pts(hit, W):
    d = W["refine_dist"]
    return [(hit[0], hit[1]), (hit[0] + d, hit[1]), (hit[0] - d, hit[1]),
            (hit[0], hit[1] + d), (hit[0], hit[1] - d)]


def _lvl_of(obs):
    m = re.search(r"lvl(\d+)", obs or "")
    return int(m.group(1)) if m else 0


def _chg_of(obs):
    m = re.search(r"chg(\d+)px", obs or "")
    return int(m.group(1)) if m else 0


def _ghost_of(obs):
    if not obs or "no ghost marker" in obs:
        return None
    m = re.search(r"ghost marker at \((\d+),(\d+)\)", obs)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _xy_of(args):
    if isinstance(args, dict) and "x" in args and "y" in args:
        return (int(args["x"]), int(args["y"]))
    return None


def _lvl_hist(history, lvl):
    return [h for h in history if _lvl_of(h[3]) == lvl]


def _tried_points(history, lvl):
    pts = set()
    for (think, cell, args, obs) in _lvl_hist(history, lvl):
        if cell in ("act.click_at", "act.ghost_follow"):
            p = _xy_of(args)
            if p:
                pts.add(p)
    return pts


def _dial_hits(history, lvl, W):
    out = []
    for (think, cell, args, obs) in _lvl_hist(history, lvl):
        if (think or "") == "h" and cell == "act.click_at":
            p = _xy_of(args)
            if p and p not in out and _chg_of(obs) >= W["resp_chg"]:
                out.append(p)
    return out


def _parse_work(think):
    """>'dXXYY' dial work | 'fXXYY' refine | 'gXXYY' ghost | 'cXXYY' probe."""
    m = re.fullmatch(r"([dcfg])(\d{2})(\d{2})", think or "")
    if m:
        return (m.group(1), (int(m.group(2)), int(m.group(3))))
    return None


def _refine_status(history, lvl, hit, W):
    """{pixel: LAST chg} for refine clicks around hit."""
    status = {}
    for (think, cell, args, obs) in _lvl_hist(history, lvl):
        q = _parse_work(think)
        if q and q[0] == "f" and q[1] == hit and cell == "act.click_at":
            p = _xy_of(args)
            if p:
                status[p] = _chg_of(obs)
    return status


def _dial_chgs(history, lvl, px, W):
    """chgs of dial-work clicks at pixel px."""
    out = []
    for (think, cell, args, obs) in _lvl_hist(history, lvl):
        q = _parse_work(think)
        if q and q[0] == "d" and cell == "act.click_at" \
                and _xy_of(args) == px:
            out.append(_chg_of(obs))
    return out


def decide(state, history, rng):
    W = WEIGHTS
    avail = state["avail"]
    lvl = state["level"]
    nobj = state["n_objects"]

    if state["game_over"]:
        return ("r", "act.reset", {})

    kbd = [a for a in avail if a not in (0, 6)]
    if avail == [6] and nobj == 1:
        return _decide_dial(state, history, W)
    if 6 in avail and (not kbd or nobj > 2):
        return _decide_chain(state, history, W)
    return _decide_kbd(state, history, W)


def _hit_done(history, lvl, hit, W):
    """True if any refine click for hit ticked (dial at goal)."""
    for (think, cell, args, obs) in _lvl_hist(history, lvl):
        q = _parse_work(think)
        if q and q[0] == "f" and q[1] == hit:
            c = _chg_of(obs)
            if W["tick_lo"] <= c <= W["tick_hi"]:
                return True
    return False


def _decide_dial(state, history, W):
    lvl = state["level"]
    last_think = history[-1][0] if history else ""
    pw = _parse_work(last_think)

    if pw and pw[0] == "f":
        hit = pw[1]
        if not _hit_done(history, lvl, hit, W):
            status = _refine_status(history, lvl, hit, W)
            pts = _refine_pts(hit, W)
            untried = [p for p in pts if p not in status]
            if untried:
                p = untried[0]
                return (f"f{hit[0]:02d}{hit[1]:02d}",
                        "act.click_at", {"x": p[0], "y": p[1]})
            ranked = sorted(pts, key=lambda p: status.get(p, 0), reverse=True)
            for best in ranked:
                c = status.get(best, 0)
                if c > 2 and c >= W["resp_chg"]:
                    return (f"f{hit[0]:02d}{hit[1]:02d}",
                            "act.click_at", {"x": best[0], "y": best[1]})
            for best in ranked:
                if status.get(best, 0) > 2:
                    return (f"f{hit[0]:02d}{hit[1]:02d}",
                            "act.click_at", {"x": best[0], "y": best[1]})

    hits = _dial_hits(history, lvl, W)
    refining = set()
    for h in history:
        q = _parse_work(h[0])
        if q and q[0] == "f" and _lvl_of(h[3]) == lvl:
            refining.add(q[1])
    for hpx in hits:
        if hpx not in refining and not _hit_done(history, lvl, hpx, W):
            p0 = _refine_pts(hpx, W)[0]
            return (f"f{hpx[0]:02d}{hpx[1]:02d}",
                    "act.click_at", {"x": p0[0], "y": p0[1]})

    tried = _tried_points(history, lvl)
    step = W["hunt_step"]
    for gx in range(8, 64, step):
        for gy in range(8, 64, step):
            if (gx, gy) not in tried:
                return ("h", "act.click_at", {"x": gx, "y": gy})
    # hunt exhausted: park on describe (RESET would wipe levels_completed)
    return ("l", "perceive.describe", {})


def _candidates(objs, W):
    small = [o for o in objs if o["area"] <= W["small_area"]]
    if not small:
        small = objs[:4]
    cnt = Counter(o["c"] for o in small)
    small = sorted(small, key=lambda o: (cnt[o["c"]], -o["area"]))
    pts = [((o["x0"] + o["x1"]) // 2, (o["y0"] + o["y1"]) // 2)
           for o in small]
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


def _parse_objects_obs(obs):
    """Parse perceive.objects text into object dicts."""
    objs = []
    for m in re.finditer(
            r"color (\d+) bbox \((\d+),(\d+)\)-\((\d+),(\d+)\) area (\d+)",
            obs or ""):
        c, x0, y0, x1, y1, a = map(int, m.groups())
        objs.append({"c": c, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                     "area": a, "cx": (x0 + x1) // 2, "cy": (y0 + y1) // 2})
    return objs


def _decide_chain(state, history, W):
    lvl = state["level"]
    last_obs = history[-1][3] if history else ""
    last_lvl = _lvl_of(last_obs) if history else lvl

    g = _ghost_of(last_obs)
    if g and g[1] >= W["ghost_min_y"] and last_lvl == lvl:
        prev_xy = _xy_of(history[-1][2])
        if prev_xy != g:
            return (f"g{g[0]:02d}{g[1]:02d}", "act.ghost_follow",
                    {"x": g[0], "y": g[1]})

    # state objects truncate to 8; the row-below heuristic finds
    # minority tokens cut off from the list (e.g. lf52 chain-start)
    objs = state["objects"]

    tried = _tried_points(history, lvl)
    n_probes = sum(1 for h in _lvl_hist(history, lvl)
                   if h[1] == "act.ghost_follow"
                   and (_parse_work(h[0]) or ("x",))[0] == "c")
    cands = _candidates(objs, W)
    if cands and n_probes < W["probe_clicks"]:
        untried = [p for p in cands if p not in tried]
        tgt = untried[0] if untried else cands[n_probes % len(cands)]
        return (f"c{tgt[0]:02d}{tgt[1]:02d}", "act.ghost_follow",
                {"x": tgt[0], "y": tgt[1]})
    # object candidates failed: the chain-start may be a minority token
    # below the majority row (cut off from truncated object list).
    # Probe below each candidate's row.
    if n_probes < W["probe_clicks"] + 6:
        for (cx, cy) in cands:
            for dy in (6, 12):
                p = (cx, cy + dy)
                if 0 <= p[1] < 64 and p not in tried:
                    return (f"c{p[0]:02d}{p[1]:02d}", "act.ghost_follow",
                            {"x": p[0], "y": p[1]})
    # chain failed: fall back to dial hunting (exercises dial weights)
    return _decide_dial(state, history, W)


def _decide_kbd(state, history, W):
    avail = state["avail"]
    kbd = [a for a in avail if a not in (0, 6)]
    if not history:
        return ("l", "perceive.describe", {})
    tested = set()
    stats = {}
    for i, h in enumerate(history):
        if h[1] == "act.try" and isinstance(h[2], dict):
            a = h[2].get("action")
            if a is None:
                continue
            tested.add(a)
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
    untested = [a for a in kbd if a not in tested and a not in swept]
    if untested:
        return ("s", "act.sweep", {})

    def demoted(a):
        s = stats.get(a)
        return bool(s and s[0] >= W["painter_uses"] and not s[2]
                    and s[1] / max(1, s[0]) > W["painter_chg"])

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
        return (f"k{a}", "act.try", {"action": a})
    if 6 in avail and state["objects"]:
        o = state["objects"][0]
        return (f"c{o['cx']:02d}{o['cy']:02d}", "act.ghost_follow",
                {"x": o["cx"], "y": o["cy"]})
    return ("r", "act.reset", {})
