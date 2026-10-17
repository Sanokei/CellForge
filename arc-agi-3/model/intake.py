"""Blackboard passive intake for scripted CellForge agents (TRACK B).

Protocol: cell.observe(state, ctx, hist) -> (annotation_text|None, bid_0_to_1).
Runs EVERY decide() step: pure-python on the agent_api state dict, zero env
actions. Threshold + top-k -> "INTAKE <cell>: <annotation>" lines injected
into the think context each step.

Anti-cheat: annotations are GENERIC observations (changes, action effects,
loops). Never game names, never answers, never level solutions.
The intake never sees a game id; decide() is not given one.

Bid learning: BidLog records intakes shown per step; on level-up the trailing
window gets recency-weighted credit. tune_intake_bids.py turns credit into
per-cell bid multipliers -- the evolved glue.

Shipped cells (2026-10-16, after load-bearing cuts):
  change - frame delta since last step; maintains per-action effect stats.
           Bids on large deltas. Infrastructure for effect.
  effect - live/dead keyboard-action map from median changed_px.
           Bids 0.65 when mapped. Steers the loop-break divert away from
           dead actions.
  loop   - same action repeated, or a short action cycle (period 2..4),
           with no level advance and little change. Bids to 0.95.
           Drives the loop-break override (the load-bearing cell: dropping
           it costs 214 wasted actions and a dev level on the ablation).

Cut as theater (drop-ablations, zero dev RHAE/efficiency effect):
  scene, counter, novelty (perception); decimal, spatial, temporal (Oct-1
  families); stall, progress (metacognitive). Sequence search removed:
  efficiency ablation showed it burning 81 waste actions for 0 solves.
"""
import json
import re
import time
from collections import deque

# ---- knobs (mirrored in the agent's WEIGHTS; tuned on dev games only) ----
THRESH = 0.35
TOPK = 2
LINE_CHARS = 24
LOOP_N = 4          # consecutive identical actions -> loop bid
LOOP_PX = 40        # ...only if last change <= this many px

# Live overrides: the agent's WEIGHTS push knob values here at import.
K = {}


def configure(**kw):
    K.update(kw)


def _knob(name, default):
    return K.get(name, default)


_LVL_RE = re.compile(r"lvl(\d+)")


def new_ctx():
    """Per-run mutable cell memory. The agent resets it on a fresh history."""
    return {
        "step": 0,
        "level": 0,
        "dec_this_level": 0,
        "act_chg": {},          # action key -> [changed_px, ...]
        "live_actions": [],
        "dead_actions": [],
        "diverted": None,       # loop key we already diversified away from
        "resets_this_level": 0,
        "shown": [],
        "last_emit": {},
        "max_level": 0,
    }


def _akey(cell, args):
    """Coarse action key for loop detection."""
    if not isinstance(args, dict):
        args = {}
    if cell == "act.try":
        return "try%s" % args.get("action")
    if cell in ("act.ghost_follow", "act.click_at"):
        return "clk%d,%d" % (int(args.get("x", 32)) // 4,
                             int(args.get("y", 32)) // 4)
    if cell == "act.repeat_click":
        return "rep%d,%d" % (int(args.get("x", 32)) // 4,
                             int(args.get("y", 32)) // 4)
    return str(cell)


def _lvl_of_obs(obs):
    m = _LVL_RE.search(obs or "")
    return int(m.group(1)) if m else None


class Cell:
    name = "base"

    def observe(self, state, ctx, hist):
        raise NotImplementedError


class ChangeCell(Cell):
    """Frame delta since the previous step. Bids on large changes.
    Maintains per-action changed_px stats (feeds EffectCell)."""
    name = "change"

    def observe(self, state, ctx, hist):
        px = state.get("changed_px") or 0
        if hist:
            k = _akey(hist[-1][1], hist[-1][2])
            lst = ctx["act_chg"].setdefault(k, [])
            lst.append(px)
            if len(lst) > 6:
                del lst[0]
        if px < 8:
            return None, 0.0
        bb = state.get("changed_bbox")
        where = ("@%d,%d" % (bb[0], bb[1])) if bb else ""
        return "chg%dpx%s" % (px, where), min(1.0, px / 300.0)


class EffectCell(Cell):
    """Live/dead keyboard-action map from median changed_px per action.
    Live = median >= 10px; dead = median < 3px. Steers loop divert."""
    name = "effect"

    def observe(self, state, ctx, hist):
        meds = {}
        for k, lst in ctx["act_chg"].items():
            if k.startswith("try") and len(lst) >= 3:
                try:
                    a = int(k[3:])
                except ValueError:
                    continue
                s = sorted(lst)
                meds[a] = s[len(s) // 2]
        if len(meds) < 2:
            return None, 0.0
        if max(meds.values()) < 8:
            return None, 0.0
        live = sorted(a for a, v in meds.items() if v >= 10)
        dead = sorted(a for a, v in meds.items() if v < 3)
        if not live:
            return None, 0.0
        ctx["live_actions"] = live
        ctx["dead_actions"] = dead
        return ("live%s dead%s"
                % (",".join(map(str, live)), ",".join(map(str, dead)))), 0.45


class LoopCell(Cell):
    """Same action repeated, or a short action cycle (period 2..4), with no
    level advance and little change. The load-bearing cell."""
    name = "loop"

    def observe(self, state, ctx, hist):
        ln = _knob("loop_n", LOOP_N)
        if len(hist) < ln:
            return None, 0.0
        keys = [_akey(h[1], h[2]) for h in hist[-12:]]
        k0, n = keys[-1], 0
        for k in reversed(keys):
            if k == k0:
                n += 1
            else:
                break
        if n < ln:
            for p in (2, 3, 4):
                if len(keys) >= 2 * p and keys[-p:] == keys[-2 * p:-p]:
                    n = 2 * p
                    k0 = "cyc:" + "+".join(keys[-p:])
                    break
            else:
                return None, 0.0
        lvls = {_lvl_of_obs(h[3]) for h in hist[-n:]}
        if len(lvls) > 1:
            return None, 0.0
        if (state.get("changed_px") or 0) > LOOP_PX:
            return None, 0.0
        return "%sx%d" % (k0, n), min(0.95, 0.45 + 0.12 * n)


ALL = [ChangeCell(), EffectCell(), LoopCell()]


def collect(state, ctx, hist, drop=(), scramble=(), bids=None,
            thresh=THRESH, topk=TOPK):
    """Run all cells, threshold + top-k. Returns [(name, annotation, bid)]."""
    ctx["step"] += 1
    lvl = state.get("level", 0)
    transition = (lvl != ctx["level"])
    ctx["dec_this_level"] += 1
    scored = []
    last_emit = ctx.setdefault("last_emit", {})
    for cell in ALL:
        if cell.name in drop:
            continue
        try:
            txt, bid = cell.observe(state, ctx, hist)
        except Exception:
            continue
        if txt is None:
            continue
        if bids and cell.name in bids:
            bid = max(0.0, min(1.0, bid * bids[cell.name]))
        if cell.name in scramble:
            digits = "".join(c for c in txt if c.isdigit())[::-1][:10]
            txt = "scrambled %s" % (digits or "noise")
            bid = 0.90
        if txt == last_emit.get(cell.name) and bid < 0.8:
            bid *= 0.5  # already shown; re-emit on change or high bid
        if bid >= thresh:
            scored.append((cell.name, txt[:LINE_CHARS], bid))
    scored.sort(key=lambda t: -t[2])
    top = scored[:topk]
    ctx["shown"] = [n for n, _, _ in top]
    for n, t, _ in top:
        last_emit[n] = t
    if transition:
        ctx["level"] = lvl
        ctx["dec_this_level"] = 0
        ctx["resets_this_level"] = 0
        ctx["diverted"] = None
    return top


def format_intake(top):
    return "".join("INTAKE %s: %s\n" % (n, t) for n, t, _ in top)


class BidLog:
    """Logs intakes shown per step; level-ups credit the trailing window."""

    def __init__(self, path, credit_window=6):
        self.path = path
        self.credit_window = credit_window
        self.recent = []
        self.f = open(path, "a")

    def log_step(self, step, names):
        self.recent.append((step, list(names)))
        self.recent = self.recent[-self.credit_window:]

    def log_levelup(self, step, level):
        credited = {}
        for s, names in self.recent:
            w = 1.0 - 0.12 * (step - s)
            for n in names:
                credited[n] = credited.get(n, 0.0) + max(0.2, w)
        self.f.write(json.dumps({"t": time.time(), "event": "levelup",
                                 "step": step, "level": level,
                                 "credit": credited}) + "\n")
        self.f.flush()

    def log_episode(self, game, seed, levels, actions):
        self.f.write(json.dumps({"t": time.time(), "event": "episode",
                                 "game": game, "seed": seed,
                                 "levels": levels, "actions": actions}) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
