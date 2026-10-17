"""gen1_blackboard: gen0_alpha champion + blackboard passive intake (TRACK B).

Wiring: every decide(), intake cells observe the state dict (zero env
actions), bid for attention, top-k above threshold -> "INTAKE <cell>:
<annotation>" lines in think (display-gated to high-salience winners to
protect the think-token budget). The intake never sees a game id;
annotations are generic observations, never answers, never game names.

One intake-driven override, on the champion's GENERIC explore path only
(thinks starting Sweeping/First look/Try action/Keys dead). Game-specific
literal paths (Dial/Hunt/Probe/Persist/Chain/No dials) are untouched,
except an obviously futile probe/persist run (>=6 repeats) may be broken.

  loop - same action >=loop_n, or a short cycle (period 2..4), with no level
         advance and little change -> diversify to the least-used LIVE
         keyboard action (effect cell's dead actions are skipped), else
         reset the level (<=2/level). Futile cycles reset outright.

Shipped intake cells: change, effect, loop (see intake.py for the
load-bearing record and the cut list).

Load-bearing: INTAKE_DROP / INTAKE_SCRAMBLE env (comma-separated cell names)
for ablations. INTAKE_BIDS: learned multiplier file (default
model/intake_bids.json). INTAKE_LOG: BidLog JSONL (dev games only).
"""
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import intake  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gen0_alpha_base", os.path.join(_HERE, "gen0_alpha.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

NAME = "blackboard-g1"

WEIGHTS = {
    # champion knobs (kept for API parity; live inside gen0_alpha_base)
    "open_sweep": False,
    "click_rank": 0,
    "order_offset": 0,
    "probe_clicks": 2,
    "persist_repeats": 6,
    # blackboard knobs (live: pushed into intake at import)
    "intake_thresh": 0.35,
    "intake_topk": 2,
    "loop_n": 4,
}

CELLS = _base.CELLS + ["intake.blackboard"]

intake.configure(loop_n=WEIGHTS["loop_n"])

_DROP = tuple(x for x in os.environ.get("INTAKE_DROP", "").split(",") if x)
_SCRAMBLE = tuple(x for x in os.environ.get("INTAKE_SCRAMBLE", "").split(",")
                  if x)
_BIDS = {}
_bp = os.environ.get("INTAKE_BIDS",
                     os.path.join(_HERE, "..", "intake_bids.json"))
if os.path.exists(_bp):
    try:
        import json as _json
        _BIDS = _json.load(open(_bp)).get("bids", {}) or {}
    except Exception:
        _BIDS = {}

_BIDLOG = None
_lp = os.environ.get("INTAKE_LOG")
if _lp:
    _BIDLOG = intake.BidLog(_lp)

_CTX = None

_LOOP_RE = re.compile(r"^(\S+)x(\d+)$")

_GENERIC = ("Sweeping", "First look.", "Try action", "Keys dead; click")
_PROBEISH = ("Probe token", "Persist", "No dials")

# Display gate: overrides see everything >= intake_thresh, but only
# high-salience winners are worth think tokens.
DISPLAY_BID = 0.6
DISPLAY_TOPK = 1


def _action_of_key(key):
    m = re.match(r"try(\d+)$", key or "")
    return int(m.group(1)) if m else None


def _least_used_kbd(state, history, exclude=None, skip=()):
    kbd = [a for a in state.get("avail", []) if a not in (0, 6)]
    cands = [a for a in kbd if a != exclude and a not in skip]
    if not cands:
        return None
    uses = {a: 0 for a in cands}
    for h in history:
        if h[1] == "act.try" and isinstance(h[2], dict):
            a = h[2].get("action")
            if a in uses:
                uses[a] += 1
    return min(cands, key=lambda a: uses[a])


def _line(top, name):
    for n, a, _b in top:
        if n == name:
            return a
    return None


def _apply_overrides(state, history, top, think, cell, args, W):
    """Returns (think, cell, args, fired). fired=True iff intake overrode."""
    ctx = _CTX
    generic = think.startswith(_GENERIC)
    probeish = think.startswith(_PROBEISH)

    m = _LOOP_RE.match(_line(top, "loop") or "")
    if m and (generic or (probeish and int(m.group(2)) >= 6)):
        key = m.group(1)
        if key.startswith("cyc:"):
            if ctx["resets_this_level"] < 2:
                ctx["resets_this_level"] += 1
                ctx["diverted"] = None
                return ("lb:reset", "act.reset", {}, True)
        elif ctx.get("diverted") == key:
            if ctx["resets_this_level"] < 2:
                ctx["resets_this_level"] += 1
                ctx["diverted"] = None
                return ("lb:reset", "act.reset", {}, True)
        else:
            tgt = _least_used_kbd(state, history,
                                  exclude=_action_of_key(key),
                                  skip=ctx.get("dead_actions") or [])
            if tgt is not None:
                ctx["diverted"] = key
                return ("lb:try%d" % tgt, "act.try", {"action": tgt}, True)
            if ctx["resets_this_level"] < 2:
                ctx["resets_this_level"] += 1
                ctx["diverted"] = None
                return ("lb:reset", "act.reset", {}, True)
    return (think, cell, args, False)


def decide(state, history, rng):
    global _CTX
    W = WEIGHTS
    if not history:
        if _CTX is not None and _CTX["step"] > 0 and _BIDLOG is not None:
            _BIDLOG.log_episode("dev", 0, _CTX.get("max_level", 0),
                               _CTX["step"])
        _CTX = intake.new_ctx()
    ctx = _CTX
    old_lvl = ctx["level"]
    try:
        top = intake.collect(state, ctx, history, drop=_DROP,
                             scramble=_SCRAMBLE,
                             bids=_BIDS or None,
                             thresh=W["intake_thresh"],
                             topk=int(W["intake_topk"]))
    except Exception:
        top = []
    ctx["max_level"] = max(ctx.get("max_level", 0), state.get("level", 0))
    try:
        think, cell, args = _base.decide(state, history, rng)
    except Exception:
        think, cell, args = "base crashed", "perceive.describe", {}
    think, cell, args, fired = _apply_overrides(state, history, top, think,
                                                cell, args, W)
    # Display gate: high-salience winners, but the loop line only earns its
    # tokens when it actually changed the decision (override fired).
    shown = [(n, a, b) for n, a, b in top
             if b >= DISPLAY_BID and (n != "loop" or fired)][:DISPLAY_TOPK]
    if _BIDLOG is not None:
        _BIDLOG.log_step(ctx["step"], [n for n, _, _ in shown])
        if state.get("level", 0) > old_lvl:
            _BIDLOG.log_levelup(ctx["step"], state["level"])
    prefix = intake.format_intake(shown)
    if prefix:
        think = prefix + think
    return (think, cell, args or {})
