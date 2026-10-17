"""Blackboard intake loop for the tiny-LM router.

Perception is PASSIVE: every observer runs each step (zero env actions),
bids for attention, top-k above threshold become INTAKE lines in the prompt.
The LM never calls perception cells; it only emits act.* CALLs.

Anti-cheat: intake lines are generic observations (positions/counts/changes).
No game names, no level solutions, no game-identity leaks.

Bid-usefulness logging: each decision logs which intakes were shown;
tune_bids.py aggregates level-up credit into per-observer multipliers
(cells learning to glue themselves together).
"""
import json
import os
import time

from observers import ALL

STATE_TOKENS = 24      # budget for the STATE line
INTAKE_TOPK = 3
INTAKE_THRESH = 0.30
INTAKE_LINE_TOKENS = 44  # truncate each annotation to ~this many chars


def new_ctx():
    return {"last": {}, "last_counts": {}, "step": 0, "level": 0,
            "shown": []}  # shown: list of intake names shown this step


def collect_intake(frame, prev, ctx, drop=(), scramble=(), bids=None):
    """Run all observers, apply bid threshold + top-k.
    drop: observer names to remove (load-bearing tests).
    scramble: observer names whose annotation is replaced with noise.
    bids: optional per-observer multiplier dict (learned glue).
    Returns list of (name, annotation, bid)."""
    scored = []
    for ob in ALL:
        if ob.name in drop:
            continue
        try:
            txt, bid = ob.observe(frame, prev, ctx)
        except Exception:
            continue
        if txt is None:
            continue
        if bids and ob.name in bids:
            bid = max(0.0, min(1.0, bid * bids[ob.name]))
        if ob.name in scramble:
            txt = "scrambled " + "".join(
                c for c in txt if c in "0123456789")[::-1][:12] or "noise"
            bid = 0.9
        if bid >= INTAKE_THRESH:
            scored.append((ob.name, txt[:INTAKE_LINE_TOKENS], bid))
    scored.sort(key=lambda t: -t[2])
    top = scored[:INTAKE_TOPK]
    ctx["last"].update({n: t for n, t, _ in top})
    ctx["shown"] = [n for n, _, _ in top]
    return top


def format_intake(top):
    return "".join(f"INTAKE {n}: {t}\n" for n, t, _ in top)


def format_state(level, step, avail):
    return f"STATE LVL{level} STEP{step} AVAIL{sorted(int(a) for a in avail)}\n"


class BidLogger:
    """Logs (step, intakes shown, bid) per decision; on level-up, credit the
    intakes shown in the trailing window. JSONL for tune_bids.py."""

    def __init__(self, path, credit_window=6):
        self.path = path
        self.credit_window = credit_window
        self.recent = []  # (step, [names])
        self.f = open(path, "a")

    def log_step(self, step, names):
        self.recent.append((step, list(names)))
        self.recent = self.recent[-self.credit_window:]

    def log_levelup(self, step, level):
        credited = {}
        for s, names in self.recent:
            w = 1.0 - 0.12 * (step - s)  # recency weight
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
