"""Passive perception observers for the blackboard router.

Protocol: observe(frame, prev_frame, ctx) -> (annotation_text|None, bid_0_to_1).
Every observer runs EVERY step (cheap numpy, zero env actions). Each authors
its own salience rules via the bid. Annotations are GENERIC observations:
positions, counts, changes. Never game names, never solutions (anti-cheat).
"""
import numpy as np

import importlib.util as _ilu
import sys as _sys

def _load_lm_cells():
    spec = _ilu.spec_from_file_location(
        "lm_cells_obs", "/srv/arc3-claude/v6run/model/lm/cells.py")
    mod = _ilu.module_from_spec(spec)
    _sys.modules["lm_cells_obs"] = mod
    spec.loader.exec_module(mod)
    return mod

_lm = _load_lm_cells()
_objs = _lm._objs  # reuse component labeling
_new_small_comp = _lm._new_small_comp  # proven ghost/marker detector


class Observer:
    name = "base"

    def observe(self, frame, prev, ctx):
        raise NotImplementedError


class SceneObserver(Observer):
    """Compact scene summary: object list. The router's peripheral vision."""
    name = "scene"

    def observe(self, frame, prev, ctx):
        objs = _objs(frame)
        parts = []
        for o in objs[:6]:
            cx, cy = (o["x0"] + o["x1"]) // 2, (o["y0"] + o["y1"]) // 2
            parts.append(f"c{o['c']}@{cx},{cy}a{o['area']}")
        txt = f"{len(objs)}obj " + " ".join(parts) if parts else "0obj"
        last = ctx["last"].get(self.name)
        bid = 0.55 if txt != last else 0.12
        return txt, bid


class ChangeObserver(Observer):
    """Frame delta since last step. Bids on large changes."""
    name = "change"

    def observe(self, frame, prev, ctx):
        if prev is None:
            return None, 0.0
        d = frame != prev
        n = int(d.sum())
        if n == 0:
            return None, 0.0
        ys, xs = np.nonzero(d)
        txt = f"chg{n}@({xs.min()},{ys.min()},{xs.max()},{ys.max()})"
        bid = min(1.0, n / 400.0)
        return txt, bid


class CounterObserver(Observer):
    """Per-color pixel counts and deltas. Bids when counts shift."""
    name = "counter"

    def observe(self, frame, prev, ctx):
        vals, cnts = np.unique(frame[frame > 0], return_counts=True)
        cur = {int(v): int(c) for v, c in zip(vals, cnts)}
        last = ctx.get("last_counts") or {}
        if not last:
            ctx["last_counts"] = cur
            return None, 0.0
        deltas = {c: cur.get(c, 0) - last.get(c, 0) for c in set(cur) | set(last)}
        big = {c: d for c, d in deltas.items() if abs(d) >= 4}
        ctx["last_counts"] = cur
        if not big:
            return None, 0.0
        txt = " ".join(f"c{c}:{d:+d}" for c, d in sorted(big.items()))
        bid = min(1.0, 0.5 + 0.1 * len(big))
        return "count " + txt, bid


class NoveltyObserver(Observer):
    """New small components vs previous frame (ghost/marker detector).

    Uses the proven _new_small_comp flood-fill detector (same as the
    ghost_follow cell) — the generic component labeler misses 1px markers.
    Bids very high when something new and small appears."""
    name = "novelty"

    def observe(self, frame, prev, ctx):
        if prev is None:
            return None, 0.0
        # nearest to the last click: mirrors the ghost_follow cell's own
        # detector semantics (teacher and eval must agree on which marker)
        near = ctx.get("last_click")
        g = _new_small_comp(prev, frame, near=near)
        if not g:
            return None, 0.0
        rx, ry, size = g
        c = int(frame[ry, rx])
        return f"new c{c}@({rx},{ry})a{size}", 0.95


class HudObserver(Observer):
    """Top-3 rows summary (progress bars / counters live here)."""
    name = "hud"

    def observe(self, frame, prev, ctx):
        rows = []
        for i in range(3):
            vals, cnts = np.unique(frame[i], return_counts=True)
            top = sorted(zip(vals.tolist(), cnts.tolist()),
                         key=lambda t: -t[1])[:2]
            rows.append("".join(f"{v}:{c}" for v, c in top))
        txt = "hud " + "|".join(rows)
        last = ctx["last"].get(self.name)
        bid = 0.4 if txt != last else 0.0
        return txt, bid


ALL = [SceneObserver(), ChangeObserver(), CounterObserver(),
       NoveltyObserver(), HudObserver()]
