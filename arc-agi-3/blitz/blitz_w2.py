"""Wave-2 blitz agents (v2): implement the three highest-EV wave-1 diagnoses.

Design decisions (from diagnosing a blitz6 regression):
- State keys stay EXACT (frame-hash). Object-level merging is too risky as
  a key replacement (it blinded lf52's toggle dots).
- Object/effect-level reasoning lives in ACTION ORDERING, not in keys.

blitz6 = blitz4 + VolatileMask2 (mask ONLY action-uncorrelated oscillators;
         load-bearing toggled pixels are never masked) + RESET trap guard.
blitz7 = blitz6 + click-pair macros v2 (salience-ordered pick->drop, from
         attempt 0 on click games).
blitz8 = blitz7 + painter-demotion: action classes whose every trial only
         ever deposits novel pixels (never revisits a known state, e.g.
         ls20's action-2 paint) are strongly demoted in the try-order.
         Demote, never prune: completeness is preserved.

All tiny/hand-authored: hashing, flood fill, counting. No training, no LLM.
"""
from __future__ import annotations

import hashlib

import numpy as np
from arcengine import GameAction, GameState

from blitz_agent import BlitzExplorer4, AGENTS as _AGENTS
from cells import components, salience_order, last_frame

RESET = ("R", -1, -1)


class VolatileMask2:
    """VolatileMask + action-discrimination VETO.

    Starts from the original VolatileMask logic (per-pixel high-rate test +
    peripheral thin-band test), which correctly masks ambient elements like
    progress bars. Adds a VETO: a pixel is NOT masked if its changes
    discriminate actions (max_a P(change|a) - min_a P(change|a) > 0.5 over
    well-tried action classes), i.e. the changes are load-bearing, not
    ambient. This keeps lf52's top progress bar masked while protecting
    action-correlated toggles.
    """

    def __init__(self, shape=(64, 64), warmup=30, every=15, min_trials=8,
                 veto_gap=0.5):
        self.counts = np.zeros(shape, np.int32)
        self.row_hits = np.zeros(shape[0], np.int32)
        self.col_hits = np.zeros(shape[1], np.int32)
        self.steps = 0
        self.mask = np.zeros(shape, bool)
        self.warmup, self.every, self.min_trials = warmup, every, min_trials
        self.veto_gap = veto_gap
        self.act_steps = {}   # aid -> n effective steps
        self.act_chg = {}     # aid -> per-pixel change counts
        self.act_row = {}     # aid -> per-row hit counts
        self.act_col = {}     # aid -> per-col hit counts
        self.owner = None

    @staticmethod
    def _aid(act):
        if act == RESET:
            return "R"
        if isinstance(act, tuple):
            if act[0] == 6:
                return "click"
            if act[0] == "M":
                return "M"
            if act[0] == "R":
                return "R"
            return ("k", act[0])
        return act

    def observe(self, prev, cur):
        act = getattr(self.owner, "_pending_act", None)
        d = prev != cur
        if not d.any():
            return False      # no-op steps carry no information
        aid = self._aid(act)
        if aid == "R":
            return False      # RESET rewrites everything
        self.counts += d
        self.row_hits += d.any(axis=1)
        self.col_hits += d.any(axis=0)
        self.steps += 1
        self.act_steps[aid] = self.act_steps.get(aid, 0) + 1
        ac = self.act_chg.get(aid)
        if ac is None:
            ac = np.zeros_like(self.counts)
            self.act_chg[aid] = ac
        ac[d] += 1
        ar = self.act_row.get(aid)
        if ar is None:
            ar = np.zeros(self.counts.shape[0], np.int32)
            self.act_row[aid] = ar
        arc_ = self.act_col.get(aid)
        if arc_ is None:
            arc_ = np.zeros(self.counts.shape[1], np.int32)
            self.act_col[aid] = arc_
        ar[d.any(axis=1)] += 1
        arc_[d.any(axis=0)] += 1
        if self.steps < self.warmup or self.steps % self.every:
            return False
        # --- original per-pixel test: changes on >90% of steps (blinkers)
        new = ((self.counts > 0.9 * self.steps)
               if self.steps >= 100 else np.zeros_like(self.mask))
        # --- band test: thin peripheral bands that hit on >70% of ALL
        # effective steps (ambient progress bars / counters). Action-correlated
        # bands (e.g. ls20's paint trail, hit only on paint steps) fall below
        # the bar and are never masked.
        for hits, axis in ((self.row_hits, 0), (self.col_hits, 1)):
            cand = hits > 0.7 * self.steps
            i = 0
            while i < len(cand):
                if not cand[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(cand) and cand[j + 1]:
                    j += 1
                peripheral = i < 8 or j >= len(cand) - 8
                if j - i + 1 <= 4 and peripheral:
                    band = np.zeros_like(self.mask)
                    if axis == 0:
                        band[i:j + 1, :] = True
                    else:
                        band[:, i:j + 1] = True
                    new |= band & (self.counts > 0)
                i = j + 1
        # --- VETO: unmask pixels whose changes discriminate actions
        good = [a for a, n in self.act_steps.items() if n >= self.min_trials]
        if len(good) >= 2 and new.any():
            veto = np.zeros_like(new)
            for i in np.flatnonzero(new):
                rates = [self.act_chg[a].flat[i] / self.act_steps[a]
                         for a in good]
                if max(rates) - min(rates) > self.veto_gap:
                    veto.flat[i] = True
            new &= ~veto
        new &= ~self.mask
        if new.any():
            self.mask |= new
            return True
        return False


class BlitzExplorer6(BlitzExplorer4):
    """blitz4 + VolatileMask2 + RESET trap guard. Exact frame-hash keys."""

    def __init__(self, env, budget, seed=0, log=None, flags=(), tags=()):
        super().__init__(env, budget, seed=seed, log=log, flags=flags,
                         tags=tags)
        self.vol = VolatileMask2()
        self.vol.owner = self
        self._pending_act = None
        self.log("blitz6: VolatileMask2, exact keys")

    def _send(self, act):
        self._pending_act = act
        return super()._send(act)

    def _fresh_attempt(self):
        if self.used - self.level_start_used <= 0:
            # latent trap: RESET with zero prior actions on a level can drop
            # to level 0 -> take a real action first instead
            u = self._next_untested()
            self.log("blitz6: zero actions this level, deferring RESET")
            return u if u is not None else RESET
        return super()._fresh_attempt()


class BlitzExplorer7(BlitzExplorer6):
    """blitz6 + click-pair macros v2.

    pick = small non-bg components (tokens), drop = salient click targets;
    salience-ordered pick->drop pairs, from attempt 0 on click games.
    """

    def actions_for(self, f):
        acts = super().actions_for(f)
        if 6 not in self.avail or "click" not in getattr(self, "tags", set()):
            return acts
        vals, cnts = np.unique(f, return_counts=True)
        bg = int(vals[np.argmax(cnts)])
        comps = components(f)
        picks = [c for c in salience_order(comps, bg)
                 if c["color"] != bg and c["size"] <= 64][:6]
        drops = [(a[1], a[2]) for a in acts if a[0] == 6][:10]
        have = {a for a in acts
                if isinstance(a, tuple) and len(a) == 5 and a[0] == "M"}
        macros = []
        for p in picks:
            for (dx, dy) in drops:
                if (p["x"], p["y"]) != (dx, dy):
                    m = ("M", p["x"], p["y"], dx, dy)
                    if m not in have:
                        macros.append(m)
                        if len(macros) >= 24:
                            break
            if len(macros) >= 24:
                break
        if macros:
            self.log(f"blitz7: {len(macros)} pick->drop macros")
        return macros + acts

    def _send(self, act):
        if isinstance(act, tuple) and len(act) == 5 and act[0] == "M":
            self._pending_act = act
            _, x1, y1, x2, y2 = act
            r = self.env.step(GameAction.ACTION6, data={"x": x1, "y": y1})
            self.used += 1
            if r is None or r.state == GameState.GAME_OVER:
                return r
            if r.levels_completed != self.level:
                self.resp = r
                return r
            r2 = self.env.step(GameAction.ACTION6, data={"x": x2, "y": y2})
            self.used += 1
            return r2
        return super()._send(act)

    def _cls(self, act, f):
        if isinstance(act, tuple) and len(act) == 5 and act[0] == "M":
            return ("m", "pair")
        return super()._cls(act, f)


class BlitzExplorer8(BlitzExplorer7):
    """blitz7 + painter-demotion.

    novstats[cls] = [n_known_arrivals, n_trials]: how often did this action
    class land on an already-known graph state? A "painter" class (every
    trial changes pixels, ≥10 trials, NEVER arrives at a known state, e.g.
    ls20's action-2 paint) only ever explodes the frontier, so it is
    strongly demoted in the try-order. Demote, never prune.
    """

    PAINTER_TRIALS = 10
    PAINTER_DEMOTE = 0.05

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.novstats = {}
        self.log("blitz8: painter-demotion")

    def _new_level(self, f):
        self.novstats = {}
        super()._new_level(f)

    def _is_painter(self, cls):
        k, n = self.novstats.get(cls, (0, 0))
        return n >= self.PAINTER_TRIALS and k == 0

    def _stat_sort(self, acts, f):
        salt = getattr(self, "_salt", 0)
        scored = []
        for a in acts:
            cls = self._cls(a, f)
            ch, tr = self.stats.get(cls, (0, 0))
            base = -(ch + 1) / (tr + 2)
            if self._is_painter(cls):
                base *= self.PAINTER_DEMOTE
            h = int.from_bytes(
                hashlib.blake2b(repr((salt, repr(a))).encode(),
                                digest_size=8).digest(), "big")
            scored.append((base, h, a))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [a for _, _, a in scored]

    def step_once(self):
        # record novelty BEFORE the parent step mutates the graph
        g = self.graph
        cur_known = set(g.edges.keys())
        super().step_once()
        # figure out what happened: last transition tells (act, known?)
        if self.transitions:
            fb, act, fa, over, pb = self.transitions[-1]
            cls = self._cls(act, fb)
            s = self.novstats.setdefault(cls, [0, 0])
            s[1] += 1
            # b = resulting key; known if it was in the graph before this step
            b = self.key(fa, act if act != RESET else None) if not over else None
            if b is not None and b in cur_known:
                s[0] += 1


_AGENTS["blitz6"] = BlitzExplorer6
_AGENTS["blitz7"] = BlitzExplorer7
_AGENTS["blitz8"] = BlitzExplorer8
AGENTS = _AGENTS


class BlitzExplorer9(BlitzExplorer8):
    """blitz8 with GATED painter-demotion.

    Painter-demotion (and its expensive novelty snapshot) only engages
    when the state graph is actually exploding (>1500 nodes): that's when
    distractor actions are drowning the search. On normal games the
    behavior is identical to blitz7, preserving lucky fast solves.
    """

    GATE_NODES = 1500

    def _gated(self):
        return len(self.graph.edges) > self.GATE_NODES

    def _stat_sort(self, acts, f):
        if not self._gated():
            return super(BlitzExplorer8, self)._stat_sort(acts, f)
        return super()._stat_sort(acts, f)

    def step_once(self):
        if not self._gated():
            return super(BlitzExplorer8, self).step_once()
        return super().step_once()


_AGENTS["blitz9"] = BlitzExplorer9


class BlitzExplorer11(BlitzExplorer8):
    """blitz8 + adaptive ghost-seeking macros for pick-and-place click games.

    Probe finding (lf52, s5i5): clicking a token selects it and a GHOST
    outline appears at the target position. The winning macro is adaptive:
    click token -> perceive the newly-appeared ghost -> click the ghost.
    Static pick->drop pairs can't do this (ghost doesn't exist yet).

    Macro ("G", x, y): click (x,y); find new small components vs the frame
    before; click the most ghost-like one (small, new). Falls back to a
    plain single click if no ghost appears.
    """

    def actions_for(self, f):
        acts = super().actions_for(f)
        if 6 not in self.avail or "click" not in getattr(self, "tags", set()):
            return acts
        vals, cnts = np.unique(f, return_counts=True)
        bg = int(vals[np.argmax(cnts)])
        comps = components(f)
        picks = [c for c in salience_order(comps, bg)
                 if c["color"] != bg and c["size"] <= 64][:6]
        have = {a for a in acts
                if isinstance(a, tuple) and len(a) == 3 and a[0] == "G"}
        macros = []
        for p in picks:
            m = ("G", p["x"], p["y"])
            if m not in have:
                macros.append(m)
        if macros:
            self.log(f"blitz11: {len(macros)} ghost-seeking macros")
        return macros + acts

    def _send(self, act):
        if isinstance(act, tuple) and len(act) == 3 and act[0] == "G":
            self._pending_act = act
            _, x, y = act
            fb = self.cur_f
            r = self.env.step(GameAction.ACTION6, data={"x": x, "y": y})
            self.used += 1
            if r is None or r.state == GameState.GAME_OVER:
                return r
            if r.levels_completed != self.level:
                self.resp = r
                return r
            f1 = last_frame(r)
            # ghost = new small component that wasn't in fb
            import numpy as _np
            vals0, _ = _np.unique(fb, return_counts=True)
            bg0 = int(vals0[_np.argmax(_)]) if len(vals0) else 0
            before = {(c["x"], c["y"], c["color"]) for c in components(fb)}
            ghosts = [c for c in components(f1)
                      if (c["x"], c["y"], c["color"]) not in before
                      and c["size"] <= 32 and c["color"] != bg0]
            if ghosts:
                # most ghost-like: smallest new component
                g = min(ghosts, key=lambda c: c["size"])
                r2 = self.env.step(GameAction.ACTION6,
                                   data={"x": g["x"], "y": g["y"]})
                self.used += 1
                return r2
            return r
        return super()._send(act)

    def _cls(self, act, f):
        if isinstance(act, tuple) and len(act) == 3 and act[0] == "G":
            return ("g", "ghost")
        return super()._cls(act, f)


_AGENTS["blitz11"] = BlitzExplorer11


class BlitzExplorer12(BlitzExplorer8):
    """blitz8 with DELAYED painter-demotion.

    Painter-demotion only engages after 100 actions on a level. This
    preserves lucky fast solves (lp85's 5-action win) while still killing
    distractor explosions (ls20's paint, which drowns the search over
    hundreds of actions).
    """

    DEMOTE_DELAY = 100

    def _stat_sort(self, acts, f):
        if self.used - self.level_start_used < self.DEMOTE_DELAY:
            return super(BlitzExplorer8, self)._stat_sort(acts, f)
        return super()._stat_sort(acts, f)

    def step_once(self):
        if self.used - self.level_start_used < self.DEMOTE_DELAY:
            return super(BlitzExplorer8, self).step_once()
        return super().step_once()


_AGENTS["blitz12"] = BlitzExplorer12


class BlitzExplorer13(BlitzExplorer8):
    """blitz8 with REVERSIBILITY-gated painter-demotion.

    A painter is an action class whose pixel effects are IRREVERSIBLE:
    every trial changed pixels, >=10 trials, and NONE of its changed pixels
    ever changed back (via any action). Avatar moves are reversible (old
    position restored) -> not painters. Paint deposits are irreversible ->
    painters. This is precise enough to run ungated from action 0.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._paint_pix = {}   # cls -> set of flat pixel idx ever changed
        self._paint_rev = set()  # cls with >=1 pixel reverted
        self.log("blitz13: reversibility-gated painter")

    def _new_level(self, f):
        self._paint_pix = {}
        self._paint_rev = set()
        super()._new_level(f)

    def _is_painter(self, cls):
        # override: reversibility gate instead of pure novelty
        if "nopainter" in self.flags:
            return False
        if cls in self._paint_rev:
            return False
        px = self._paint_pix.get(cls)
        if px is None or len(px) == 0:
            return False
        k, n = self.novstats.get(cls, (0, 0))
        return n >= 10 and k == 0

    def step_once(self):
        super().step_once()
        # update reversibility: which classes' pixels just changed again?
        if self.transitions:
            fb, act, fa, over, pb = self.transitions[-1]
            cls = self._cls(act, fb)
            d = (fb != fa)
            if d.any():
                idx = set(np.flatnonzero(d).tolist())
                s = self._paint_pix.setdefault(cls, set())
                # pixels in idx that were previously changed by ANY class
                # and now change again -> reversible
                for c2, px2 in self._paint_pix.items():
                    if idx & px2:
                        self._paint_rev.add(c2)
                s |= idx


_AGENTS["blitz13"] = BlitzExplorer13
