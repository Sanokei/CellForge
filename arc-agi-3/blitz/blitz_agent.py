"""Blitz agents: tiny hand-authored ARC-AGI-3 players.

Subclass explorer5.Explorer. All logic is small authored cells / tables:
no gradient training, no learned models, no LLM calls at runtime.

blitz1 ideas (from paper intel, kept tiny):
- Sakana/Doc-to-LoRA: a tiny StrategyCell maps the game descriptor
  (tags, action space, initial frame stats) -> a few policy knobs.
  Game-conditioned weights, still a hand-written table.
- Jev-Mem System-One: fast structured rules own budget allocation.
  Per-level ATTEMPT loop: on full stall, RESET to the level root and
  re-explore with a reshuffled action ordering (diverse restarts beat one
  3000-action wander). Attempt count capped; expanded action sets on later
  attempts (coarse click lattice for click games).
- Harness-Zero: heuristics live in small cells (StrategyCell) with a few
  knobs, not scattered harness code.
- UWS: knob search stays in a tiny subspace (one dict of ~5 scalars).
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
from arcengine import GameAction, GameState

from explorer5 import Explorer as Explorer5, RESET
from cells import last_frame, components, click_targets


def strategy_cell(tags, avail, frame):
    """Game descriptor -> policy knobs. Pure table lookup, ~6 scalars."""
    tags = set(tags or [])
    has_click = 6 in avail
    kb = [a for a in avail if a not in (0, 6)]
    k = dict(attempt_budget=600, max_attempts=5, lattice_attempt=99)
    if has_click and (not kb or "click" in tags):
        k.update(attempt_budget=700, max_attempts=5, lattice_attempt=2)
    elif not has_click or tags == {"keyboard"}:
        k.update(attempt_budget=500, max_attempts=6, lattice_attempt=99)
    else:
        k.update(attempt_budget=650, max_attempts=5, lattice_attempt=3)
    return k


class BlitzExplorer(Explorer5):
    """explorer5 + attempt loop with diverse restarts + game-conditioned knobs."""

    def __init__(self, env, budget, seed=0, log=None, flags=(), tags=()):
        super().__init__(env, budget, seed=seed, log=log, flags=flags)
        self.tags = set(tags or [])
        self.knobs = strategy_cell(self.tags, self.avail, self.cur_f)
        self.attempt = 0
        self._salt = 0
        self.attempt_start = 0
        self.log("blitz knobs", self.knobs, "tags", sorted(self.tags),
                 "avail", self.avail)

    def _stat_sort(self, acts, f):
        salt = getattr(self, "_salt", 0)
        if not salt:
            return super()._stat_sort(acts, f)
        scored = []
        for a in acts:
            ch, tr = self.stats.get(self._cls(a, f), (0, 0))
            s = -(ch + 1) / (tr + 2)
            h = hashlib.blake2b(repr((salt, a)).encode(), digest_size=8).digest()
            scored.append((s, int.from_bytes(h, "big"), a))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [a for _, _, a in scored]

    def actions_for(self, f):
        acts = super().actions_for(f)
        knobs = getattr(self, "knobs", None)
        attempt = getattr(self, "attempt", 0)
        if knobs is not None and 6 in self.avail and attempt >= knobs["lattice_attempt"]:
            have = {(a[1], a[2]) for a in acts if a[0] == 6}
            extra = 0
            for x, y in click_targets(f, 1):
                if (x, y) not in have:
                    acts.append((6, x, y))
                    extra += 1
            if extra:
                self.log(f"blitz: attempt {attempt} added {extra} lattice clicks")
        return acts

    def _fresh_attempt(self):
        k = self.knobs
        self.attempt += 1
        self._salt = self.rng.randrange(1 << 30)
        self.log(f"blitz: attempt {self.attempt}/{k["max_attempts"]} "
                 f"reset+reshuffle salt={self._salt}")
        r = self._send(RESET)
        if r is None or not r.frame:
            return None
        self.resp = r
        self.avail = list(r.available_actions) or self.avail
        self.cur_f = last_frame(r)
        if r.levels_completed != self.level:
            self.level = r.levels_completed
            self.attempt = 0
        self.attempt_start = self.used
        self._new_level(self.cur_f)
        u = self._next_untested()
        return u if u is not None else RESET

    def _stalled(self):
        if self.attempt < self.knobs["max_attempts"]:
            return self._fresh_attempt()
        return super()._stalled()

    def run(self):
        k = self.knobs
        while self.used < self.budget and self.resp.state != GameState.WIN:
            lvl = self.level
            self.step_once()
            if self.level != lvl:
                self.attempt = 0
                self._salt = 0
                self.attempt_start = self.used
                self.log(f"blitz: level {lvl} -> {self.level}, attempts reset")
            elif self.used - self.attempt_start >= k["attempt_budget"]:
                self.log(f"blitz: attempt budget {k["attempt_budget"]} hit, cycling")
                self._fresh_attempt()
        return self.used


AGENTS = {"blitz1": BlitzExplorer}


class BlitzExplorer2(BlitzExplorer):
    """blitz1 + Jev-Mem progress-aware attempt cycling.

    New: _last_progress tracks the last action that produced evidence of
    progress (novel graph state, director suggestion, level-up). When no
    progress for progress_window actions, cycle to a fresh attempt instead
    of wandering. On fresh attempts the director keeps its learned models
    (effects/walls/goals) but resets per-level counters.
    """

    def __init__(self, env, budget, seed=0, log=None, flags=(), tags=()):
        super().__init__(env, budget, seed=seed, log=log, flags=flags, tags=tags)
        self.knobs.setdefault("progress_window", 400)
        self._last_progress = 0
        d = self.director
        if d is not None:
            orig_suggest = d.suggest

            def suggest(frame, mask, level):
                r = orig_suggest(frame, mask, level)
                if r is not None:
                    self._last_progress = self.used
                return r

            d.suggest = suggest

    def step_once(self):
        g_before = self.graph
        n_before = len(g_before.edges)
        super().step_once()
        # novel graph state discovered -> progress
        if self.graph is g_before and len(self.graph.edges) > n_before:
            self._last_progress = self.used

    def _fresh_attempt(self):
        # reset director per-level state, keep learned models
        d = self.director
        if d is not None:
            d.lvl_used = 0
            d.target = None
            d.av_pos = None
            d.visited = {(lv, a) for (lv, a) in d.visited if lv != self.level}
            if hasattr(d, "goals"):
                d.goals.done = {k for k in d.goals.done if k[0] != self.level}
        self._last_progress = self.used
        return super()._fresh_attempt()

    def run(self):
        k = self.knobs
        pw = k["progress_window"]
        while self.used < self.budget and self.resp.state != GameState.WIN:
            lvl = self.level
            self.step_once()
            if self.level != lvl:
                self.attempt = 0
                self._salt = 0
                self.attempt_start = self.used
                self._last_progress = self.used
                self.log(f"blitz2: level {lvl} -> {self.level}, attempts reset")
            elif self.used - self._last_progress >= pw:
                self.log(f"blitz2: no progress for {pw} actions, cycling attempt")
                self._fresh_attempt()
        return self.used


AGENTS["blitz2"] = BlitzExplorer2


class BlitzExplorer3(BlitzExplorer2):
    """blitz2 + Luby-style attempt budgets optimized for quadratic scoring.

    Score per level = ((baseline/actions)^2)*100, so SPEED dominates.
    Attempt budgets follow a front-loaded geometric sequence: many short
    attempts (multiple "lucks" at fast solutions) with longer fallbacks.
    Progress window tightened to 150.
    """

    LUBY_BUDGETS = [120, 120, 240, 480, 960, 1920]

    def __init__(self, env, budget, seed=0, log=None, flags=(), tags=()):
        super().__init__(env, budget, seed=seed, log=log, flags=flags, tags=tags)
        self.knobs["progress_window"] = 150
        self._attempt_used_start = 0

    def _attempt_budget(self):
        idx = min(self.attempt, len(self.LUBY_BUDGETS) - 1)
        return self.LUBY_BUDGETS[idx]

    def _fresh_attempt(self):
        self._attempt_used_start = self.used
        return super()._fresh_attempt()

    def run(self):
        k = self.knobs
        pw = k["progress_window"]
        self._attempt_used_start = 0
        while self.used < self.budget and self.resp.state != GameState.WIN:
            lvl = self.level
            self.step_once()
            if self.level != lvl:
                self.attempt = 0
                self._salt = 0
                self.attempt_start = self.used
                self._attempt_used_start = self.used
                self._last_progress = self.used
                self.log(f"blitz3: level {lvl} -> {self.level}, attempts reset")
            else:
                ab = self._attempt_budget()
                if self.used - self._attempt_used_start >= ab:
                    self.log(f"blitz3: attempt budget {ab} hit, cycling")
                    self._fresh_attempt()
                elif self.used - self._last_progress >= pw:
                    self.log(f"blitz3: no progress for {pw} actions, cycling")
                    self._fresh_attempt()
        return self.used


AGENTS["blitz3"] = BlitzExplorer3


class BlitzExplorer4(BlitzExplorer):
    """Clean restart strategy: stall-triggered diverse restarts ONLY.

    Removes the fixed attempt_budget cycling (it disrupts systematic
    exploration). Attempts end when the state graph is fully explored
    (stall), then RESET + reshuffle for a diverse retry. The director
    provides smart actions; restarts provide ordering diversity.
    """

    def run(self):
        while self.used < self.budget and self.resp.state != GameState.WIN:
            lvl = self.level
            self.step_once()
            if self.level != lvl:
                self.attempt = 0
                self._salt = 0
                self.attempt_start = self.used
                self.log(f"blitz4: level {lvl} -> {self.level}, attempts reset")
        return self.used


AGENTS["blitz4"] = BlitzExplorer4


class BlitzExplorer5(BlitzExplorer4):
    """blitz4 + click-pair macros for pick-and-drop games (lf52-class).

    From probeB: lf52 level 0 requires click token (pick up) then click cell
    (drop) as an ordered 2-step macro. The flat action graph never tries
    dependent pairs efficiently. This adds ordered click-pair macros as
    atomic actions for click games.
    """

    def actions_for(self, f):
        acts = super().actions_for(f)
        if 6 in self.avail and getattr(self, "attempt", 0) >= 1:
            # top salient click targets
            clicks = [(a[1], a[2]) for a in acts if a[0] == 6][:6]
            have = set()
            for a in acts:
                if isinstance(a, tuple) and len(a) == 5 and a[0] == "M":
                    have.add((a[1], a[2], a[3], a[4]))
            added = 0
            for x1, y1 in clicks:
                for x2, y2 in clicks:
                    if (x1, y1) != (x2, y2) and (x1, y1, x2, y2) not in have:
                        acts.append(("M", x1, y1, x2, y2))
                        added += 1
                        if added >= 30:
                            break
                if added >= 30:
                    break
            if added:
                self.log(f"blitz5: added {added} click-pair macros")
        return acts

    def _send(self, act):
        if isinstance(act, tuple) and len(act) == 5 and act[0] == "M":
            _, x1, y1, x2, y2 = act
            r = self.env.step(GameAction.ACTION6, data={"x": x1, "y": y1})
            self.used += 1
            if r is None or r.state == GameState.GAME_OVER:
                return r
            # check level-up after first click
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


AGENTS["blitz5"] = BlitzExplorer5
