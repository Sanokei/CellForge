"""Training-free directed state-action graph explorer (pure algorithmic).

Per level: nodes = masked-frame hashes, edges = tested (state, action) -> state.
Policy: take an untested action at the current state (priority ordered); otherwise BFS the
known graph (incl. a RESET edge back to the level root) to the nearest state that still has
untested actions and walk there. Level change => fresh graph. Deterministic given a seed.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
from arcengine import GameAction, GameState

from cells import VolatileMask, click_targets, last_frame, state_key

RESET = ("R", -1, -1)
DEAD = b"DEAD"   # shared sink for game-over arrivals (keeps hidden-energy states from being poisoned)


class Graph:
    def __init__(self):
        self.root = None
        self.edges = {DEAD: {}}      # key -> {action: next_key}
        self.dead = {DEAD}   # game-over sink
        self.untested = {DEAD: []}   # key -> list of actions (priority order)

    def add_node(self, key, actions):
        if key not in self.edges:
            self.edges[key] = {}
            self.untested[key] = list(actions)

    def record(self, a, act, b):
        self.edges[a][act] = b
        if act in self.untested[a]:
            self.untested[a].remove(act)

    def bfs_to_frontier(self, src):
        """Shortest action path from src to a live node with untested actions."""
        prev = {src: None}
        q = deque([src])
        while q:
            k = q.popleft()
            if self.untested.get(k) and k not in self.dead:
                path = []
                while prev[k] is not None:
                    k, a = prev[k]
                    path.append(a)
                return path[::-1]
            nxt = dict(self.edges[k])
            if k != self.root and k not in self.dead and self.root is not None:
                nxt.setdefault(RESET, self.root)
            if k in self.dead:
                nxt = {RESET: self.root}
            for a, n in nxt.items():
                if n not in prev and n in self.edges:
                    prev[n] = (k, a)
                    q.append(n)
        return None


class Explorer:
    def __init__(self, env, budget, seed=0, log=None, use_action_stats=False):
        self.env, self.budget, self.rng = env, budget, random.Random(seed)
        self.log = log or (lambda *a: None)
        self.use_stats = use_action_stats
        self.used = 0
        self.resp = env.observation_space
        self.level = self.resp.levels_completed
        self.avail = list(self.resp.available_actions)
        self.vol = VolatileMask()
        self.stall_level = 0
        self.stats = {}  # action class -> [changed, tried]  (simple counters, used only if use_stats)
        self._new_level(last_frame(self.resp))
        self.history = []

    # -- cells ---------------------------------------------------------------
    def key(self, f):
        return state_key(f, self.vol.mask)

    def actions_for(self, f):
        acts = [(i, -1, -1) for i in self.avail if i not in (0, 6)]
        if 6 in self.avail:
            acts += [(6, x, y) for x, y in click_targets(f, self.stall_level if self.stall_level <= 2 else 2)]
        if self.use_stats:
            acts = self._stat_sort(acts, f)
        return acts

    def _cls(self, act, f):
        if act[0] != 6:
            return ("k", act[0])
        return ("c", int(f[act[2], act[1]]))

    def _stat_sort(self, acts, f):
        def score(a):
            ch, tr = self.stats.get(self._cls(a, f), (0, 0))
            return -(ch + 1) / (tr + 2)   # optimistic prior; stable sort keeps salience order ties
        return sorted(acts, key=score)

    def _new_level(self, f):
        self.graph = Graph()
        self.frames = {}
        self.transitions = []   # (frame_before, act, frame_after, gameover)
        self.cur_f = f
        self.stall_level = 0
        self.plan = []
        k = self.key(f)
        self.graph.root = k
        self.graph.add_node(k, self.actions_for(f))
        self.cur = k

    def _rebuild(self):
        """Mask changed: rehash all stored transitions."""
        g = Graph()
        trs = self.transitions
        if not trs:
            return
        first = trs[0][0]
        k0 = self.key(first)
        g.root = k0
        g.add_node(k0, self.actions_for(first))
        for fb, act, fa, over in trs:
            a = self.key(fb)
            b = DEAD if over else self.key(fa)
            g.add_node(a, self.actions_for(fb))
            g.add_node(b, self.actions_for(fa))
            g.record(a, act, b)
        self.graph = g
        self.cur = DEAD if self.resp.state == GameState.GAME_OVER else self.key(self.cur_f)
        self.plan = []

    # -- acting --------------------------------------------------------------
    def _send(self, act):
        if act == RESET:
            r = self.env.step(GameAction.RESET)
        elif act[0] == 6:
            r = self.env.step(GameAction.ACTION6, data={"x": act[1], "y": act[2]})
        else:
            r = self.env.step(GameAction.from_id(act[0]))
        self.used += 1
        return r

    def step_once(self):
        g = self.graph
        if self.resp.state == GameState.GAME_OVER:
            act = RESET
        elif self.plan:
            act = self.plan.pop(0)
        elif g.untested[self.cur] and self.cur not in g.dead:
            act = g.untested[self.cur][0]
        else:
            path = g.bfs_to_frontier(self.cur)
            if path:
                self.plan = path
                act = self.plan.pop(0)
            else:
                act = self._stalled()
                if act is None:
                    return
        g = self.graph
        expected = g.edges[self.cur].get(act) if act != RESET else (g.root if self.plan or True else None)
        fb = self.cur_f
        r = self._send(act)
        if r is None or not r.frame:
            return
        fa = last_frame(r)
        self.resp = r
        self.avail = list(r.available_actions) or self.avail
        over = r.state == GameState.GAME_OVER
        if r.levels_completed != self.level:
            self.log("level", self.level, "->", r.levels_completed, "used", self.used)
            self.level = r.levels_completed
            self._new_level(fa)
            return
        if self.vol.observe(fb, fa):
            self.cur_f = fa
            self.transitions.append((fb, act, fa, over))
            self._rebuild()
            self.log("mask grew", int(self.vol.mask.sum()))
            return
        a = self.cur
        b = DEAD if over else self.key(fa)
        g.add_node(b, self.actions_for(fa))
        if act == RESET:
            # reset returns to level start; do not treat as a normal edge
            if b != g.root:
                self.log("reset landed on non-root state")
                g.root = b
            g.untested[a] = [x for x in g.untested[a]]
        else:
            g.record(a, act, b)
            self.transitions.append((fb, act, fa, over))
            if self.use_stats:
                c = self._cls(act, fb)
                s = self.stats.setdefault(c, [0, 0])
                s[1] += 1
                s[0] += int(a != b)
        if expected is not None and act != RESET and expected != b and self.plan:
            self.plan = []      # nondeterminism / hidden state: replan
        self.cur, self.cur_f = b, fa

    def _stalled(self):
        """Graph fully explored without progress: widen click lattice, else restart graph/random walk."""
        if 6 in self.avail and self.stall_level < 2:
            self.stall_level += 1
            self.log("stall: widening click lattice to", self.stall_level)
            for k, f in list(self._node_frames().items()):
                have = set(self.graph.edges[k]) | set(self.graph.untested[k])
                self.graph.untested[k] += [a for a in self.actions_for(f) if a not in have]
            return self.graph.untested[self.cur][0] if self.graph.untested[self.cur] else RESET
        # forget graph (hidden state / nondeterminism) and continue from current state
        self.log("stall: forgetting graph")
        self.stall_level = 0
        self._new_level(self.cur_f)
        acts = [a for a in self.actions_for(self.cur_f)]
        return self.rng.choice(acts) if acts else RESET

    def _node_frames(self):
        d = {}
        for fb, act, fa, over in self.transitions:
            d.setdefault(self.key(fb), fb)
            d.setdefault(self.key(fa), fa)
        d.setdefault(self.cur, self.cur_f)
        return {k: f for k, f in d.items() if k in self.graph.edges}

    def run(self):
        while self.used < self.budget and self.resp.state != GameState.WIN:
            self.step_once()
        return self.used
