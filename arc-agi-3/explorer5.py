"""v5 (flags nm,ck,gv,pr) of the training-free directed state-action graph explorer (pure algorithmic).

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

from cells import VolatileMask, click_targets, last_frame, state_key, components
import numpy as np
from collections import deque as _dq

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

    def bfs_to_frontier(self, src, valfn=None):
        """Shortest action path from src to a live node with untested actions.
        With valfn: among frontier nodes within 2*dmin+8 steps pick the highest value (ties: nearest)."""
        prev = {src: None}
        dist = {src: 0}
        cands = []
        q = deque([src])
        while q:
            k = q.popleft()
            if self.untested.get(k) and k not in self.dead:
                if valfn is None:
                    cands.append(k); break
                cands.append(k)
                if dist[k] > 2 * dist[cands[0]] + 8:
                    break
            nxt = dict(self.edges[k])
            if k != self.root and k not in self.dead and self.root is not None:
                nxt.setdefault(RESET, self.root)
            if k in self.dead:
                nxt = {RESET: self.root}
            for a, n in nxt.items():
                if n not in prev and n in self.edges:
                    prev[n] = (k, a)
                    dist[n] = dist[k] + 1
                    q.append(n)
        if not cands:
            return None
        best = cands[0] if valfn is None else max(cands, key=lambda c: (valfn(c), -dist[c]))
        k = best
        path = []
        while prev[k] is not None:
            k, a = prev[k]
            path.append(a)
        return path[::-1]




class Explorer:
    """flags: nm = non-Markov handling (add last action to the key after repeated (state,action) conflicts),
    ck = click-class statistics keyed by (colour,size bucket), gv = goal-event value bias transferred across levels,
    pr = prune action classes with zero observed effect. All counters live inside the episode."""

    def __init__(self, env, budget, seed=0, log=None, flags=("pr",)):
        self.env, self.budget, self.rng = env, budget, random.Random(seed)
        self.log = log or (lambda *a: None)
        self.flags = set(flags)
        self.used = 0
        self.resp = env.observation_space
        self.level = self.resp.levels_completed
        self.avail = list(self.resp.available_actions)
        self.vol = VolatileMask()
        self.stall_level = 0
        self.stats = {}
        self.tinfo = {}            # click (x,y) -> (colour, size) from latest perception
        self.hist = False          # non-Markov mode: last action id in the state key
        self.conflicts = 0
        self.sig = np.zeros(16)    # goal-event signature (summed sign of colour-count change before level-ups)
        self.nsig = 0
        self.recent = deque(maxlen=10)
        self.nh = {}
        self.cur_last = None
        self.level_start_used = 0
        self.level_paths = []
        self.director = None
        if "d6" in self.flags:
            import os, sys
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v6"))
            from director import Director
            self.director = Director(has_click=6 in self.avail)
        self._new_level(last_frame(self.resp))

    # -- cells ---------------------------------------------------------------
    def key(self, f, last=None):
        k = state_key(f, self.vol.mask)
        if self.hist and last is not None:
            k += bytes([0 if last[0] == "R" else last[0] + 1])
        return k

    def hist_of(self, f):
        return np.bincount(np.where(self.vol.mask, 16, f).ravel(), minlength=17)[:16]

    def actions_for(self, f):
        acts = [(i, -1, -1) for i in self.avail if i not in (0, 6)]
        if 6 in self.avail:
            lv = self.stall_level if self.stall_level <= 2 else 2
            if "ck" in self.flags and lv == 0:
                from cells import salience_order
                comps = components(f)
                vals, cnts = np.unique(f, return_counts=True)
                bg = int(vals[np.argmax(cnts)])
                seen = set()
                for c in salience_order(comps, bg):
                    t = (c["x"], c["y"])
                    if t not in seen:
                        seen.add(t)
                        self.tinfo[t] = (c["color"], c["size"])
                        acts.append((6, t[0], t[1]))
            else:
                acts += [(6, x, y) for x, y in click_targets(f, lv)]
        return self._stat_sort(acts, f)

    def _cls(self, act, f):
        if act[0] != 6:
            return ("k", act[0])
        if "ck" in self.flags and (act[1], act[2]) in self.tinfo:
            c, sz = self.tinfo[(act[1], act[2])]
            return ("c", c, sz.bit_length())
        return ("c", int(f[act[2], act[1]]))

    def _stat_sort(self, acts, f):
        def score(a):
            ch, tr = self.stats.get(self._cls(a, f), (0, 0))
            return -(ch + 1) / (tr + 2)
        return sorted(acts, key=score)

    def _dead_class(self, act, f):
        if "pr" not in self.flags:
            return False
        ch, tr = self.stats.get(self._cls(act, f), (0, 0))
        return ch == 0 and tr >= (6 if act[0] != 6 else 10)

    def _value(self, k):
        h = self.nh.get(k)
        if h is None or self.nsig == 0:
            return 0.0
        w = np.sign(self.sig) * (np.abs(self.sig) >= 0.5 * self.nsig)
        return float(np.dot(w, h))

    def _new_level(self, f):
        self.graph = Graph()
        self.transitions = []   # (frame_before, act, frame_after, gameover, last_act_before)
        self.cur_f = f
        self.cur_last = None
        self.stall_level = 0
        self.plan = []
        self.nh = {}
        self.conflicts = 0
        k = self.key(f)
        self.graph.root = k
        self.graph.add_node(k, self.actions_for(f))
        self.nh[k] = self.hist_of(f)
        self.cur = k

    def _rebuild(self):
        g = Graph()
        trs = self.transitions
        if not trs:
            return
        self.nh = {}
        first = trs[0][0]
        k0 = self.key(first, trs[0][4])
        g.root = k0
        g.add_node(k0, self.actions_for(first))
        for fb, act, fa, over, pb in trs:
            a = self.key(fb, pb)
            b = DEAD if over else self.key(fa, act)
            g.add_node(a, self.actions_for(fb))
            g.add_node(b, self.actions_for(fa))
            self.nh.setdefault(a, self.hist_of(fb))
            self.nh.setdefault(b, self.hist_of(fa))
            g.record(a, act, b)
        self.graph = g
        self.cur = DEAD if self.resp.state == GameState.GAME_OVER else self.key(self.cur_f, self.cur_last)
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

    def _next_untested(self):
        u = self.graph.untested[self.cur]
        while u and self._dead_class(u[0], self.cur_f):
            u.pop(0)
        return u[0] if u else None

    def step_once(self):
        g = self.graph
        if self.resp.state == GameState.GAME_OVER:
            act = RESET
        elif self.plan:
            act = self.plan.pop(0)
        elif self.director is not None and (dsg := self.director.suggest(self.cur_f, self.vol.mask, self.level)) is not None:
            act = dsg
        elif self.cur not in g.dead and self._next_untested() is not None:
            act = g.untested[self.cur][0]
        else:
            vf = self._value if ("gv" in self.flags and self.nsig) else None
            path = g.bfs_to_frontier(self.cur, vf)
            if path:
                self.plan = path
                act = self.plan.pop(0)
            else:
                act = self._stalled()
                if act is None:
                    return
        g = self.graph
        expected = g.edges[self.cur].get(act) if act != RESET else None
        fb = self.cur_f
        r = self._send(act)
        if r is None or not r.frame:
            return
        fa = last_frame(r)
        self.resp = r
        self.avail = list(r.available_actions) or self.avail
        over = r.state == GameState.GAME_OVER
        if self.director is not None and act != RESET:
            self.director.observe(fb, act, fa, r.levels_completed != self.level, over, self.vol.mask, self.level)
        if r.levels_completed != self.level:
            self.log("level", self.level, "->", r.levels_completed, "used", self.used)
            if len(self.recent) >= 3:
                d = self.recent[-1].astype(int) - self.recent[0].astype(int)
                self.sig += np.sign(d)
                self.nsig += 1
                self.log("sig", d.tolist())
            self.level_paths.append((self.level, self.used - self.level_start_used))
            self.level_start_used = self.used
            self.level = r.levels_completed
            self.recent.clear()
            self._new_level(fa)
            return
        pb = self.cur_last
        last = None if act == RESET else act
        if self.vol.observe(fb, fa):
            self.cur_f = fa
            self.transitions.append((fb, act, fa, over, pb))
            self.cur_last = last
            self._rebuild()
            self.log("mask grew", int(self.vol.mask.sum()))
            return
        a = self.cur
        b = DEAD if over else self.key(fa, last)
        g.add_node(b, self.actions_for(fa))
        if b not in self.nh:
            self.nh[b] = self.hist_of(fa)
        self.recent.append(self.hist_of(fa))
        if act == RESET:
            if b != g.root:
                self.log("reset landed on non-root state")
                g.root = b
            self.recent.clear()
        else:
            old = g.edges[a].get(act)
            g.record(a, act, b)
            self.transitions.append((fb, act, fa, over, pb))
            s = self.stats.setdefault(self._cls(act, fb), [0, 0])
            s[1] += 1
            s[0] += int(not np.array_equal(fb, fa))
            if "nm" in self.flags and old is not None and old != b and not self.hist:
                self.conflicts += 1
                if self.conflicts >= 5:
                    self.hist = True
                    self.log("non-Markov: adding last action to key")
                    self.cur_f, self.cur_last = fa, last
                    self._rebuild()
                    return
        if expected is not None and act != RESET and expected != b and self.plan:
            self.plan = []
        self.cur, self.cur_f, self.cur_last = b, fa, last

    def _stalled(self):
        if 6 in self.avail and self.stall_level < 2:
            self.stall_level += 1
            self.log("stall: widening click lattice to", self.stall_level)
            for k, f in list(self._node_frames().items()):
                have = set(self.graph.edges[k]) | set(self.graph.untested[k])
                self.graph.untested[k] += [a for a in self.actions_for(f) if a not in have]
            return self.graph.untested[self.cur][0] if self.graph.untested[self.cur] else RESET
        self.log("stall: forgetting graph")
        self.stall_level = 0
        self._new_level(self.cur_f)
        acts = [a for a in self.actions_for(self.cur_f)]
        return self.rng.choice(acts) if acts else RESET

    def _node_frames(self):
        d = {}
        for fb, act, fa, over, pb in self.transitions:
            d.setdefault(self.key(fb, pb), fb)
            d.setdefault(self.key(fa, act), fa)
        d.setdefault(self.cur, self.cur_f)
        return {k: f for k, f in d.items() if k in self.graph.edges}

    def run(self):
        while self.used < self.budget and self.resp.state != GameState.WIN:
            self.step_once()
        return self.used
