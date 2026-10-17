"""Director: composes the authored cells into a goal-directed policy layered on the v5 graph explorer.

The explorer keeps its state-graph exploration and is consulted whenever the director has no suggestion.
Director API used by explorer5.Explorer:  observe(fb, act, fa, levelup, over, mask, level)  /  suggest(frame, mask, level).
"""
from __future__ import annotations

import numpy as np

from effects import EffectTable
from goals import ClickSequence, TouchGoals
from perceive import match_moves, segment
from planner import plan_to
from walls import WallModel, dest_colors


RESET_ID = -99


class Director:
    def __init__(self, has_click: bool, level_budget: int = 600, enabled=("avatar", "click")):
        self.has_click = has_click
        self.level_budget = level_budget
        self.enabled = set(enabled)
        self.fx = EffectTable()
        self.walls = WallModel()
        self.goals = TouchGoals()
        self.clicks = ClickSequence()
        self.target = None
        self.lvl_used = 0
        self.av_pos = None
        self.visited = set()        # (level, anchor) positions the avatar has occupied
        self._objs = None          # cached segmentation of the current frame
        self._objs_key = None
        self.ver = 0
        self._unreach_ver = -1
        self.stats = dict(directed=0, plans=0, replays=0)

    # -- helpers ---------------------------------------------------------------
    def _seg(self, frame, mask):
        key = frame.tobytes()
        if self._objs_key != key:
            self._objs = segment(frame, mask)
            self._objs_key = key
        return self._objs

    def _avatar_obj(self, objs, cls):
        opts = [o for o in objs if o["cls"] == cls]
        if not opts:
            return None
        if self.av_pos is not None and len(opts) > 1:
            opts.sort(key=lambda o: abs(o["bbox"][0] - self.av_pos[1]) + abs(o["bbox"][1] - self.av_pos[0]))
        return opts[0]

    # -- learning ----------------------------------------------------------------
    def observe(self, fb, act, fa, levelup, over, mask, level):
        ob, oa = self._seg(fb, mask), segment(fa, mask)
        if act[0] != 6:
            moves, vanished, appeared = match_moves(ob, oa)
            noop = bool((np.where(mask, 0, fb) == np.where(mask, 0, fa)).all())
            av_before = self.fx.avatar()
            self.fx.observe(act[0], moves, noop)
            if av_before is not None and act[0] in av_before[1]:
                cls, vecs = av_before
                o = self._avatar_obj(ob, cls)
                if o is not None:
                    vec = vecs[act[0]]
                    cols, oob = dest_colors(fb, o["pix"], vec)
                    moved = any(m[0] == cls and m[1] == vec for m in moves)
                    anchor = (o["bbox"][1], o["bbox"][0])
                    before = (frozenset(self.walls.blocked), frozenset(self.walls.hazard))
                    if oob and not moved:
                        self.walls.edges.add((level, anchor, act[0]))
                    else:
                        self.walls.learn(level, anchor, act[0], cols, moved, died=over)
                    if before != (frozenset(self.walls.blocked), frozenset(self.walls.hazard)):
                        self.ver += 1
                    if moved:
                        self.av_pos = (anchor[0] + vec[0], anchor[1] + vec[1])
                    self._touch_outcome(o, vec, moved, oa, levelup, level)
        else:
            x, y = act[1], act[2]
            cls = None
            for o in ob:
                if (y, x) in o["pix"]:
                    cls = (o["color"], o["size"])
                    break
            changed = bool((np.where(mask, 0, fb) != np.where(mask, 0, fa)).any())
            self.clicks.record((x, y), cls if cls else (-1, -1), changed and not levelup)
            if levelup:
                self.clicks.record((x, y), cls if cls else (-1, -1), True)
        if not over and not levelup:
            av = self.fx.avatar()
            if (av is None or act[0] not in av[1]) and not np.array_equal(np.where(mask, 0, fb), np.where(mask, 0, fa)):
                # a non-movement action changed the world: old coverage / touch verdicts are stale
                self.visited = {v for v in self.visited if v[0] != level}
                self.goals.done = {k for k in self.goals.done if k[0] != level}
                self.ver += 1
        if levelup:
            d = np.where(mask, False, fb != fa)
            ys, xs = np.nonzero(d)
            self.win_log = getattr(self, "win_log", []) + [dict(level=level, act=act, avpos=self.av_pos, target=(self.target or {}).get("cls"),
                     tbbox=(self.target or {}).get("bbox"), changed=int(d.sum()), box=(ys.min(), xs.min(), ys.max(), xs.max()) if d.any() else None,
                     before=sorted(set(fb[d].tolist())), after=sorted(set(fa[d].tolist())))]
            if self.target is not None:
                self.goals.outcome(level, self.target, "win")
            self.target = None
            self.clicks.new_level(True)
            self.lvl_used = 0
            self.av_pos = None
            self.ver += 1
        else:
            self.lvl_used += 1
        self._objs, self._objs_key = oa, fa.tobytes()

    def _touch_outcome(self, o, vec, moved, oa, levelup, level):
        t = self.target
        if t is None:
            return
        dest = {(y + vec[0], x + vec[1]) for (y, x) in o["pix"]}
        if not (dest & t["pix"]):
            return
        if levelup:
            return          # credited as a win by the caller
        if not moved:
            self.goals.outcome(level, t, "solid")
        else:
            still = any(p["color"] == t["color"] and p["bbox"] == t["bbox"] for p in oa)
            self.goals.outcome(level, t, "inert" if still else "vanish")
        self.target = None

    # -- acting -------------------------------------------------------------------
    def suggest(self, frame, mask, level):
        if self.lvl_used >= self.level_budget:
            return None
        objs = self._seg(frame, mask)
        if "avatar" in self.enabled:
            a = self._suggest_avatar(frame, objs, level)
            if a is not None:
                self.stats["directed"] += 1
                return a
        if "click" in self.enabled and self.has_click:
            xy = self.clicks.suggest(objs)
            if xy is not None:
                self.stats["replays"] += 1
                return (6, xy[0], xy[1])
        return None

    def _suggest_avatar(self, frame, objs, level):
        av = self.fx.avatar()
        if av is None:
            self._why("no_avatar")
            return None
        cls, vecs = av
        me = self._avatar_obj(objs, cls)
        if me is None:
            self._why("avatar_not_found")
            return None
        ay0, ax0 = me["bbox"][1], me["bbox"][0]
        if self._unreach_ver != self.ver:
            self.goals.unreachable.clear()
            self._unreach_ver = self.ver
        bad = self.walls.blocked | self.walls.hazard
        self.visited.add((level, (ay0, ax0)))
        ranked = self.goals.pick(objs, cls, level, bad | (self.walls.floor() - set(c for c, k in self.goals.win.items() if k)), None)
        if not ranked:
            return self._explore(frame, me, vecs, level)
        ay, ax = me["bbox"][1], me["bbox"][0]
        best_tier, best = None, None
        tried = 0
        for tier, _, o in sorted(ranked, key=lambda t: (t[0], abs(t[2]["bbox"][0] - ax) + abs(t[2]["bbox"][1] - ay))):
            if best_tier is not None and tier != best_tier:
                break
            if tried >= 8:
                break
            tried += 1
            path = plan_to(frame, me["pix"], vecs, self.walls, o["pix"], level, allow={o["color"]})
            if path is None:
                self.goals.unreachable.add((level, o["color"], o["bbox"]))
                continue
            if not path:
                self.goals.outcome(level, o, "inert")
                continue
            if best is None or len(path) < len(best[1]):
                best, best_tier = (o, path), tier
        if best is None:
            return self._explore(frame, me, vecs, level)
        self.target = best[0]
        self.stats["plans"] += 1
        return (best[1][0], -1, -1)

    def _explore(self, frame, me, vecs, level):
        """No reachable goal hypothesis: walk to the nearest avatar position not yet visited on this level."""
        seen = {a for (l, a) in self.visited if l == level}
        path = plan_to(frame, me["pix"], vecs, self.walls, frozenset(), level, unvisited=seen)
        if not path:
            self._why("nothing_to_explore")
            return None
        self.target = None
        self.stats["explore"] = self.stats.get("explore", 0) + 1
        return (path[0], -1, -1)

    def _why(self, k):
        self.stats[k] = self.stats.get(k, 0) + 1
