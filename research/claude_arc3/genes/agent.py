"""Push-contract agent composed of visual, evidence, and planning genes.

Only observations enter this module. No generator imports, oracle, genome, or env.
"""
from collections import Counter
import random

import numpy as np

from .perception import TileGrid, AvatarDetection, positions, singletons
from .learning import ActionEffects, TerrainLearning, GoalHypotheses, KeyDoorLearning, PushDetection, HazardMotion
from .planning import navigation, push_plan, ClickEffects


def state_name(obs):
    return getattr(obs.state, "name", str(obs.state).split(".")[-1])


class GeneAgent:
    """is_done(frames, latest_frame)/choose_action(frames, latest_frame).

    Optional disabled gene names support controlled ablations. Frame history is
    accepted for contract compatibility; bounded internal memory drives decisions.
    """
    def __init__(self, budget=3000, seed=0, disabled=()):
        self.budget, self.actions = budget, 0
        self.rng = random.Random(seed)
        self.disabled = set(disabled)
        self.level = None
        self.previous = None
        self.last = None
        self.events = Counter()
        self._fresh()

    def _fresh(self):
        self.tiles = None
        self.avatar = AvatarDetection()
        self.effects = ActionEffects()
        self.terrain = TerrainLearning()
        self.goals = GoalHypotheses()
        self.keys = KeyDoorLearning()
        self.push = PushDetection()
        self.clicks = ClickEffects()
        self.visits = Counter()
        self.probes = Counter()
        self.plan = []
        self.objects = {}
        self.wall = None
        self.pending_target = None
        self.push_trials = Counter()
        self.dynamic = set()
        self.motion = HazardMotion()
        self.push_target = None
        self.collected = []
        self.sequence_tests = Counter()
        self.depth = 0
        self.previous = None
        self.last = None

    def is_done(self, frames, latest_frame):
        return state_name(latest_frame) == "WIN" or self.actions >= self.budget

    def close(self):
        pass

    def _observe(self, grid, latest):
        if self.previous is None or self.last is None or self.last[0] == 0:
            return
        old = self.previous
        aid, click = self.last
        if aid == 6:
            self.clicks.observe(old, grid, click)
            return
        identity = self.avatar.color
        moved = self.avatar.observe(old, grid, aid)
        if identity is not None and identity != self.avatar.color:
            self.effects = ActionEffects()
            self.terrain = TerrainLearning()
            self.dynamic.clear()
            self.plan.clear()
        p, q = self.avatar.locate(old), self.avatar.locate(grid)
        if "effects" not in self.disabled:
            self.effects.observe(aid, p, q)
        mapping = self.effects.mapping()
        died = state_name(latest) == "GAME_OVER"
        if "terrain" not in self.disabled:
            self.terrain.observe(old, p, q, aid, mapping, died)
        if "goals" not in self.disabled:
            self.goals.observe(old, grid, p, q, self.avatar.color)
        if "keydoor" not in self.disabled:
            self.keys.observe(old, grid, p, q, self.avatar.color, self.terrain)
        if "push" not in self.disabled:
            self.push.observe(moved, self.avatar.color, p, q)
        for c, (b, b2) in moved.items():
            if c != self.avatar.color and b != q:
                self.dynamic.add(c)
        self.dynamic.discard(self.avatar.color)
        self.dynamic.difference_update(self.push.colors)
        self.motion.observe(grid, self.dynamic)
        if q is not None:
            self.visits[q] += 1
        if died:
            self.events["death"] += 1
            self.plan.clear()
        if p == q:
            self.plan.clear()
        if p is not None and q is not None and p != q:
            for c, pos in self.objects.items():
                if pos == p and c != self.avatar.color and not positions(grid, c):
                    if c not in self.collected:
                        self.collected.append(c)
                    self.events["collection"] += 1
            touched = int(old[q[1], q[0]])
            if touched in self.objects and touched != self.avatar.color:
                self.sequence_tests[(self.depth, touched)] += 1
        self.depth = sum(not positions(grid, c) for c in self.collected)
        # Object positions stay hypotheses until collected or moved.
        for c, pos in singletons(grid).items():
            if c != self.avatar.color:
                self.objects[c] = pos

    def _allowed(self, grid):
        return {(x, y) for y in range(grid.shape[0]) for x in range(grid.shape[1])
                if int(grid[y, x]) != self.wall and (x, y) not in self.terrain.dead
                and int(grid[y, x]) not in self.terrain.dead_colors}

    def _keyboard(self, grid, available):
        mapping = self.effects.mapping()
        p = self.avatar.locate(grid)
        keyboard = [a for a in available if a != 0 and a != 6]
        unknown = [a for a in keyboard if a not in mapping and a in (1, 2, 3, 4)]
        probe_now = unknown and (min(self.probes[a] for a in unknown) < 3 or self.actions % 4 == 0)
        if probe_now or p is None or "avatar" in self.disabled:
            pool = unknown or keyboard
            a = min(pool, key=lambda a: (self.probes[a], self.rng.random()))
            self.probes[a] += 1
            return a
        if self.plan:
            return self.plan.pop(0)
        allowed = self._allowed(grid)
        danger = self.motion.predicted() if "hazard_motion" not in self.disabled else set()
        if self.dynamic:
            # Moving hazards invalidate location-specific death labels on reset.
            allowed.update(self.terrain.dead)
        block_color = self.push.colors.most_common(1)[0][0] if self.push.colors else None
        blocks = positions(grid, block_color) if block_color is not None else []
        if blocks and "push" not in self.disabled:
            candidates = []
            for c, target in self.objects.items():
                if c in (self.avatar.color, block_color) or c in self.dynamic:
                    continue
                if c in self.goals.tested and blocks[0] == target:
                    self.push_target = None
                path = push_plan(p, blocks[0], target, mapping, allowed)
                if path:
                    candidates.append((0 if c == self.push_target else 1, self.push_trials[c], len(path), c, path))
            if candidates:
                _, _, _, c, path = min(candidates)
                if self.push_target != c:
                    self.push_trials[c] += 1
                self.push_target = c
                # Replan on each real observation, avoiding stale push state.
                self.events["push_plan"] += 1
                return path[0]
            if self.actions % 20 == 0:
                self.push_target = None
                self.terrain.blocked.clear()
                self.events["deadlock_reset"] += 1
                return 0
        candidates = []
        for c, target in self.objects.items():
            if c == self.avatar.color or c in self.dynamic or c in self.terrain.dead_colors:
                continue
            if not positions(grid, c):
                continue
            path = navigation(p, target, mapping, allowed, self.terrain.blocked)
            if path:
                dx, dy = mapping[path[0]]
                if (p[0] + dx, p[1] + dy) in danger:
                    continue
                replay = self.collected[self.depth] if self.depth < len(self.collected) else None
                priority = -1 if c == replay else self.sequence_tests[(self.depth, c)]
                candidates.append((priority, len(path), c, path))
        if candidates and "goals" not in self.disabled:
            _, _, c, path = min(candidates)
            self.events["goal_test"] += 1
            return path[0]
        # Frontier exploration also contacts movable objects before their role is known.
        choices = []
        for a, (dx, dy) in mapping.items():
            q = p[0] + dx, p[1] + dy
            if a in available and q in allowed and q not in danger and (p, q) not in self.terrain.blocked:
                choices.append((self.visits[q], self.rng.random(), a))
        return min(choices)[2] if choices else self.rng.choice(keyboard)

    def choose_action(self, frames, latest_frame):
        # Runtime import allows either the real SDK or a caller-installed test shim.
        from arcengine import GameAction
        level = getattr(latest_frame, "levels_completed", 0)
        if self.level != level:
            self._fresh()
            self.level = level
        raw = np.asarray(latest_frame.frame[-1], dtype=np.uint8)
        if self.tiles is None:
            self.tiles = TileGrid(raw)
        grid = self.tiles.read(raw)
        if self.wall is None:
            self.wall = int(grid[0, 0])
            self.objects = singletons(grid)
        self._observe(grid, latest_frame)
        available = [int(getattr(a, "value", a)) for a in latest_frame.available_actions]
        click = None
        if state_name(latest_frame) == "GAME_OVER":
            aid = 0
            self.plan.clear()
        elif 6 in available and not any(a in available for a in (1, 2, 3, 4)):
            if not self.clicks.targets:
                self.clicks.discover(grid)
            unknown = [p for p in self.clicks.targets if p not in self.clicks.columns]
            if unknown:
                click = unknown[0]
            else:
                if not self.plan:
                    self.plan = self.clicks.plan(grid) or []
                click = self.plan.pop(0) if self.plan else self.rng.choice(self.clicks.targets)
            aid = 6
        else:
            aid = self._keyboard(grid, available)
        self.previous, self.last = grid, (aid, click)
        self.actions += 1
        act = GameAction.from_id(aid)
        if click is not None:
            act.set_data(self.tiles.pixel(click))
        return act
