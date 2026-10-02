"""Tests use invented observations, never public-game implementation code."""
from types import SimpleNamespace
from unittest.mock import patch
import enum
import sys

import numpy as np

from research.claude_arc3.genes import GeneAgent
from research.claude_arc3.genes.learning import ActionEffects, TerrainLearning, KeyDoorLearning, HazardMotion
from research.claude_arc3.genes.perception import TileGrid, AvatarDetection
from research.claude_arc3.genes.planning import ClickEffects, navigation, push_plan


class Action(enum.Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION6 = 6

    @classmethod
    def from_id(cls, a):
        return cls(a)

    def set_data(self, data):
        self.action_data = data


def obs(grid, state="NOT_FINISHED", level=0, actions=(1, 2, 3, 4)):
    return SimpleNamespace(frame=[np.kron(grid, np.ones((4, 4), dtype=np.uint8))],
                           state=SimpleNamespace(name=state), levels_completed=level,
                           available_actions=list(actions))


def test_tile_inference_and_coordinate_order():
    grid = np.array([[2, 2, 2], [2, 7, 0], [2, 0, 2]], dtype=np.uint8)
    tiles = TileGrid(obs(grid).frame[-1])
    assert tiles.scale == 4
    np.testing.assert_array_equal(tiles.read(obs(grid).frame[-1]), grid)
    assert tiles.pixel((2, 1)) == {"x": 10, "y": 6}


def test_avatar_motion_and_permuted_action_effect():
    a = np.zeros((5, 5), dtype=np.uint8)
    a[2, 2] = 8
    b = a.copy(); b[2, 2] = 0; b[1, 2] = 8
    gene = AvatarDetection()
    gene.observe(a, b, 4)
    assert gene.color == 8 and gene.locate(b) == (2, 1)
    effects = ActionEffects()
    effects.observe(4, (2, 2), (2, 1))
    effects.observe(1, (2, 1), (2, 1))
    assert effects.mapping() == {4: (0, -1)}


def test_death_does_not_label_all_floor_deadly():
    terrain = TerrainLearning()
    grid = np.zeros((5, 5), dtype=np.uint8)
    terrain.observe(grid, (2, 2), (3, 2), 2, {2: (1, 0)}, True)
    assert (3, 2) in terrain.dead and not terrain.dead_colors


def test_key_removes_remote_door_and_unblocks_edges():
    a = np.zeros((5, 5), dtype=np.uint8)
    a[1, 1], a[1, 2], a[3, 3] = 8, 9, 10
    b = a.copy(); b[1, 1], b[1, 2], b[3, 3] = 0, 8, 0
    terrain = TerrainLearning(); terrain.blocked.add(((2, 3), (3, 3)))
    key = KeyDoorLearning()
    key.observe(a, b, (1, 1), (2, 1), 8, terrain)
    assert key.pairs == {(9, 10)} and not terrain.blocked


def test_navigation_respects_directed_blocked_edge():
    allowed = {(x, y) for y in range(3) for x in range(3)}
    mapping = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}
    path = navigation((0, 0), (2, 0), mapping, allowed, {((0, 0), (1, 0))})
    assert path == [2, 1, 1, 4]


def test_push_planner_repositions_avatar_and_avoids_corner():
    mapping = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}
    allowed = {(x, y) for y in range(5) for x in range(5)}
    path = push_plan((2, 3), (2, 2), (3, 2), mapping, allowed)
    assert path == [3, 4, 1]
    assert push_plan((1, 0), (0, 0), (3, 3), mapping, allowed) is None


def test_learned_click_columns_solve_binary_goal():
    a = np.zeros((4, 5), dtype=np.uint8); a[2, 1] = 3; a[2, 3] = 4
    clicks = ClickEffects(); clicks.discover(a)
    b = a.copy(); b[2, 1] = 4
    clicks.observe(a, b, (1, 2))
    c = b.copy(); c[2, 3] = 3
    clicks.observe(b, c, (3, 2))
    plan = clicks.plan(a)
    assert plan in ([(1, 2)], [(3, 2)])


def test_motion_cycle_prediction():
    motion = HazardMotion()
    motion.tracks[4] = [(1, 2), (2, 2), (3, 2), (2, 2)]
    assert motion.predicted() == {(1, 2)}


def test_contract_budget_death_and_level_lifecycle():
    grid = np.zeros((5, 5), dtype=np.uint8); grid[2, 2] = 8
    agent = GeneAgent(budget=2)
    with patch.dict(sys.modules, {"arcengine": SimpleNamespace(GameAction=Action)}):
        action = agent.choose_action([], obs(grid))
        assert action.value in (1, 2, 3, 4)
        assert agent.choose_action([], obs(grid, "GAME_OVER")) == Action.RESET
        assert agent.is_done([], obs(grid))
        assert GeneAgent().is_done([], obs(grid, "WIN"))
        agent = GeneAgent()
        agent.choose_action([], obs(grid))
        agent.terrain.dead.add((1, 1))
        agent.choose_action([], obs(grid, level=1))
        assert not agent.terrain.dead and agent.level == 1
