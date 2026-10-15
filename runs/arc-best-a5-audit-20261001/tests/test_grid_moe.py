import importlib.util
from pathlib import Path

import numpy as np

from cellforge.ir.profiles import check_profile
from cellforge.runtime.grid import GridPredictor

spec = importlib.util.spec_from_file_location("build_grid_moe", Path(__file__).parents[1] / "scripts/build_grid_moe.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_all_twenty_experts_generalize_from_demo_to_different_grid():
    graph = builder.build_grid_moe()
    assert len(check_profile(graph, "ladder-micro/1")["experts"]) == 20
    predictor = GridPredictor(graph)
    demo = np.array([[0, 1, 4], [7, 3, 9]])
    query = np.array([[2, 4], [9, 0], [5, 3], [8, 1]])
    for transpose in (False, True):
        for shift in range(10):
            expected_demo = ((demo.T if transpose else demo) + shift) % 10
            expected_query = ((query.T if transpose else query) + shift) % 10
            assert predictor([{"input": demo.tolist(), "output": expected_demo.tolist()}], query.tolist()) == [expected_query.tolist()]


def test_multiple_demos_disambiguate_initial_tie():
    predictor = GridPredictor(builder.build_grid_moe())
    # Symmetric first demonstration cannot distinguish identity from transpose.
    demos = [{"input": [[1]], "output": [[1]]},
             {"input": [[1, 2, 3]], "output": [[1], [2], [3]]}]
    assert predictor(demos, [[8, 9]]) == [[[8], [9]]]


def test_query_does_not_change_accumulated_expert_scores():
    predictor = GridPredictor(builder.build_grid_moe())
    demos = [{"input": [[1, 2]], "output": [[3, 4]]}]
    predictor(demos, [[8]])
    assert predictor.last_exec["route_counts"]["/grid/dispatch"]["expert_2"] == 2
