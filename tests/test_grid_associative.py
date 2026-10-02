import importlib.util
from pathlib import Path

import numpy as np

from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.runtime.grid import GridPredictor

spec = importlib.util.spec_from_file_location("build_grid_associative", Path(__file__).parents[1] / "scripts/build_grid_associative.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def transform(grid, geometry, mapping):
    if geometry & 1:
        grid = grid[::-1]
    if geometry & 2:
        grid = grid[:, ::-1]
    if geometry & 4:
        grid = grid.T
    return mapping[grid].tolist()


def test_eight_geometries_with_unseen_arbitrary_color_functions(tmp_path):
    graph = builder.build_associative_grid()
    check_profile(graph, "ladder-micro/1")
    graph.save(tmp_path)
    predictor = GridPredictor(Graph.load(tmp_path))
    rng = np.random.default_rng(812)
    first = rng.permutation(10).reshape(2, 5)
    second = rng.integers(0, 10, (5, 3))
    query = rng.integers(0, 10, (4, 7))
    for geometry in range(8):
        # Include many-to-one recoloring, not just permutations.
        for mapping in (rng.permutation(10), rng.integers(0, 10, 10)):
            demos = [{"input": x.tolist(), "output": transform(x, geometry, mapping)}
                     for x in (first, second)]
            assert predictor(demos, query.tolist()) == [transform(query, geometry, mapping)]


def test_unobserved_color_defaults_to_identity_and_tasks_reset():
    predictor = GridPredictor(builder.build_associative_grid())
    assert predictor([{"input": [[1]], "output": [[7]]}], [[1, 5]]) == [[[7, 5]]]
    assert predictor([{"input": [[1]], "output": [[3]]}], [[1, 5]]) == [[[3, 5]]]


def test_edges_at_maximum_dimensions():
    predictor = GridPredictor(builder.build_associative_grid())
    rng = np.random.default_rng(945)
    demo = rng.integers(0, 10, (30, 30))
    query = rng.integers(0, 10, (30, 1))
    mapping = rng.permutation(10)
    geometry = 7
    assert predictor([{"input": demo.tolist(), "output": transform(demo, geometry, mapping)}],
                     query.tolist()) == [transform(query, geometry, mapping)]
