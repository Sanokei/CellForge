import numpy as np
import pytest

from cellforge.ir.graph import Graph, GraphError
from cellforge.reference.arc_grid_operator import score_tasks
from cellforge.runtime.grid import GridPredictor, pack_grid


def recolor_graph():
    # An actual authored cell maps each color c to (c+1)%10. Dimensions
    # travel through the graph; the adapter never copies the input as output.
    weight = np.zeros((10, 10))
    for color in range(10):
        weight[(color + 1) % 10, color] = 1
    return Graph(
        defs={
            "color": {"kind": "cell", "params": {"W1": "W", "b1": "b"}},
            "root": {"kind": "bundle", "inputs": ["input", "input_height", "input_width", "state"],
                     "nodes": [
                         {"id": "c", "use": "color", "in": {"x": "$input"}},
                         {"id": "cs", "use": "op:reshape", "in": {"x": "c"},
                          "args": {"shape": [1, 30, 30, 10]}},
                         {"id": "h", "use": "op:reshape", "in": {"x": "$input_height"},
                          "args": {"shape": [1, 30]}},
                         {"id": "w", "use": "op:reshape", "in": {"x": "$input_width"},
                          "args": {"shape": [1, 30]}}],
                     "outputs": {"height_logits": "h", "width_logits": "w",
                                 "color_logits": "cs", "state": "$state"}}},
        params={"W": weight, "b": np.zeros(10)}, entries={"grid_step": "root"},
        meta={"task": "grid", "grid_interface": "grid-graph/1", "state_dim": 0})


def test_compiled_grid_matches_reference_and_resets_state():
    graph = recolor_graph()
    reference = GridPredictor(graph)
    compiled = GridPredictor(graph, backend="compiled")
    demos = [{"input": [[1, 2]], "output": [[2, 3]]}]
    for query in [[[1, 2]], [[5], [7]], [[1, 2]]]:
        assert compiled(demos, query) == reference(demos, query)
        assert compiled.last_exec == reference.last_exec
    graph.params['b'][0] += 1
    with pytest.raises(GraphError, match="changed"):
        compiled(demos, [[1]])


def test_authored_grid_artifact_roundtrip_and_scoring(tmp_path):
    graph = recolor_graph()
    graph.save(tmp_path)
    predictor = GridPredictor(Graph.load(tmp_path))
    demo = [{"input": [[0]], "output": [[1]]}]
    assert predictor(demo, [[9, 1], [2, 0]]) == [[[0, 2], [3, 1]]]
    result = score_tasks([("synthetic", {"train": demo, "test": [
        {"input": [[9, 1]], "output": [[0, 2]]}]})], predictor)
    assert result["tasks_solved"] == 1
    assert predictor.last_exec["macs"] > 0


def test_padding_distinguishes_black_from_missing():
    values, mask, height, width = pack_grid([[0, 9]])
    assert values[0, 0, 0, 0] == 1 and mask[0, 0, 0, 0] == 1
    assert values[0, 1].sum() == 0 and mask[0, 1].sum() == 0
    assert height.argmax() == 0 and width.argmax() == 1


@pytest.mark.parametrize("grid", [[], [[True]], [[10]], [[1], [2, 3]]])
def test_invalid_inputs_rejected(grid):
    with pytest.raises(GraphError):
        pack_grid(grid)


def test_malformed_output_is_not_repaired():
    graph = recolor_graph()
    graph.meta["grid_attempts"] = 2
    with pytest.raises(GraphError, match="height_logits"):
        GridPredictor(graph)([{"input": [[0]], "output": [[1]]}], [[3]])


def test_incompatible_text_graph_rejected():
    graph = recolor_graph()
    graph.meta.pop("grid_interface")
    with pytest.raises(GraphError, match="declare"):
        GridPredictor(graph)


def test_query_target_absent_and_state_resets(monkeypatch):
    from cellforge.runtime import grid as runtime
    original = runtime.Interpreter
    received = []

    class ObservedInterpreter(original):
        def run_bundle(self, name, inputs, path=""):
            if name == "root":
                received.append({k: v.copy() for k, v in inputs.items()})
            return super().run_bundle(name, inputs, path)

    monkeypatch.setattr(runtime, "Interpreter", ObservedInterpreter)
    graph = recolor_graph()
    graph.meta["state_dim"] = 1
    graph.defs["root"]["nodes"].append({
        "id": "s", "use": "op:bias", "in": {"x": "$state"}, "args": {"c": 1}})
    graph.defs["root"]["outputs"]["state"] = "s"
    predictor = GridPredictor(graph)
    demo = [{"input": [[0]], "output": [[1]]}]
    for _ in range(2):
        predictor(demo, [[2]])
    assert [r["state"].item() for r in received] == [0, 1, 0, 1]
    for record in (received[1], received[3]):
        assert record["phase"].item() == 1
        for key in ("output", "output_mask", "output_height", "output_width"):
            assert not record[key].any()
