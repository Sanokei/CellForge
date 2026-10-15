"""Grid inference through an authored graph, without an inference-time author.

Demonstrations update graph-owned recurrent state. A final query has no output
grid supplied. Dimensions and colors are decoded solely from graph logits.
"""
from __future__ import annotations

import numpy as np

from ..ir.graph import Graph, GraphError
from .interp import Interpreter


def pack_grid(grid):
    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
        raise GraphError("grid height must be 1..30")
    width = len(grid[0]) if isinstance(grid[0], list) else 0
    if not 1 <= width <= 30:
        raise GraphError("grid width must be 1..30")
    values = np.zeros((1, 30, 30, 10))
    mask = np.zeros((1, 30, 30, 1))
    for y, row in enumerate(grid):
        if not isinstance(row, list) or len(row) != width:
            raise GraphError("grid must be rectangular")
        for x, color in enumerate(row):
            if type(color) is not int or not 0 <= color <= 9:
                raise GraphError("grid colors must be integers 0..9")
            values[0, y, x, color] = 1
            mask[0, y, x, 0] = 1
    height = np.eye(30)[[len(grid) - 1]]
    widths = np.eye(30)[[width - 1]]
    return values, mask, height, widths


class GridPredictor:
    """``grid-graph/1`` adapter for the exact-grid scorer.

    The grid_step entry consumes input/output one-hots, masks, dimension
    one-hots, phase (0=demo, 1=query), index, and state. It returns state plus
    query height_logits [1,A,30], width_logits [1,A,30], and color_logits
    [1,A,30,30,10], where A is the declared one or two attempts. No labels or
    task identifiers enter the graph on a query. State resets for each call.
    """

    def __init__(self, graph: Graph):
        if graph.meta.get("grid_interface") != "grid-graph/1":
            raise GraphError("artifact must declare grid-graph/1")
        graph.validate()
        if "grid_step" not in graph.entries:
            raise GraphError("artifact needs a grid_step entry")
        self.graph = graph
        self.state_dim = graph.meta.get("state_dim", 0)
        self.attempts = graph.meta.get("grid_attempts", 1)
        if type(self.state_dim) is not int or not 0 <= self.state_dim <= 100000:
            raise GraphError("invalid grid state_dim")
        if type(self.attempts) is not int or self.attempts not in (1, 2):
            raise GraphError("grid_attempts must be 1 or 2")
        self.last_exec = None

    def __call__(self, demonstrations, test_input):
        state = np.zeros((1, self.state_dim))
        interpreter = Interpreter(self.graph)
        if not demonstrations:
            raise GraphError("at least one demonstration is required")
        inputs = [dict(demo) for demo in demonstrations] + [{"input": test_input}]
        for index, item in enumerate(inputs):
            query = index == len(demonstrations)
            x, xm, xh, xw = pack_grid(item["input"])
            if query:
                y, ym, yh, yw = (np.zeros_like(v) for v in (x, xm, xh, xw))
            else:
                y, ym, yh, yw = pack_grid(item["output"])
            out = interpreter.run_bundle(self.graph.entries["grid_step"], {
                "input": x, "input_mask": xm, "input_height": xh, "input_width": xw,
                "output": y, "output_mask": ym, "output_height": yh, "output_width": yw,
                "phase": np.array([[float(query)]]), "index": np.array([[float(index)]]),
                "state": state,
            }, "/grid")
            state = np.asarray(out["state"])
            if state.shape != (1, self.state_dim) or not np.isfinite(state).all():
                raise GraphError("invalid grid recurrent state")
        logits = {}
        for name, shape in {
            "height_logits": (1, self.attempts, 30),
            "width_logits": (1, self.attempts, 30),
            "color_logits": (1, self.attempts, 30, 30, 10),
        }.items():
            logits[name] = np.asarray(out[name])
            if logits[name].shape != shape or not np.isfinite(logits[name]).all():
                raise GraphError(f"invalid {name}; expected {shape}")
        self.last_exec = interpreter.stats.as_dict()
        grids = []
        for attempt in range(self.attempts):
            height = int(logits["height_logits"][0, attempt].argmax()) + 1
            width = int(logits["width_logits"][0, attempt].argmax()) + 1
            grids.append(logits["color_logits"][0, attempt, :height, :width].argmax(-1).tolist())
        return grids
