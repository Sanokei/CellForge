"""Author a demonstration-conditioned grid MoE using existing graph primitives.

Twenty experts: identity/transpose crossed with ten cyclic color permutations.
The router accumulates negative squared demonstration error in recurrent state.
All inference runs in the graph interpreter; there is no Python task solver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile


def build_grid_moe():
    g = Graph(meta={"task": "grid", "grid_interface": "grid-graph/1",
                    "state_dim": 20, "grid_attempts": 1, "profile": "ladder-micro/1",
                    "construction": "direct authored; no fitting; grid20/1"})
    # Ten outputs, each through a <=40-slot cell. Shared across experts.
    for shift in range(10):
        nodes = []
        for color in range(10):
            name = f"color_{shift}_{color}"
            w = np.zeros((1, 10))
            w[0, (color - shift) % 10] = 1
            g.params[name + ".W"] = w
            g.params[name + ".b"] = np.zeros(1)
            g.defs[name] = {"kind": "cell", "params": {"W1": name + ".W", "b1": name + ".b"}}
            nodes.append({"id": f"c{color}", "use": name, "in": {"x": "$x"}})
        nodes.append({"id": "join", "use": "op:concat", "args": {"axis": 2},
                      "in": {f"x{i}": f"c{i}" for i in range(10)}})
        g.defs[f"colors_{shift}"] = {"kind": "bundle", "role": "cluster", "inputs": ["x"],
                                       "nodes": nodes, "outputs": {"y": "join"}}

    router_nodes = []
    branches = []
    for transpose in (False, True):
        for shift in range(10):
            i = int(transpose) * 10 + shift
            name = f"transform_{i}"
            # Cluster restrictions forbid cluster nesting: inline instances of
            # the shared scalar cells into this cluster.
            nodes = []
            src = "$input"
            if transpose:
                nodes.append({"id": "t", "use": "op:transpose", "in": {"x": src},
                              "args": {"perm": [1, 0, 2]}})
                src = "t"
            for color in range(10):
                nodes.append({"id": f"c{color}", "use": f"color_{shift}_{color}", "in": {"x": src}})
            nodes.append({"id": "join", "use": "op:concat", "args": {"axis": 2},
                          "in": {f"x{j}": f"c{j}" for j in range(10)}})
            h = "$input_width" if transpose else "$input_height"
            w = "$input_height" if transpose else "$input_width"
            g.defs[name] = {"kind": "bundle", "role": "cluster",
                           "inputs": ["input", "input_height", "input_width"], "nodes": nodes,
                           "outputs": {"colors": "join", "height": h, "width": w}}
            ex = f"expert_{i}"
            branches.append(ex)
            g.defs[ex] = {"kind": "bundle", "role": "expert",
                         "inputs": ["input", "input_height", "input_width"],
                         "nodes": [
                             {"id": "transform", "use": name, "in": {
                                 "input": "$input", "input_height": "$input_height", "input_width": "$input_width"}},
                             {"id": "c", "use": "op:reshape", "in": {"x": "transform.colors"},
                              "args": {"shape": [1, 30, 30, 10]}},
                             {"id": "h", "use": "op:reshape", "in": {"x": "transform.height"},
                              "args": {"shape": [1, 30]}},
                             {"id": "w", "use": "op:reshape", "in": {"x": "transform.width"},
                              "args": {"shape": [1, 30]}}],
                         "outputs": {"color_logits": "c", "height_logits": "h", "width_logits": "w"}}
            # The router's cluster computes its own candidate/error; no experts
            # are nested in the router. Cells are shared by parameter identity.
            error_nodes = list(nodes)
            for port, ref, target, size in (
                ("c", "join", "$output", 9000), ("h", h, "$output_height", 30),
                ("w", w, "$output_width", 30)):
                error_nodes.extend([
                    {"id": f"{port}diff", "use": "op:sub", "in": {"a": ref, "b": target}},
                    {"id": f"{port}sq", "use": "op:square", "in": {"x": f"{port}diff"}},
                    {"id": f"{port}flat", "use": "op:reshape", "in": {"x": f"{port}sq"}, "args": {"shape": [size]}},
                    {"id": f"{port}sum", "use": "op:sum", "in": {"x": f"{port}flat"},
                     "args": {"axis": 0, "keepdims": True}}])
            error_nodes.extend([
                {"id": "hw", "use": "op:add", "in": {"a": "hsum", "b": "wsum"}},
                {"id": "total", "use": "op:add", "in": {"a": "csum", "b": "hw"}},
                {"id": "score", "use": "op:neg", "in": {"x": "total"}}])
            error = f"error_{i}"
            g.defs[error] = {"kind": "bundle", "role": "cluster",
                             "inputs": ["input", "input_height", "input_width", "output", "output_height", "output_width"],
                             "nodes": error_nodes, "outputs": {"score": "score"}}
            router_nodes.append({"id": f"e{i}", "use": error,
                                 "in": {p: "$" + p for p in g.defs[error]["inputs"]}})
    router_nodes.extend([
        {"id": "join", "use": "op:concat", "args": {"axis": 0}, "in": {f"x{i}": f"e{i}" for i in range(20)}},
        {"id": "negphase", "use": "op:neg", "in": {"x": "$phase"}},
        {"id": "gate", "use": "op:bias", "in": {"x": "negphase"}, "args": {"c": 1}},
        {"id": "update", "use": "op:mul", "in": {"a": "join", "b": "gate"}},
        {"id": "scores", "use": "op:add", "in": {"a": "$state", "b": "update"}}])
    router_inputs = ["input", "input_height", "input_width", "output", "output_height", "output_width", "phase", "state"]
    g.defs["router"] = {"kind": "bundle", "role": "router", "inputs": router_inputs,
                        "nodes": router_nodes, "outputs": {"scores": "scores"}}
    g.defs["root"] = {"kind": "bundle", "inputs": router_inputs,
                      "nodes": [
                          {"id": "router", "use": "router", "in": {p: "$" + p for p in router_inputs}},
                          {"id": "dispatch", "use": "dispatch", "args": {"branches": branches},
                           "in": {"scores": "router.scores", "input": "$input",
                                  "input_height": "$input_height", "input_width": "$input_width"}}],
                      "outputs": {"state": "router.scores", "height_logits": "dispatch.height_logits",
                                  "width_logits": "dispatch.width_logits", "color_logits": "dispatch.color_logits"}}
    g.entries = {"grid_step": "root"}
    # Discard the unused construction templates; all remaining definitions are
    # either reachable or shared cells invoked by reachable clusters.
    for shift in range(10):
        del g.defs[f"colors_{shift}"]
    g.validate()
    check_profile(g, "ladder-micro/1")
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixed-router", action="store_true")
    args = parser.parse_args()
    g = build_grid_moe()
    if args.fixed_router:
        g.defs["router"]["nodes"] = [{"id": "scores", "use": "op:scale",
                                        "in": {"x": "$state"}, "args": {"c": 0}}]
        g.meta["construction"] = "direct authored; no fitting; grid20-fixed-router/1"
        g.validate()
        check_profile(g, "ladder-micro/1")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output must be new; preserve prior artifacts")
    digest = g.save(out)
    card = {"model_hash": digest, "stats": g.stats(), "profile": "ladder-micro/1",
            "interface": "grid-graph/1", "training": "none",
            "fixed_router_control": args.fixed_router,
            "experts": ("20 transforms, identity branch always selected" if args.fixed_router else
                        "20 fixed transforms selected by demonstration reconstruction error"),
            "limitations": "no object reasoning; no general color mapping; no task IDs or answer lookup"}
    (out / "MODEL_CARD.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
