"""Author eight spatial experts with demonstration-derived color associations.

The serialized graph performs every transform, association update and routing
decision. Authoring creates weights/wiring; it never reads benchmark tasks.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile


class Nodes:
    def __init__(self):
        self.nodes = []

    def add(self, use, inputs=None, **args):
        name = f"n{len(self.nodes)}"
        node = {"id": name, "use": use, "in": inputs or {}}
        if args:
            node["args"] = args
        self.nodes.append(node)
        return name

    def op(self, op, **inputs):
        return self.add("op:" + op, inputs)

    def reshape(self, ref, shape):
        return self.add("op:reshape", {"x": ref}, shape=shape)

    def slice(self, ref, start, stop):
        return self.add("op:slice", {"x": ref}, axis=0, start=start, stop=stop)

    def sum(self, ref):
        return self.add("op:sum", {"x": ref}, axis=0, keepdims=True)

    def bundle(self, role, inputs, outputs):
        return {"kind": "bundle", "role": role, "inputs": inputs,
                "nodes": self.nodes, "outputs": outputs}


def build_associative_grid():
    g = Graph(meta={"task": "grid", "grid_interface": "grid-graph/1", "grid_attempts": 1,
                    "state_dim": 808, "profile": "ladder-micro/1",
                    "construction": "direct authored; no fitting; grid-associative/1",
                    "donor_hash": "5a5e8e210bc56159bf9b70212c9aced604ef6cae6494564c577c5075436cbf2e"})
    # A <=40-slot cell decodes dimension one-hots; no dynamic transform
    # primitives or Python rule application are added to the interpreter.
    g.params["dimension.W"] = np.arange(1, 31, dtype=float)[None]
    g.params["dimension.b"] = np.zeros(1)
    g.defs["dimension"] = {"kind": "cell", "params": {"W1": "dimension.W", "b1": "dimension.b"}}
    for i in range(10):
        g.params[f"fallback_{i}"] = np.eye(10)[i:i + 1] * 0.001

    for i in range(8):
        n = Nodes()
        x, h, w = "$input", "$input_height", "$input_width"
        # Bits 0 and 1 flip the input vertically/horizontally; bit 2 transposes.
        for bit, dim in ((1, h), (2, w)):
            if not i & bit:
                continue
            if bit == 2:
                x = n.add("op:transpose", {"x": x}, perm=[1, 0, 2])
            size = n.add("dimension", {"x": dim})
            last = n.add("op:bias", {"x": size}, c=-1)
            axis = n.add("op:coords", n=30)
            row = n.reshape(axis, [30, 1])
            col = n.reshape(axis, [1, 30])
            sums = n.op("add", a=row, b=col)
            dist = n.op("sub", a=sums, b=n.reshape(last, [1, 1]))
            tri = n.add("op:bias", {"x": n.op("neg", x=n.op("abs", x=dist))}, c=1)
            reverse = n.op("relu", x=tri)
            x = n.reshape(n.op("matmul", a=reverse, b=n.reshape(x, [30, 300])), [30, 30, 10])
            if bit == 2:
                x = n.add("op:transpose", {"x": x}, perm=[1, 0, 2])
        if i & 4:
            x = n.add("op:transpose", {"x": x}, perm=[1, 0, 2])
            h, w = w, h
        g.defs[f"geometry_{i}"] = n.bundle("cluster", ["input", "input_height", "input_width"],
                                             {"colors": x, "height": h, "width": w})

    n = Nodes()
    gate = n.add("op:bias", {"x": n.op("neg", x="$phase")}, c=1)
    x = n.add("op:transpose", {"x": n.reshape("$colors", [900, 10])}, perm=[1, 0])
    correlation = n.op("matmul", a=x, b=n.reshape("$output", [900, 10]))
    delta = n.op("mul", a=n.reshape(correlation, [100]), b=gate)
    counts = n.op("add", a=n.slice("$state", 0, 100), b=delta)
    errors = []
    for predicted, target in (("$height", "$output_height"), ("$width", "$output_width")):
        errors.append(n.sum(n.op("square", x=n.op("sub", a=predicted, b=target))))
    delta_error = n.op("mul", a=n.op("add", a=errors[0], b=errors[1]), b=gate)
    error = n.op("add", a=n.slice("$state", 100, 101), b=delta_error)
    state = n.add("op:concat", {"x0": counts, "x1": error}, axis=0)
    g.defs["association_update"] = n.bundle("cluster", ["colors", "height", "width", "output",
        "output_height", "output_width", "phase", "state"], {"state": state})

    n = Nodes()
    scores = []
    for i in range(8):
        state = n.slice("$state", i * 101, (i + 1) * 101)
        counts = n.reshape(n.slice(state, 0, 100), [10, 10])
        total = n.sum(n.add("op:sum", {"x": counts}, axis=1))
        majority = n.sum(n.add("op:max", {"x": counts}, axis=1))
        conflicts = n.op("sub", a=total, b=majority)
        shape_error = n.add("op:scale", {"x": n.slice(state, 100, 101)}, c=9001)
        scores.append(n.op("neg", x=n.op("add", a=conflicts, b=shape_error)))
    joined = n.add("op:concat", {f"x{i}": ref for i, ref in enumerate(scores)}, axis=0)
    g.defs["router"] = n.bundle("router", ["state"], {"scores": joined})

    for i in range(8):
        n = Nodes()
        counts = n.reshape(n.slice("$state", i * 101, i * 101 + 100), [10, 10])
        fallback_rows = [n.add(f"param:fallback_{j}") for j in range(10)]
        fallback = n.add("op:concat", {f"x{j}": ref for j, ref in enumerate(fallback_rows)}, axis=0)
        mapping = n.add("op:onehot_argmax", {"x": n.op("add", a=counts, b=fallback)}, axis=1)
        colors = n.op("matmul", a="$colors", b=mapping)
        g.defs[f"mapping_{i}"] = n.bundle("cluster", ["state", "colors"], {"colors": colors})
        n = Nodes()
        geometry = n.add(f"geometry_{i}", {"input": "$input", "input_height": "$input_height", "input_width": "$input_width"})
        mapped = n.add(f"mapping_{i}", {"state": "$state", "colors": geometry + ".colors"})
        g.defs[f"expert_{i}"] = n.bundle("expert", ["input", "input_height", "input_width", "state"], {
            "color_logits": n.reshape(mapped, [1, 30, 30, 10]),
            "height_logits": n.reshape(geometry + ".height", [1, 30]),
            "width_logits": n.reshape(geometry + ".width", [1, 30])})

    n = Nodes()
    updates = []
    for i in range(8):
        geometry = n.add(f"geometry_{i}", {"input": "$input", "input_height": "$input_height", "input_width": "$input_width"})
        updates.append(n.add("association_update", {
            "colors": geometry + ".colors", "height": geometry + ".height", "width": geometry + ".width",
            "output": "$output", "output_height": "$output_height", "output_width": "$output_width",
            "phase": "$phase", "state": n.slice("$state", i * 101, (i + 1) * 101)}))
    state = n.add("op:concat", {f"x{i}": ref for i, ref in enumerate(updates)}, axis=0)
    router = n.add("router", {"state": state})
    dispatch = n.add("dispatch", {"scores": router + ".scores", "state": state,
                    "input": "$input", "input_height": "$input_height", "input_width": "$input_width"},
                     branches=[f"expert_{i}" for i in range(8)])
    g.defs["root"] = n.bundle("root", ["input", "input_height", "input_width", "output",
        "output_height", "output_width", "phase", "state"], {"state": state,
        "color_logits": dispatch + ".color_logits", "height_logits": dispatch + ".height_logits",
        "width_logits": dispatch + ".width_logits"})
    g.entries = {"grid_step": "root"}
    g.validate()
    check_profile(g, "ladder-micro/1")
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--identity-colors", action="store_true",
                        help="Control: preserve geometry/routing, disable inferred color mapping")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("output must be new")
    g = build_associative_grid()
    if args.identity_colors:
        for i in range(8):
            g.defs[f"mapping_{i}"]["nodes"] = []
            g.defs[f"mapping_{i}"]["outputs"] = {"colors": "$colors"}
        g.meta["construction"] = "direct authored; no fitting; grid-associative-identity-colors/1"
        g.validate()
        check_profile(g, "ladder-micro/1")
    digest = g.save(output)
    card = {"model_hash": digest, "stats": g.stats(), "profile": "ladder-micro/1",
            "interface": "grid-graph/1", "training": "none",
            "identity_colors_control": args.identity_colors,
            "inference": ("eight dihedral geometries; associative routing; output colors unchanged"
                          if args.identity_colors else
                          "demonstration-derived color associations in recurrent state; eight dihedral geometries"),
            "limitations": "global transforms/color functions only; no objects, cropping or contextual recoloring",
            "donor_hash": g.meta["donor_hash"]}
    (output / "MODEL_CARD.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
