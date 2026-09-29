"""Reference CPU interpreter for recursive cell/bundle graphs.

Transparent and unoptimized on purpose: it is the semantics that any
accelerated backend must match under differential testing.
"""
from __future__ import annotations

import numpy as np

from ..ir.graph import Graph, GraphError, parse_ref
from .ops import ACTIVATIONS, OPS


class ExecStats:
    def __init__(self):
        self.node_execs = 0
        self.macs = 0
        self.route_counts: dict[str, dict[str, int]] = {}
        self.peak_elements = 0

    def as_dict(self):
        return {"node_execs": self.node_execs, "macs": int(self.macs),
                "route_counts": self.route_counts, "peak_tensor_elements": int(self.peak_elements)}


class Interpreter:
    def __init__(self, graph: Graph, trace: bool = False, trace_filter=None):
        self.g = graph
        self.trace_enabled = trace
        self.trace_filter = trace_filter
        self.trace: dict[str, np.ndarray] = {}
        self.stats = ExecStats()

    # -------------------------------------------------------------- cells
    def run_cell(self, name: str, x: np.ndarray) -> np.ndarray:
        d = self.g.defs[name]
        p = {k: self.g.params[v] for k, v in d["params"].items()}
        W1, b1 = p["W1"], p["b1"]
        if x.shape[-1] != W1.shape[1]:
            raise GraphError(f"cell {name}: input width {x.shape[-1]} != {W1.shape[1]}")
        lead = int(np.prod(x.shape[:-1]))
        h = ACTIVATIONS[d.get("act", "identity")](x @ W1.T + b1)
        self.stats.macs += lead * W1.size
        if "W2" in p:
            h = h @ p["W2"].T + p["b2"]
            self.stats.macs += lead * p["W2"].size
        return h

    # ------------------------------------------------------------ bundles
    def run_bundle(self, name: str, inputs: dict, path: str = "") -> dict:
        d = self.g.defs[name]
        for k in d.get("inputs", []):
            if k not in inputs:
                raise GraphError(f"bundle {name}: missing input {k}")
        vals: dict[str, dict] = {}

        def resolve(ref):
            kind, tgt, port = parse_ref(ref)
            if kind == "input":
                return inputs[tgt]
            outs = vals[tgt]
            if port is None:
                if len(outs) != 1:
                    raise GraphError(f"bundle {name}: {tgt} has several outputs; name a port")
                return next(iter(outs.values()))
            if port not in outs:
                raise GraphError(f"bundle {name}: {tgt} has no output {port}")
            return outs[port]

        for n in d["nodes"]:
            nid, use = n["id"], n["use"]
            npath = f"{path}/{nid}"
            args = n.get("args", {})
            ins = {port: resolve(ref) for port, ref in n.get("in", {}).items()}
            self.stats.node_execs += 1
            if use.startswith("op:"):
                fn, _ = OPS[use[3:]]
                try:
                    out = {"out": fn(ins, args)}
                except (ValueError, IndexError, KeyError, TypeError) as e:
                    raise GraphError(f"{name}/{nid} ({use}): {e}") from e
                if use == "op:matmul":
                    a, b = ins["a"], ins["b"]
                    self.stats.macs += int(np.prod(out["out"].shape)) * a.shape[-1]
            elif use.startswith("param:"):
                out = {"out": self.g.params[use[6:]][None, ...]}
            elif use == "dispatch":
                out = self.run_dispatch(name, n, ins, npath)
            else:
                sub = self.g.defs[use]
                if sub["kind"] == "cell":
                    if "x" not in ins:
                        raise GraphError(f"{name}/{nid}: cell instance needs input x")
                    out = {"y": self.run_cell(use, ins["x"])}
                else:
                    out = self.run_bundle(use, ins, npath)
            for v in out.values():
                self.stats.peak_elements = max(self.stats.peak_elements, v.size)
            if self.trace_enabled and (self.trace_filter is None or self.trace_filter(npath)):
                for k, v in out.items():
                    self.trace[f"{npath}.{k}"] = v
            vals[nid] = out
        return {o: resolve(ref) for o, ref in d["outputs"].items()}

    def run_dispatch(self, parent, node, ins, path):
        """Top-1 lazy dispatch. Ties go to the lowest branch index.

        Rows with non-finite scores produce NaN outputs (recorded failures).
        Only branches selected by at least one row execute, on those rows only.
        """
        branches = node["args"]["branches"]
        scores = ins.pop("scores")
        B = scores.shape[0]
        scores = scores.reshape(B, -1)
        if scores.shape[1] != len(branches):
            raise GraphError(f"{parent}/{node['id']}: {scores.shape[1]} scores for {len(branches)} branches")
        finite = np.all(np.isfinite(scores), axis=1)
        choice = np.argmax(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
        counts = self.stats.route_counts.setdefault(path, {})
        results = {}
        for bi, bname in enumerate(branches):
            rows = np.nonzero((choice == bi) & finite)[0]
            counts[bname] = counts.get(bname, 0) + int(rows.size)
            if rows.size == 0:
                continue
            sub_in = {k: (v[rows] if v.shape[0] == B and B > 1 else v) for k, v in ins.items()}
            out = self.run_bundle(bname, sub_in, f"{path}[{bname}]")
            for k, v in out.items():
                if k not in results:
                    results[k] = np.full((B,) + v.shape[1:], np.nan)
                if v.shape[0] == 1 and rows.size > 1:
                    v = np.broadcast_to(v, (rows.size,) + v.shape[1:])
                results[k][rows] = v
        if not results:
            # every row invalid: run first branch for shapes, then poison
            out = self.run_bundle(branches[0], ins, f"{path}[{branches[0]}]")
            results = {k: np.full((B,) + v.shape[1:], np.nan) for k, v in out.items()}
        if self.trace_enabled:
            self.trace[f"{path}.route"] = choice.astype(np.float64)
        return results
