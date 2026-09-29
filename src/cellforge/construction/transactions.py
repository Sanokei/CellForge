"""Atomic, validated model transactions authored by LLM agents.

A proposal is JSON::

    {"base": "<graph hash>", "author": "...", "task_id": "...",
     "rationale": "short summary", "ops": [ {...}, ... ]}

Op types
  set_param   {name, value}           explicit nested-list tensor
              {name, pattern}         deterministic expansion of an
                                      author-specified pattern (see expand_pattern)
  delete_param {name}
  put_def     {name, def}             create or replace a cell/bundle definition
  delete_def  {name}
  set_entry   {entry, def}
  set_state_dim {value}
  wrap        {parent, nodes, new_def, new_node}   pure refactor
  unwrap      {parent, node}                        pure refactor (inline)

Deterministic tools only materialize choices written in the proposal.  None of
them fits values to data.
"""
from __future__ import annotations

import copy

import numpy as np

from ..ir.graph import Graph, GraphError, Limits, parse_ref
from ..runtime.decoder import decode

MAX_PATTERN_ELEMENTS = 1_000_000


class TransactionError(ValueError):
    pass


def expand_pattern(pat: dict, graph: Graph) -> np.ndarray:
    """Expand an author-written pattern into an array.

    Supported forms (all fully determined by the author's text):
      {"fill": v, "shape": [...]}
      {"ascii": ["#..#", ...], "map": {"#": 1.0, ".": 0.0}}  -> 2-D array
      {"sparse": [[i, j, ..., v], ...], "shape": [...], "default": 0}
      {"eye": n, "scale": s}
      {"copy": "<param name>"}            exact copy (unshared)
      {"stack": [<pattern or list>, ...]} stack along a new axis 0
      {"flatten": <pattern>}              row-major flatten
      {"tile": <pattern>, "reps": [...]}
      {"scale": s, "of": <pattern>}       multiply an expansion by s
      {"add": [<pattern>, <pattern>]}     elementwise sum of two expansions
      {"arange": [start, stop, step]}
    """
    def sub(p):
        if isinstance(p, dict):
            return expand_pattern(p, graph)
        return np.asarray(p, dtype=np.float64)

    if "fill" in pat:
        a = np.full(tuple(pat["shape"]), float(pat["fill"]))
    elif "ascii" in pat:
        rows = pat["ascii"]
        m = {k: float(v) for k, v in pat.get("map", {"#": 1.0, ".": 0.0}).items()}
        width = max(len(r) for r in rows)
        a = np.zeros((len(rows), width))
        for i, r in enumerate(rows):
            for j, ch in enumerate(r):
                if ch not in m:
                    raise TransactionError(f"ascii pattern character {ch!r} not in map")
                a[i, j] = m[ch]
    elif "sparse" in pat:
        a = np.full(tuple(pat["shape"]), float(pat.get("default", 0.0)))
        for entry in pat["sparse"]:
            *idx, v = entry
            a[tuple(int(i) for i in idx)] = float(v)
    elif "eye" in pat:
        a = np.eye(int(pat["eye"])) * float(pat.get("scale", 1.0))
    elif "copy" in pat:
        if pat["copy"] not in graph.params:
            raise TransactionError(f"copy of unknown param {pat['copy']}")
        a = graph.params[pat["copy"]].copy()
    elif "stack" in pat:
        a = np.stack([sub(p) for p in pat["stack"]])
    elif "flatten" in pat:
        a = sub(pat["flatten"]).reshape(-1)
    elif "tile" in pat:
        a = np.tile(sub(pat["tile"]), tuple(pat["reps"]))
    elif "of" in pat and "scale" in pat:
        a = sub(pat["of"]) * float(pat["scale"])
    elif "add" in pat:
        x, y = (sub(p) for p in pat["add"])
        a = x + y
    elif "arange" in pat:
        a = np.arange(*[float(v) for v in pat["arange"]])
    else:
        raise TransactionError(f"unknown pattern keys {sorted(pat)}")
    if "reshape" in pat:
        a = a.reshape(tuple(pat["reshape"]))
    if a.size > MAX_PATTERN_ELEMENTS:
        raise TransactionError("pattern expands beyond element limit")
    return a


# ---------------------------------------------------------------- refactors
def _topo(nodes: list[dict]) -> list[dict]:
    ids = {n["id"] for n in nodes}
    deps = {n["id"]: {parse_ref(r)[1] for r in n.get("in", {}).values()
                      if parse_ref(r)[0] == "node" and parse_ref(r)[1] in ids} for n in nodes}
    order, placed = [], set()
    remaining = list(nodes)
    while remaining:
        for i, n in enumerate(remaining):
            if deps[n["id"]] <= placed:
                order.append(n)
                placed.add(n["id"])
                remaining.pop(i)
                break
        else:
            raise TransactionError("refactor would create a cycle")
    return order


def wrap(graph: Graph, parent: str, node_ids: list[str], new_def: str, new_node: str) -> None:
    P = graph.defs[parent]
    if P["kind"] != "bundle":
        raise TransactionError("wrap parent must be a bundle")
    if new_def in graph.defs:
        raise TransactionError(f"definition {new_def} already exists")
    S = set(node_ids)
    inner = [n for n in P["nodes"] if n["id"] in S]
    if len(inner) != len(S):
        raise TransactionError("wrap names unknown nodes")
    if any(n["id"] == new_node for n in P["nodes"] if n["id"] not in S):
        raise TransactionError("new node id collides")
    in_ports: dict[str, str] = {}  # external ref -> port name

    def ext(ref):
        if ref not in in_ports:
            in_ports[ref] = f"in{len(in_ports)}"
        return "$" + in_ports[ref]

    new_nodes = []
    for n in inner:
        n2 = copy.deepcopy(n)
        for port, ref in n.get("in", {}).items():
            k, tgt, _ = parse_ref(ref)
            if k == "input" or tgt not in S:
                n2["in"][port] = ext(ref)
        new_nodes.append(n2)
    out_ports: dict[str, str] = {}

    def out_for(ref):
        if ref not in out_ports:
            out_ports[ref] = f"out{len(out_ports)}"
        return out_ports[ref]

    outer = []
    for n in P["nodes"]:
        if n["id"] in S:
            continue
        n2 = copy.deepcopy(n)
        for port, ref in n.get("in", {}).items():
            k, tgt, _ = parse_ref(ref)
            if k == "node" and tgt in S:
                n2["in"][port] = f"{new_node}.{out_for(ref)}"
        outer.append(n2)
    outputs = dict(P["outputs"])
    for o, ref in P["outputs"].items():
        k, tgt, _ = parse_ref(ref)
        if k == "node" and tgt in S:
            outputs[o] = f"{new_node}.{out_for(ref)}"
    if not out_ports:
        raise TransactionError("wrapped nodes have no consumers; nothing to wrap")
    graph.defs[new_def] = {"kind": "bundle", "inputs": list(in_ports.values()), "nodes": new_nodes,
                           "outputs": {v: k for k, v in out_ports.items()}}
    wrapper = {"id": new_node, "use": new_def, "in": {v: k for k, v in in_ports.items()}}
    graph.defs[parent] = dict(P, nodes=_topo(outer + [wrapper]), outputs=outputs)


def unwrap(graph: Graph, parent: str, node_id: str) -> None:
    P = graph.defs[parent]
    node = next((n for n in P["nodes"] if n["id"] == node_id), None)
    if node is None:
        raise TransactionError(f"no node {node_id} in {parent}")
    B = graph.defs.get(node["use"])
    if not B or B["kind"] != "bundle":
        raise TransactionError("unwrap target must be a bundle instance")
    prefix = f"{node_id}__"
    inlined = []
    for n in B["nodes"]:
        n2 = copy.deepcopy(n)
        n2["id"] = prefix + n["id"]
        for port, ref in n.get("in", {}).items():
            k, tgt, p = parse_ref(ref)
            n2["in"][port] = node["in"][tgt] if k == "input" else prefix + ref
        inlined.append(n2)

    def mapped(port):
        ref = B["outputs"][port]
        k, tgt, p = parse_ref(ref)
        return node["in"][tgt] if k == "input" else prefix + ref

    def rewrite(ref):
        k, tgt, p = parse_ref(ref)
        if k == "node" and tgt == node_id:
            if p is None:
                if len(B["outputs"]) != 1:
                    raise TransactionError("ambiguous reference to multi-output node")
                p = next(iter(B["outputs"]))
            return mapped(p)
        return ref

    new_nodes = []
    for n in P["nodes"]:
        if n["id"] == node_id:
            new_nodes.extend(inlined)
            continue
        n2 = copy.deepcopy(n)
        n2["in"] = {k: rewrite(v) for k, v in n.get("in", {}).items()}
        new_nodes.append(n2)
    graph.defs[parent] = dict(P, nodes=new_nodes, outputs={k: rewrite(v) for k, v in P["outputs"].items()})


# ----------------------------------------------------------------- applying
def apply_ops(graph: Graph, ops: list[dict]) -> tuple[Graph, bool]:
    """Apply ops to a copy. Returns (new_graph, pure_refactor_only)."""
    g = graph.copy()
    pure = True
    for i, op in enumerate(ops):
        t = op.get("op")
        try:
            if t == "set_param":
                if "value" in op:
                    a = np.asarray(op["value"], dtype=np.float64)
                elif "pattern" in op:
                    a = expand_pattern(op["pattern"], g)
                else:
                    raise TransactionError("set_param needs value or pattern")
                if "shape" in op:
                    a = a.reshape(tuple(op["shape"]))
                if not np.all(np.isfinite(a)):
                    raise TransactionError("non-finite parameter values")
                g.params[op["name"]] = a
                pure = False
            elif t == "delete_param":
                del g.params[op["name"]]
                pure = False
            elif t == "put_def":
                d = op["def"]
                if not isinstance(d, dict):
                    raise TransactionError("def must be an object")
                g.defs[op["name"]] = copy.deepcopy(d)
                pure = False
            elif t == "delete_def":
                del g.defs[op["name"]]
                pure = False
            elif t == "set_entry":
                if op["entry"] not in ("encode", "step"):
                    raise TransactionError("entry must be encode or step")
                g.entries[op["entry"]] = op["def"]
                pure = False
            elif t == "set_state_dim":
                g.meta["state_dim"] = int(op["value"])
                pure = False
            elif t == "wrap":
                wrap(g, op["parent"], op["nodes"], op["new_def"], op["new_node"])
            elif t == "unwrap":
                unwrap(g, op["parent"], op["node"])
            else:
                raise TransactionError(f"unknown op type {t!r}")
        except (KeyError, TypeError, ValueError) as e:
            raise TransactionError(f"op {i} ({t}): {e}") from e
    # drop parameters no definition references? No: unreferenced params are kept
    # explicitly and reported; authors delete them deliberately.
    return g, pure


def behavior_equal(a: Graph, b: Graph, imgs: np.ndarray, tol: float = 1e-9) -> tuple[bool, str]:
    pa, ia = decode(a, imgs)
    pb, ib = decode(b, imgs)
    if pa != pb or ia["failure"] != ib["failure"]:
        return False, "predictions differ"
    if len(ia["step_logits"]) != len(ib["step_logits"]):
        return False, "step count differs"
    for (xa, la), (xb, lb) in zip(ia["step_logits"], ib["step_logits"]):
        if not np.array_equal(xa, xb) or la.shape != lb.shape:
            return False, "active set differs"
        if not np.allclose(la, lb, atol=tol, rtol=0, equal_nan=True):
            return False, f"logits differ by {np.nanmax(np.abs(la - lb)):.3g}"
    return True, "equivalent"


def validate_proposal(graph: Graph, proposal: dict, sample_imgs: np.ndarray,
                      limits: Limits | None = None) -> dict:
    """Validate against exact base. Returns dict with new graph or error."""
    if proposal.get("base") != graph.hash():
        return {"ok": False, "error": "stale or missing base hash"}
    ops = proposal.get("ops")
    if not isinstance(ops, list) or not ops:
        return {"ok": False, "error": "proposal has no ops"}
    try:
        g, pure = apply_ops(graph, ops)
        g.validate(limits)
        preds, info = decode(g, sample_imgs)
    except (TransactionError, GraphError) as e:
        return {"ok": False, "error": str(e)[:500]}
    except (ValueError, IndexError, FloatingPointError, MemoryError) as e:
        return {"ok": False, "error": f"execution error: {type(e).__name__}: {e}"[:500]}
    res = {"ok": True, "graph": g, "pure_refactor": pure, "new_hash": g.hash()}
    if pure:
        eq, why = behavior_equal(graph, g, sample_imgs)
        res["equivalence"] = why
        if not eq:
            return {"ok": False, "error": f"refactor changed behavior: {why}"}
    return res
