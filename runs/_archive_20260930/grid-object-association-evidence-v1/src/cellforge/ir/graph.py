"""Recursive component registry: primitive cells and composable bundles.

A graph is a registry of named definitions plus a parameter store.
Definitions come in two kinds:

* ``cell`` -- a parameterized primitive ``h = act(W1 x + b1); y = W2 h + b2``
  (``W2``/``b2`` optional, giving ``y = act(W1 x + b1)``).  Applied to the last
  axis of its input, broadcasting over all leading axes.
* ``bundle`` -- a typed DAG whose nodes reference ops, params, cells, other
  bundles, or a conditional ``dispatch`` among bundles.  Every nesting level
  uses this same representation.

Node ``use`` values:
  ``op:<name>``      parameter-free primitive from ``runtime.ops``
  ``param:<name>``   emits a stored parameter tensor (counted as parameters)
  ``dispatch``       top-1 conditional execution among ``args.branches``
  ``<def name>``     instance of a cell or bundle definition

Input references: ``$name`` (bundle input), ``node`` (default output) or
``node.port``.  Cells take ``x`` and produce ``y``.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..runtime.ops import ACTIVATIONS, CONST_ARGS, OPS

FORMAT = "cellforge-graph/1"
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:\-]{0,127}$")
STRICT_CELL_SLOTS = 40


class GraphError(ValueError):
    pass


@dataclass
class Limits:
    max_unique_params: int = 2_000_000
    max_expanded_instances: int = 50_000
    max_defs: int = 5_000
    max_depth: int = 64
    strict_microcells: bool = False


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a, dtype=np.float64)
    return sha256_bytes(str(a.shape).encode() + a.tobytes())


def parse_ref(ref: str):
    if ref.startswith("$"):
        return ("input", ref[1:], None)
    if "." in ref:
        node, port = ref.split(".", 1)
        return ("node", node, port)
    return ("node", ref, None)


@dataclass
class Graph:
    defs: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    entries: dict = field(default_factory=dict)  # {"encode": def, "step": def}
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    def copy(self) -> "Graph":
        return Graph(copy.deepcopy(self.defs), {k: v.copy() for k, v in self.params.items()},
                     dict(self.entries), copy.deepcopy(self.meta))

    def manifest(self) -> dict:
        return {
            "format": FORMAT,
            "defs": self.defs,
            "entries": self.entries,
            "meta": self.meta,
            "params": {k: {"shape": list(v.shape), "sha256": array_hash(v)}
                       for k, v in sorted(self.params.items())},
        }

    def hash(self) -> str:
        return sha256_bytes(canonical_json(self.manifest()))

    def save(self, directory: str | Path) -> str:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        man = self.manifest()
        (d / "graph.json").write_bytes(json.dumps(man, indent=1, sort_keys=True).encode())
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: np.asarray(v, dtype=np.float64) for k, v in self.params.items()})
        (d / "params.npz").write_bytes(buf.getvalue())
        h = sha256_bytes(canonical_json(man))
        (d / "HASH").write_text(h + "\n")
        return h

    @classmethod
    def load(cls, directory: str | Path, max_bytes: int = 512 * 1024 * 1024) -> "Graph":
        d = Path(directory)
        man_bytes = (d / "graph.json").read_bytes()
        if len(man_bytes) > max_bytes:
            raise GraphError("graph manifest too large")
        man = json.loads(man_bytes)
        if man.get("format") != FORMAT:
            raise GraphError(f"unsupported format {man.get('format')!r}")
        params = {}
        pfile = d / "params.npz"
        if pfile.stat().st_size > max_bytes:
            raise GraphError("parameter file too large")
        with np.load(pfile, allow_pickle=False) as z:
            for k, spec in man["params"].items():
                if k not in z.files:
                    raise GraphError(f"missing parameter array {k}")
                a = np.asarray(z[k], dtype=np.float64)
                if list(a.shape) != spec["shape"] or array_hash(a) != spec["sha256"]:
                    raise GraphError(f"parameter {k} fails integrity check")
                params[k] = a
        g = cls(man["defs"], params, man["entries"], man["meta"])
        return g

    # ------------------------------------------------------------ analysis
    def children(self, name: str) -> list[str]:
        """Definition names referenced by a definition (instances + branches)."""
        d = self.defs[name]
        out = []
        if d["kind"] == "bundle":
            for n in d["nodes"]:
                use = n["use"]
                if use == "dispatch":
                    out.extend(n.get("args", {}).get("branches", []))
                elif not (use.startswith("op:") or use.startswith("param:")):
                    out.append(use)
        return out

    def params_of(self, name: str) -> list[str]:
        d = self.defs[name]
        if d["kind"] == "cell":
            return list(d["params"].values())
        return [n["use"][6:] for n in d["nodes"] if n["use"].startswith("param:")]

    def validate(self, limits: Limits | None = None) -> None:
        limits = limits or Limits()
        if len(self.defs) > limits.max_defs:
            raise GraphError("too many definitions")
        for k, v in self.params.items():
            if not NAME_RE.match(k):
                raise GraphError(f"bad parameter name {k!r}")
            if not np.all(np.isfinite(v)):
                raise GraphError(f"parameter {k} contains non-finite values")
        n_params = sum(int(v.size) for v in self.params.values())
        if n_params > limits.max_unique_params:
            raise GraphError(f"{n_params} unique parameters exceeds limit {limits.max_unique_params}")
        for name, d in self.defs.items():
            if not NAME_RE.match(name):
                raise GraphError(f"bad definition name {name!r}")
            self._validate_def(name, d, limits)
        required = {"lm": ("step",), "clf": ("predict",), "grid": ("grid_step",),
                    "binding": ("predict",), "arithmetic": ("step",),
                    "deduction": ("step",)}.get(self.meta.get("task"), ("encode", "step"))
        for entry in required:
            if entry not in self.entries:
                raise GraphError(f"missing entry {entry}")
            e = self.entries[entry]
            if e not in self.defs or self.defs[e]["kind"] != "bundle":
                raise GraphError(f"entry {entry} must name a bundle")
        # recursion / cycles
        state: dict[str, int] = {}

        def visit(n, depth):
            if depth > limits.max_depth:
                raise GraphError("nesting depth limit exceeded")
            s = state.get(n)
            if s == 1:
                raise GraphError(f"recursive definition reference through {n}")
            if s == 2:
                return
            state[n] = 1
            for c in self.children(n):
                if c not in self.defs:
                    raise GraphError(f"{n} references unknown definition {c}")
                visit(c, depth + 1)
            state[n] = 2

        for n in self.defs:
            visit(n, 0)
        exp = self.expanded_instances()
        if exp > limits.max_expanded_instances:
            raise GraphError(f"{exp} expanded instances exceeds limit")

    def _validate_def(self, name, d, limits):
        kind = d.get("kind")
        if kind == "cell":
            p = d.get("params", {})
            if "W1" not in p or "b1" not in p:
                raise GraphError(f"cell {name} needs W1 and b1")
            if ("W2" in p) != ("b2" in p):
                raise GraphError(f"cell {name}: W2 and b2 must appear together")
            if d.get("act", "identity") not in ACTIVATIONS:
                raise GraphError(f"cell {name}: unknown activation {d.get('act')}")
            for role, pn in p.items():
                if pn not in self.params:
                    raise GraphError(f"cell {name}: missing parameter {pn}")
            W1, b1 = self.params[p["W1"]], self.params[p["b1"]]
            if W1.ndim != 2 or b1.shape != (W1.shape[0],):
                raise GraphError(f"cell {name}: W1 must be [hidden,in] and b1 [hidden]")
            slots = W1.size + b1.size
            if "W2" in p:
                W2, b2 = self.params[p["W2"]], self.params[p["b2"]]
                if W2.ndim != 2 or W2.shape[1] != W1.shape[0] or b2.shape != (W2.shape[0],):
                    raise GraphError(f"cell {name}: W2 must be [out,hidden] and b2 [out]")
                slots += W2.size + b2.size
            if limits.strict_microcells and slots > STRICT_CELL_SLOTS:
                raise GraphError(f"cell {name}: {slots} slots exceeds strict microcell cap")
        elif kind == "bundle":
            ids = set()
            inputs = set(d.get("inputs", []))
            for n in d.get("nodes", []):
                nid = n.get("id")
                if not nid or not NAME_RE.match(nid) or "." in nid or nid in ids:
                    raise GraphError(f"bundle {name}: bad or duplicate node id {nid!r}")
                use = n.get("use", "")
                if use.startswith("op:"):
                    if use[3:] not in OPS:
                        raise GraphError(f"bundle {name}: unknown op {use}")
                elif use.startswith("param:"):
                    if use[6:] not in self.params:
                        raise GraphError(f"bundle {name}: unknown param {use}")
                elif use == "dispatch":
                    br = n.get("args", {}).get("branches")
                    if not br or "scores" not in n.get("in", {}):
                        raise GraphError(f"bundle {name}: dispatch needs scores and branches")
                elif use not in self.defs:
                    raise GraphError(f"bundle {name}: node {nid} uses unknown {use}")
                for port, ref in n.get("in", {}).items():
                    k, tgt, _ = parse_ref(ref)
                    if k == "input" and tgt not in inputs:
                        raise GraphError(f"bundle {name}: unknown input {ref}")
                    if k == "node" and tgt not in ids:
                        raise GraphError(f"bundle {name}: node {nid} references {ref} before definition")
                ids.add(nid)
            outs = d.get("outputs", {})
            if not outs:
                raise GraphError(f"bundle {name}: no outputs")
            for o, ref in outs.items():
                k, tgt, _ = parse_ref(ref)
                if (k == "input" and tgt not in inputs) or (k == "node" and tgt not in ids):
                    raise GraphError(f"bundle {name}: output {o} references unknown {ref}")
        else:
            raise GraphError(f"definition {name}: unknown kind {kind!r}")

    def expanded_instances(self) -> int:
        memo: dict[str, int] = {}

        def count(n):
            if n in memo:
                return memo[n]
            memo[n] = 1 + sum(count(c) for c in self.children(n) if c in self.defs)
            return memo[n]
        return sum(count(self.entries[e]) for e in self.entries)

    def stats(self) -> dict:
        """Resource and structure counts (bundle count is not capability evidence)."""
        uses: dict[str, int] = {}
        depth_memo: dict[str, int] = {}
        edge_count = 0
        const_slots = 0
        reachable: dict[str, int] = {}

        def walk(n, mult):
            reachable[n] = reachable.get(n, 0) + mult
            for c in self.children(n):
                walk(c, mult)

        for e in list(self.entries):
            if e in self.entries:
                walk(self.entries[e], 1)

        def depth(n):
            if n not in depth_memo:
                ch = self.children(n)
                depth_memo[n] = 1 + (max(depth(c) for c in ch) if ch else 0)
            return depth_memo[n]

        for name, mult in reachable.items():
            for p in self.params_of(name):
                uses[p] = uses.get(p, 0) + mult
            d = self.defs[name]
            if d["kind"] == "bundle":
                for n in d["nodes"]:
                    edge_count += len(n.get("in", {}))
                    if n["use"].startswith("op:"):
                        for a in CONST_ARGS.get(n["use"][3:], []):
                            if a in n.get("args", {}):
                                const_slots += 1
        unique = sum(int(v.size) for v in self.params.values())
        return {
            "unique_param_storage": unique,
            "expanded_param_uses": int(sum(self.params[p].size * m for p, m in uses.items() if p in self.params)),
            "nonzero_slots": int(sum(np.count_nonzero(v) for v in self.params.values())),
            "constant_slots_in_op_args": const_slots,
            "cell_defs": sum(1 for d in self.defs.values() if d["kind"] == "cell"),
            "bundle_defs": sum(1 for d in self.defs.values() if d["kind"] == "bundle"),
            "reachable_defs": len(reachable),
            "expanded_instances": self.expanded_instances(),
            "nesting_depth": max((depth(self.entries[e]) for e in self.entries), default=0),
            "edges": edge_count,
            "dispatch_nodes": sum(1 for d in self.defs.values() if d["kind"] == "bundle"
                                  for n in d["nodes"] if n["use"] == "dispatch"),
            "param_bytes_float64": unique * 8,
        }
