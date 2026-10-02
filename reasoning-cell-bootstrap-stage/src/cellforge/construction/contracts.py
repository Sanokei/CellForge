"""Component contracts: unit-test any cell, cluster, expert or router in isolation.

A parent author (orchestrator or expert-maker) writes a contract for a
component it delegates; the child author must satisfy it before integration.

Contract JSON::

    {"component": "cl_vowel",            # definition name
     "inputs": {"x": {"shape": [32]}},   # per-example shapes
     "output": "y",                      # output port to check
     "cases": [
        {"in": {"x": {"onehot": 6, "n": 32}}, "expect": [1.0], "tol": 0.1},
        {"in": {"x": [0, 0, ...]},           "expect_argmax": 3},
        {"in": {"x": {"onehot": 3, "n": 32}}, "expect_sign": [-1]}
     ]}

Input value forms: explicit nested list, ``{"onehot": i, "n": n}``,
``{"onehots": [i, j, ...], "n": n}`` (stacked rows, e.g. a ctx window;
``-1`` gives an all-zero row), ``{"fill": v, "shape": [...]}``.
Expectation forms: ``expect`` (+``tol``, default 1e-6), ``expect_argmax``,
``expect_sign`` (elementwise sign, 0 = don't care), ``expect_gt``/``expect_lt``
(elementwise bounds).  Contracts are specifications written by authors; they
are not fitted to data and carry no authority over reference evaluation.
"""
from __future__ import annotations

import numpy as np

from ..ir.graph import Graph, GraphError
from ..runtime.interp import Interpreter


def _value(spec):
    if isinstance(spec, dict):
        if "onehot" in spec:
            v = np.zeros(int(spec["n"]))
            v[int(spec["onehot"])] = 1.0
            return v
        if "onehots" in spec:
            n = int(spec["n"])
            rows = np.zeros((len(spec["onehots"]), n))
            for r, i in enumerate(spec["onehots"]):
                if int(i) >= 0:
                    rows[r, int(i)] = 1.0
            return rows
        if "fill" in spec:
            return np.full(tuple(spec["shape"]), float(spec["fill"]))
        raise ValueError(f"unknown input spec {sorted(spec)}")
    return np.asarray(spec, dtype=np.float64)


def run_contract(graph: Graph, contract: dict) -> dict:
    name = contract["component"]
    if name not in graph.defs:
        return {"ok": False, "error": f"unknown component {name}"}
    d = graph.defs[name]
    port = contract.get("output", "y")
    results = []
    passed = 0
    for ci, case in enumerate(contract["cases"]):
        ins = {k: _value(v)[None, ...] for k, v in case["in"].items()}
        ins2 = {k: np.concatenate([v, v], axis=0) for k, v in ins.items()}  # batch-of-2 must agree
        it = Interpreter(graph)
        try:
            if d["kind"] == "cell":
                out = it.run_cell(name, ins["x"])
            else:
                outs = it.run_bundle(name, ins, f"/{name}")
                out = outs[port]
        except (GraphError, KeyError, ValueError, IndexError) as e:
            results.append({"case": ci, "ok": False, "error": str(e)[:200]})
            continue
        y = np.asarray(out)[0]
        ok = bool(np.all(np.isfinite(y)))
        why = []
        try:
            it2 = Interpreter(graph)
            if d["kind"] == "cell":
                out2 = it2.run_cell(name, ins2["x"])
            else:
                out2 = it2.run_bundle(name, ins2, f"/{name}")[port]
            out2 = np.asarray(out2)
            if out2.shape[0] != 2 or not np.allclose(out2[0], y, atol=1e-9) or not np.allclose(out2[1], y, atol=1e-9):
                ok = False
                why.append("batch-2 output differs from batch-1")
        except (GraphError, KeyError, ValueError, IndexError) as e:
            ok = False
            why.append("batch-2 execution failed: " + str(e)[:120])
        if "expect" in case:
            e = np.asarray(case["expect"], dtype=np.float64)
            tol = float(case.get("tol", 1e-6))
            if y.shape != e.shape or not np.allclose(y, e, atol=tol, rtol=0):
                ok = False
                why.append("value")
        if "expect_argmax" in case and int(np.argmax(y)) != int(case["expect_argmax"]):
            ok = False
            why.append(f"argmax {int(np.argmax(y))}")
        if "expect_sign" in case:
            s = np.asarray(case["expect_sign"])
            if np.any((s != 0) & (np.sign(y.ravel()[:s.size]) != s)):
                ok = False
                why.append("sign")
        if "expect_gt" in case and np.any(y <= np.asarray(case["expect_gt"])):
            ok = False
            why.append("gt")
        if "expect_lt" in case and np.any(y >= np.asarray(case["expect_lt"])):
            ok = False
            why.append("lt")
        passed += ok
        results.append({"case": ci, "ok": ok, "got": np.round(y, 4).ravel()[:40].tolist(), "fail": why})
    return {"ok": passed == len(contract["cases"]), "component": name, "passed": passed,
            "total": len(contract["cases"]), "cases": results}
