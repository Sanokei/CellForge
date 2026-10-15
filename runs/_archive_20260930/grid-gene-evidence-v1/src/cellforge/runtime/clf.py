"""Classification driver (protocol family ``mnist1d``).

The model is a single ``predict`` bundle: input ``x`` [L] (one signal per
example), output ``logits`` [C].  Prediction is argmax with ties to the lowest
class index; non-finite logits are recorded failures counted as errors.
"""
from __future__ import annotations

import numpy as np

from ..ir.graph import Graph, GraphError
from .interp import Interpreter


def predict_logits(graph: Graph, X: np.ndarray, chunk: int = 1000, trace_filter=None, trace=False):
    C = int(graph.meta["n_classes"])
    outs, traces, stats = [], {}, None
    it = Interpreter(graph, trace=trace, trace_filter=trace_filter)
    for i in range(0, len(X), chunk):
        out = it.run_bundle(graph.entries["predict"], {"x": np.asarray(X[i:i + chunk], dtype=np.float64)}, "/predict")
        lg = np.asarray(out["logits"]).reshape(len(X[i:i + chunk]), -1)
        if lg.shape[1] != C:
            raise GraphError(f"logits width {lg.shape[1]} != n_classes {C}")
        outs.append(lg)
    return np.concatenate(outs), it


def evaluate_clf(graph: Graph, X: np.ndarray, y: np.ndarray) -> dict:
    try:
        lg, it = predict_logits(graph, X)
    except (GraphError, ValueError, IndexError, FloatingPointError) as e:
        return {"n": len(y), "acc": 0.0, "failures": len(y), "error": str(e)[:300]}
    bad = ~np.all(np.isfinite(lg), axis=1)
    pred = np.argmax(np.where(np.isfinite(lg), lg, -np.inf), axis=1)
    ok = (pred == y) & ~bad
    C = int(graph.meta["n_classes"])
    conf = np.zeros((C, C), dtype=int)
    for t, p, b in zip(y, pred, bad):
        if not b:
            conf[t, p] += 1
    return {"n": int(len(y)), "acc": float(ok.mean()), "correct": int(ok.sum()), "failures": int(bad.sum()),
            "confusion": conf.tolist(), "pred": pred, "exec": it.stats.as_dict()}
