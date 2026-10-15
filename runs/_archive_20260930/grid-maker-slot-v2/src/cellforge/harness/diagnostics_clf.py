"""Development diagnostics for classification (agent-improvable, versioned)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..runtime.clf import evaluate_clf

DIAGNOSTICS_VERSION = "dev-diag-clf/1"


def run(graph, X, y, out_dir=None, max_wrong=12) -> dict:
    r = evaluate_clf(graph, X, y)
    if "error" in r:
        return {"version": DIAGNOSTICS_VERSION, "error": r["error"]}
    conf = np.array(r["confusion"])
    C = conf.shape[0]
    per_class = {c: round(conf[c, c] / max(1, conf[c].sum()), 3) for c in range(C)}
    off = [(conf[i, j], i, j) for i in range(C) for j in range(C) if i != j and conf[i, j] > 0]
    off.sort(reverse=True)
    wrong = [i for i in range(len(y)) if r["pred"][i] != y[i]]
    rep = {"version": DIAGNOSTICS_VERSION, "n": r["n"], "acc": round(r["acc"], 4), "failures": r["failures"],
           "per_class_acc": per_class,
           "top_confusions": [f"true {i} -> pred {j}: {int(n)}" for n, i, j in off[:8]],
           "wrong_examples": [{"i": int(i), "true": int(y[i]), "pred": int(r["pred"][i])} for i in wrong[:max_wrong]],
           "exec": {k: v for k, v in r["exec"].items() if k != "route_counts"},
           "route_counts": r["exec"]["route_counts"]}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "dev_signals_wrong.npy", X[wrong[:max_wrong]] if wrong else np.zeros((0, X.shape[1])))
        (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep
