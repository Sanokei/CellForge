"""Operator-side final evaluation for mnist1d protocols (evaluation server only).

Every run draws a fresh secret seed and generates new signals with the same
procedural generator (never the authors' dev seeds), scores the artifact, and
publishes aggregates only (accuracy with Wilson interval, per-class accuracy).
"""
from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

import numpy as np

from ..data.mnist1d_data import make_split
from ..ir.graph import Graph
from ..runtime.clf import evaluate_clf
from .bench_operator import wilson
from .protocols import PROTOCOLS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--protocol", default="mnist1d/1")
    ap.add_argument("--private", required=True)
    ap.add_argument("--public", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    proto = PROTOCOLS[a.protocol]
    g = Graph.load(a.artifact)
    seed = secrets.randbits(31)
    n = proto["final"]["n"]
    # num_samples=2n with train_split=0.5 -> n test signals
    X, y = make_split(seed, 2 * n, 0.5, "x_test")
    d = Path(a.private) / "final_clf" / (time.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3))
    d.mkdir(parents=True)
    (d / "SEED").write_text(str(seed))
    r = evaluate_clf(g, X, y)
    conf = np.array(r.pop("confusion")) if "confusion" in r else None
    r.pop("pred", None)
    out = {"label": a.label, "protocol": a.protocol, "model_hash": g.hash(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n": r["n"], "acc": round(r["acc"], 4), "acc_wilson95": wilson(r.get("correct", 0), r["n"]),
           "failures": r["failures"], "chance": 0.1,
           "per_class_acc": ({str(c): round(conf[c, c] / max(1, conf[c].sum()), 3) for c in range(conf.shape[0])}
                             if conf is not None else None),
           "macs_total": (r.get("exec") or {}).get("macs"),
           "literature_reference": proto["literature_reference"]}
    with (Path(a.public) / "final_results_clf.jsonl").open("a") as f:
        f.write(json.dumps(out, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
