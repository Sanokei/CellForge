"""Gradient-trained controls for classification (never used by the primary arm).

* ``logistic``: softmax regression.
* ``mlp``: one hidden layer (ReLU), Adam, minibatches.
Trained on the author-visible dev_train split only; exported into the IR so the
same reference evaluator scores them.  Literature numbers for CNN/GRU/human are
cited in the protocol, not reproduced.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..construction.bootstrap import bootstrap_clf
from ..data.mnist1d_data import split_from_protocol
from ..reference.protocols import PROTOCOLS


def to_graph(P, hidden, tag):
    proto = PROTOCOLS["mnist1d/1"]
    g = bootstrap_clf(proto["n_classes"], proto["input_length"])
    del g.params["boot_logits"]
    if hidden:
        for k, v in P.items():
            g.params[f"mlp.{k}"] = v
        g.defs["mlp"] = {"kind": "cell", "act": "relu", "params": {k: f"mlp.{k}" for k in P}}
        use, pk = "mlp", None
    else:
        g.params["lin.W1"] = P["W"]
        g.params["lin.b1"] = P["b"]
        g.defs["mlp"] = {"kind": "cell", "act": "identity", "params": {"W1": "lin.W1", "b1": "lin.b1"}}
    g.defs["predict_root"] = {"kind": "bundle", "inputs": ["x"],
                              "nodes": [{"id": "lg", "use": "mlp", "in": {"x": "$x"}}],
                              "outputs": {"logits": "lg"}}
    g.meta["baseline"] = tag
    return g


def train(kind, hidden, epochs, seed, lr=3e-3, batch=100, l2=0.0):
    proto = PROTOCOLS["mnist1d/1"]
    X, y = split_from_protocol(proto, "dev_train")
    Xe, ye = split_from_protocol(proto, "dev_eval")
    C, L = proto["n_classes"], proto["input_length"]
    rng = np.random.default_rng(seed)
    if kind == "logistic":
        P = {"W": rng.normal(0, 0.05, (C, L)), "b": np.zeros(C)}
    else:
        P = {"W1": rng.normal(0, np.sqrt(2 / L), (hidden, L)), "b1": np.zeros(hidden),
             "W2": rng.normal(0, np.sqrt(1 / hidden), (C, hidden)), "b2": np.zeros(C)}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v2 = {k: np.zeros_like(v) for k, v in P.items()}
    t = 0
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            x, yy = X[idx], y[idx]
            if kind == "logistic":
                z = x @ P["W"].T + P["b"]
            else:
                hp = x @ P["W1"].T + P["b1"]
                h = np.maximum(hp, 0)
                z = h @ P["W2"].T + P["b2"]
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            dz = p
            dz[np.arange(len(idx)), yy] -= 1
            dz /= len(idx)
            if kind == "logistic":
                gr = {"W": dz.T @ x + l2 * P["W"], "b": dz.sum(0)}
            else:
                dh = dz @ P["W2"] * (hp > 0)
                gr = {"W2": dz.T @ h + l2 * P["W2"], "b2": dz.sum(0), "W1": dh.T @ x + l2 * P["W1"], "b1": dh.sum(0)}
            t += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * gr[k]
                v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
    g = to_graph(P, hidden if kind == "mlp" else 0, f"{kind} H={hidden} epochs={epochs} seed={seed} on dev_train (4000)")
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["logistic", "mlp"])
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    g = train(a.kind, a.hidden, a.epochs, a.seed)
    g.validate()
    g.save(a.out)
    print(json.dumps({"saved": a.out, "params": g.stats()["unique_param_storage"], "baseline": g.meta["baseline"]}))


if __name__ == "__main__":
    main()
