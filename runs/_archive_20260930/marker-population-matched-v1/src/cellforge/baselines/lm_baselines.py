"""Control arms for tinylm (fitted from data; never used by the primary arm).

* n-gram count models (order 1-3, add-k smoothing) exported to the IR.
* gradient-trained MLP over the 8-token context window (Adam), exported to the IR.
All are trained only on the author-visible ``dev_train`` text.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..construction.bootstrap import bootstrap_lm
from ..reference.protocols import PROTOCOLS
from ..runtime.lm import encode_text


def _seqs(vocab, texts):
    return [encode_text(vocab, t) for t in texts]


def ngram_graph(texts, order: int, k: float = 0.1, protocol="tinylm/1"):
    proto = PROTOCOLS[protocol]
    g = bootstrap_lm(proto["charset"], proto["ctx_len"])
    vocab = g.meta["vocab"]
    V = len(vocab)
    counts = np.full((V ** (order - 1), V), k)
    for s in _seqs(vocab, texts):
        padded = [vocab.index("<bos>")] * (order - 2) + s if order > 2 else s
        for i in range(order - 1, len(padded)) if order > 1 else range(1, len(padded)):
            ctx = padded[i - order + 1:i]
            row = 0
            for c in ctx:
                row = row * V + c
            counts[row, padded[i]] += 1
    logp = np.log(counts / counts.sum(1, keepdims=True))
    g.params["ng.W1"] = logp.T.copy()  # [V, V^(order-1)]
    g.params["ng.b1"] = np.zeros(V)
    g.defs["ng_cell"] = {"kind": "cell", "act": "identity", "params": {"W1": "ng.W1", "b1": "ng.b1"}}
    K = proto["ctx_len"]
    nodes = []
    if order == 1:
        nodes = [{"id": "one", "use": "op:sum", "in": {"x": "$prev"}, "args": {"axis": 0, "keepdims": True}},
                 {"id": "lg", "use": "ng_cell", "in": {"x": "one"}}]
    elif order == 2:
        nodes = [{"id": "lg", "use": "ng_cell", "in": {"x": "$prev"}}]
    elif order == 3:
        nodes = [
            {"id": "p2", "use": "op:slice", "in": {"x": "$ctx"}, "args": {"axis": 0, "start": K - 2, "stop": K - 1}},
            {"id": "p2t", "use": "op:transpose", "in": {"x": "p2"}, "args": {"perm": [1, 0]}},
            {"id": "p1", "use": "op:slice", "in": {"x": "$ctx"}, "args": {"axis": 0, "start": K - 1, "stop": K}},
            {"id": "outer", "use": "op:mul", "in": {"a": "p2t", "b": "p1"}},
            {"id": "flat", "use": "op:reshape", "in": {"x": "outer"}, "args": {"shape": [V * V]}},
            # doc start: ctx row K-2 is all-zero; treat as <bos>
            {"id": "lg", "use": "ng_cell", "in": {"x": "flat"}}]
        # route zero-row case to <bos> by adding a bos indicator: handled by counting with bos padding
        # and an extra term: if p2 is empty, use row (bos, p1)
        bos = vocab.index("<bos>")
        sel = np.zeros((V, V * V))  # maps p1 one-hot -> (bos,p1) flat index
        for c in range(V):
            sel[c, bos * V + c] = 1.0
        g.params["ng.bosmap"] = sel
        nodes[-1:] = [
            {"id": "has2", "use": "op:sum", "in": {"x": "p2"}, "args": {"axis": -1, "keepdims": True}},
            {"id": "no2", "use": "op:scale", "in": {"x": "has2"}, "args": {"c": -1.0}},
            {"id": "no2b", "use": "op:bias", "in": {"x": "no2"}, "args": {"c": 1.0}},
            {"id": "bm", "use": "param:ng.bosmap"},
            {"id": "p1b", "use": "op:matmul", "in": {"a": "p1", "b": "bm"}},
            {"id": "p1bs", "use": "op:mul", "in": {"a": "p1b", "b": "no2b"}},
            {"id": "p1f", "use": "op:reshape", "in": {"x": "p1bs"}, "args": {"shape": [V * V]}},
            {"id": "flat2", "use": "op:add", "in": {"a": "flat", "b": "p1f"}},
            {"id": "lg", "use": "ng_cell", "in": {"x": "flat2"}}]
    g.defs["step_root"] = {"kind": "bundle", "inputs": ["ctx", "prev", "step", "state"], "nodes": nodes,
                           "outputs": {"logits": "lg", "state": "$state"}}
    del g.params["boot_logits"]
    g.meta["baseline"] = f"{order}-gram counts add-k={k} on dev_train"
    return g


def mlp_graph(texts, hidden=256, epochs=8, lr=2e-3, seed=0, protocol="tinylm/1", batch=512):
    proto = PROTOCOLS[protocol]
    g = bootstrap_lm(proto["charset"], proto["ctx_len"])
    vocab = g.meta["vocab"]
    V, K = len(vocab), proto["ctx_len"]
    X, Y = [], []
    for s in _seqs(vocab, texts):
        for i in range(1, len(s)):
            ctx = s[max(0, i - K):i]
            X.append([-1] * (K - len(ctx)) + ctx)
            Y.append(s[i])
    X = np.array(X)
    Y = np.array(Y)
    Xo = np.zeros((len(X), K * V), dtype=np.float32)
    for j in range(K):
        m = X[:, j] >= 0
        Xo[np.nonzero(m)[0], j * V + X[m, j]] = 1.0
    rng = np.random.default_rng(seed)
    P = {"W1": rng.normal(0, np.sqrt(2 / (K * V)), (hidden, K * V)), "b1": np.zeros(hidden),
         "W2": rng.normal(0, np.sqrt(1 / hidden), (V, hidden)), "b2": np.zeros(V)}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v2 = {k: np.zeros_like(v) for k, v in P.items()}
    t = 0
    log = []
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            x = Xo[idx].astype(np.float64)
            hp = x @ P["W1"].T + P["b1"]
            h = np.maximum(hp, 0)
            z = h @ P["W2"].T + P["b2"]
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            tot += -np.log(p[np.arange(len(idx)), Y[idx]] + 1e-12).sum()
            dz = p
            dz[np.arange(len(idx)), Y[idx]] -= 1
            dz /= len(idx)
            gr = {"W2": dz.T @ h, "b2": dz.sum(0)}
            dh = dz @ P["W2"] * (hp > 0)
            gr["W1"] = dh.T @ x
            gr["b1"] = dh.sum(0)
            t += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * gr[k]
                v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
        log.append({"epoch": ep, "train_bpc": tot / len(X) / np.log(2), "elapsed_s": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
    for k, val in P.items():
        g.params[f"mlp.{k}"] = val
    g.defs["mlp"] = {"kind": "cell", "act": "relu", "params": {k: f"mlp.{k}" for k in P}}
    g.defs["step_root"] = {"kind": "bundle", "inputs": ["ctx", "prev", "step", "state"], "nodes": [
        {"id": "flat", "use": "op:reshape", "in": {"x": "$ctx"}, "args": {"shape": [K * V]}},
        {"id": "lg", "use": "mlp", "in": {"x": "flat"}}], "outputs": {"logits": "lg", "state": "$state"}}
    del g.params["boot_logits"]
    g.meta["baseline"] = f"gradient MLP ctx{K} H={hidden} epochs={epochs} seed={seed} on dev_train"
    return g, log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ngram", "mlp"])
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--protocol", default="tinylm/1")
    ap.add_argument("--train", default="data/tinylm/dev_train.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    texts = json.loads(Path(a.train).read_text())
    if a.kind == "ngram":
        g = ngram_graph(texts, a.order, protocol=a.protocol)
    else:
        g, log = mlp_graph(texts, a.hidden, a.epochs, protocol=a.protocol)
    g.validate()
    g.save(a.out)
    print(json.dumps({"saved": a.out, "baseline": g.meta["baseline"], "params": g.stats()["unique_param_storage"]}))


if __name__ == "__main__":
    main()
