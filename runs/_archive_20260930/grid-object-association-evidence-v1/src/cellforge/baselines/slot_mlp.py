"""Separate comparison arms (never used to initialize or repair the primary).

* ``gradient``: a conventional slot-MLP trained with backpropagation + Adam.
* ``random``:   (1+1) random direct-edit hill climbing on the same architecture.

Both export into the CellForge IR so they are scored by the same reference
evaluator and decoder as authored artifacts.  Architecture::

    encode: image[32,128] -> flatten[4096] -> relu(W1 x + b1)[H] -> W2 h + b2 [S*C] -> [S,C]
    step:   slot weights = softmax(-50 (coords(S) - step)^2); logits_c = w @ slots
            vocab logits = logits_c @ M  (M maps C={EOS,0..9} into the vocab)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..construction.bootstrap import vocab_for
from ..data.render import RenderSpec, generate
from ..ir.graph import Graph
from ..reference.protocols import PROTOCOLS

S = 9  # 8 characters + EOS slot


def to_graph(W1, b1, W2, b2, charset: str, max_len: int = 12, tag: str = "") -> Graph:
    vocab = vocab_for(charset)
    C = len(charset) + 1
    M = np.full((C, len(vocab)), -100.0)
    M[0, vocab.index("<eos>")] = 0.0
    for i, ch in enumerate(charset):
        M[i + 1, vocab.index(ch)] = 0.0
    # M is additive-mask style: logits_v = max_c (l_c + M[c,v]); implement as matmul with 0/1 instead
    M01 = (M == 0).astype(np.float64)
    pad = np.zeros(len(vocab))
    pad[[vocab.index("<pad>"), vocab.index("<bos>")]] = -100.0
    g = Graph()
    g.params.update({"mlp.W1": W1, "mlp.b1": b1, "mlp.W2": W2, "mlp.b2": b2, "map": M01, "mask": pad})
    g.defs["mlp"] = {"kind": "cell", "act": "relu",
                     "params": {"W1": "mlp.W1", "b1": "mlp.b1", "W2": "mlp.W2", "b2": "mlp.b2"}}
    g.defs["encode_root"] = {"kind": "bundle", "inputs": ["image"], "nodes": [
        {"id": "flat", "use": "op:reshape", "in": {"x": "$image"}, "args": {"shape": [4096]}},
        {"id": "h", "use": "mlp", "in": {"x": "flat"}},
        {"id": "slots", "use": "op:reshape", "in": {"x": "h"}, "args": {"shape": [S, C]}}],
        "outputs": {"slots": "slots"}}
    g.defs["step_root"] = {"kind": "bundle", "inputs": ["slots", "step", "prev", "state"], "nodes": [
        {"id": "pos", "use": "op:coords", "args": {"n": S}},
        {"id": "d", "use": "op:sub", "in": {"a": "pos", "b": "$step"}},
        {"id": "d2", "use": "op:square", "in": {"x": "d"}},
        {"id": "nd", "use": "op:scale", "in": {"x": "d2"}, "args": {"c": -50.0}},
        {"id": "w", "use": "op:softmax", "in": {"x": "nd"}, "args": {"axis": 0}},
        {"id": "w2", "use": "op:expand", "in": {"x": "w"}, "args": {"axis": 0}},
        {"id": "lc", "use": "op:matmul", "in": {"a": "w2", "b": "$slots"}},
        {"id": "m", "use": "param:map"},
        {"id": "lv", "use": "op:matmul", "in": {"a": "lc", "b": "m"}},
        {"id": "lv1", "use": "op:reshape", "in": {"x": "lv"}, "args": {"shape": [len(vocab)]}},
        {"id": "mk", "use": "param:mask"},
        {"id": "out", "use": "op:add", "in": {"a": "lv1", "b": "mk"}}],
        "outputs": {"logits": "out", "state": "$state"}}
    g.entries = {"encode": "encode_root", "step": "step_root"}
    g.meta = {"vocab": vocab, "state_dim": 0, "max_len": max_len,
              "pixel_contract": {"height": 32, "width": 128, "normalize": "uint8/255", "ink": "bright"},
              "primitive_version": "cellforge-ops/1", "baseline": tag}
    return g


def targets(labels, charset):
    C = len(charset) + 1
    y = np.full((len(labels), S), -1, dtype=np.int64)
    for i, s in enumerate(labels):
        for j, ch in enumerate(s[:S - 1]):
            y[i, j] = charset.index(ch) + 1
        y[i, min(len(s), S - 1)] = 0
    return y


def forward(P, X, C):
    h_pre = X @ P["W1"].T + P["b1"]
    h = np.maximum(h_pre, 0)
    z = (h @ P["W2"].T + P["b2"]).reshape(len(X), S, C)
    return h_pre, h, z


def loss_grad(P, X, y, C):
    h_pre, h, z = forward(P, X, C)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    mask = y >= 0
    yy = np.where(mask, y, 0)
    n = mask.sum()
    lp = np.log(np.take_along_axis(p, yy[..., None], -1)[..., 0] + 1e-12)
    loss = -(lp * mask).sum() / n
    dz = p.copy()
    np.put_along_axis(dz, yy[..., None], np.take_along_axis(dz, yy[..., None], -1) - 1, -1)
    dz *= mask[..., None] / n
    dz = dz.reshape(len(X), S * C)
    g = {"W2": dz.T @ h, "b2": dz.sum(0)}
    dh = dz @ P["W2"] * (h_pre > 0)
    g["W1"] = dh.T @ X
    g["b1"] = dh.sum(0)
    return loss, g


def train_gradient(protocol, n_train, hidden, epochs, seed, out, lr=1e-3, batch=128):
    proto = PROTOCOLS[protocol]
    cs = proto["charset"]
    C = len(cs) + 1
    spec = proto["anchor"]["spec"]
    X, labels, _ = generate(spec, n_train, seed, "baseline-train")
    X = X.reshape(len(X), -1).astype(np.float64) / 255.0
    y = targets(labels, cs)
    rng = np.random.default_rng(seed)
    P = {"W1": rng.normal(0, np.sqrt(2 / 4096), (hidden, 4096)), "b1": np.zeros(hidden),
         "W2": rng.normal(0, np.sqrt(1 / hidden), (S * C, hidden)), "b2": np.zeros(S * C)}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    t = 0
    log = []
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss, g = loss_grad(P, X[idx], y[idx], C)
            tot += loss * len(idx)
            t += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        log.append({"epoch": ep, "loss": tot / len(X), "elapsed_s": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
    g = to_graph(P["W1"], P["b1"], P["W2"], P["b2"], cs, proto["max_len"],
                 tag=f"gradient slot-MLP H={hidden} n_train={n_train} epochs={epochs} seed={seed}")
    g.save(out)
    (Path(out) / "train_log.json").write_text(json.dumps({"log": log, "n_train": n_train, "hidden": hidden,
                                                          "epochs": epochs, "seed": seed,
                                                          "sgd_steps": t, "example_passes": t * batch}))
    return g


def random_search(protocol, n_dev, hidden, evals, seed, out, sigma=0.05):
    """(1+1) random direct-edit hill climbing scored on a dev sample (slot accuracy)."""
    proto = PROTOCOLS[protocol]
    cs = proto["charset"]
    C = len(cs) + 1
    X, labels, _ = generate(proto["anchor"]["spec"], n_dev, seed + 7, "baseline-randdev")
    X = X.reshape(len(X), -1).astype(np.float64) / 255.0
    y = targets(labels, cs)
    rng = np.random.default_rng(seed)
    P = {"W1": rng.normal(0, np.sqrt(2 / 4096), (hidden, 4096)), "b1": np.zeros(hidden),
         "W2": rng.normal(0, np.sqrt(1 / hidden), (S * C, hidden)), "b2": np.zeros(S * C)}

    def fit(P):
        _, _, z = forward(P, X, C)
        pred = z.argmax(-1)
        mask = y >= 0
        return ((pred == y) & mask).sum() / mask.sum()

    best = fit(P)
    log = []
    for e in range(evals):
        Q = {k: v + rng.normal(0, sigma, v.shape) * (rng.random(v.shape) < 0.01) for k, v in P.items()}
        f = fit(Q)
        if f >= best:
            P, best = Q, f
        if e % max(1, evals // 50) == 0:
            log.append({"eval": e, "slot_acc": float(best)})
            print(json.dumps(log[-1]), flush=True)
    g = to_graph(P["W1"], P["b1"], P["W2"], P["b2"], cs, proto["max_len"],
                 tag=f"random (1+1) edits H={hidden} evals={evals} n_dev={n_dev} seed={seed}")
    g.save(out)
    (Path(out) / "train_log.json").write_text(json.dumps({"log": log, "evals": evals, "n_dev": n_dev,
                                                          "example_passes": evals * n_dev}))
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["gradient", "random"])
    ap.add_argument("--protocol", default="digits-mono/1")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--evals", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.kind == "gradient":
        train_gradient(a.protocol, a.n, a.hidden, a.epochs, a.seed, a.out)
    else:
        random_search(a.protocol, a.n, a.hidden, a.evals, a.seed, a.out)


if __name__ == "__main__":
    main()
