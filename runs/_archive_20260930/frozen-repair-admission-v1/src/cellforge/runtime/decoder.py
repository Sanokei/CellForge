"""Bounded greedy decoding driver.

Inference inputs after the image are only the generated prefix (previous
token one-hot), the step index, and the model's own declared state vector.
Target text and target length never enter the model.
"""
from __future__ import annotations

import numpy as np

from ..ir.graph import Graph, GraphError
from .interp import Interpreter

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def decode(graph: Graph, images: np.ndarray, trace: bool = False, trace_filter=None):
    """Decode a batch of normalized images [B,H,W] -> list of results.

    Returns (predictions, info) where predictions is a list of strings (None
    for failed cases) and info holds per-example failure reasons and exec stats.
    """
    meta = graph.meta
    vocab = meta["vocab"]
    V = len(vocab)
    S = int(meta.get("state_dim", 0))
    max_len = int(meta.get("max_len", 12))
    eos, bos = vocab.index(EOS), vocab.index(BOS)
    B = images.shape[0]
    it = Interpreter(graph, trace=trace, trace_filter=trace_filter)
    enc_out = it.run_bundle(graph.entries["encode"], {"image": images.astype(np.float64)}, "/encode")
    enc_trace = dict(it.trace) if trace else None
    tokens = [[] for _ in range(B)]
    done = np.zeros(B, dtype=bool)
    failure: list[str | None] = [None] * B
    prev = np.zeros((B, V))
    prev[:, bos] = 1.0
    state = np.zeros((B, S))
    step_traces = []
    step_logits = []
    for t in range(max_len):
        active = np.nonzero(~done)[0]
        if active.size == 0:
            break
        inp = {k: (v[active] if v.shape[0] == B and B > 1 else v) for k, v in enc_out.items()}
        inp.update({"step": np.full((active.size, 1), float(t)), "prev": prev[active], "state": state[active]})
        if trace:
            it.trace = {}
        out = it.run_bundle(graph.entries["step"], inp, f"/step{t}")
        if trace:
            step_traces.append({"active": active, "trace": it.trace})
        logits = np.asarray(out["logits"]).reshape(active.size, -1)
        if logits.shape[1] != V:
            raise GraphError(f"step logits width {logits.shape[1]} != vocab {V}")
        new_state = np.asarray(out.get("state", np.zeros((active.size, S)))).reshape(active.size, -1)
        if new_state.shape[1] != S:
            raise GraphError(f"state width {new_state.shape[1]} != declared {S}")
        step_logits.append((active.copy(), logits.copy()))
        bad = ~np.all(np.isfinite(logits), axis=1) | ~np.all(np.isfinite(new_state), axis=1)
        choice = np.argmax(np.where(np.isfinite(logits), logits, -np.inf), axis=1)
        for j, i in enumerate(active):
            if bad[j]:
                failure[i] = "nonfinite"
                done[i] = True
                continue
            c = int(choice[j])
            if c == eos:
                done[i] = True
            else:
                tokens[i].append(c)
        prev[active] = 0.0
        prev[active, choice] = 1.0
        state[active] = np.where(np.isfinite(new_state), new_state, 0.0)
    preds = []
    for i in range(B):
        if failure[i]:
            preds.append(None)
            continue
        if not done[i]:
            failure[i] = "missing_eos"
        s = "".join(vocab[c] if len(vocab[c]) == 1 else "" for c in tokens[i])
        # special tokens other than EOS emitted mid-sequence make the output invalid
        if any(len(vocab[c]) != 1 for c in tokens[i]):
            failure[i] = failure[i] or "special_token"
        preds.append(s if failure[i] is None else None)
    info = {"failure": failure, "exec": it.stats.as_dict(), "raw_tokens": tokens,
            "step_logits": step_logits, "encode_out": enc_out}
    if trace:
        info["encode_trace"] = enc_trace
        info["step_traces"] = step_traces
    return preds, info
