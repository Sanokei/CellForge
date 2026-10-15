"""Character language-model driver (protocol family ``tinylm``).

The model is a single ``step`` bundle called once per position with:
  ctx   [K,V]  one-hot of the last K tokens (oldest first, most recent last);
               positions before the sequence start are all-zero rows
  prev  [V]    one-hot of the most recent token (== ctx[K-1])
  step  [1]    position index (0 = predicting the first character)
  state [S]    the model's own recurrent state (zeros at position 0)
and must return ``logits`` [V] and ``state`` [S].  Only the generated or
given prefix enters the model; there is no lookahead.

Scoring (teacher forcing): tokens = <bos> + text + <eos>; every token after
<bos> is predicted.  Non-finite logits at a position make that document a
recorded failure; its positions are charged FAILURE_BITS each.
"""
from __future__ import annotations

import numpy as np

from ..ir.graph import Graph, GraphError
from .decoder import BOS, EOS, PAD
from .interp import Interpreter
from .text_codec import BYTE_CODEC, encode_content, decode_content

FAILURE_BITS = 8.0


def encode_text(vocab: list[str], text: str, codec=None) -> list[int]:
    idx = {c: i for i, c in enumerate(vocab)}
    return [idx[BOS]] + encode_content(vocab, text, codec) + [idx[EOS]]


def _log2_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return (z - np.log(np.exp(z).sum(axis=1, keepdims=True))) / np.log(2)


def score_texts(graph: Graph, texts: list[str], collect_routes: bool = False):
    """Teacher-forced scoring. Returns per-doc dict with bits, n_tokens, correct, failed."""
    vocab = graph.meta["vocab"]
    V = len(vocab)
    K = int(graph.meta.get("ctx_len", 8))
    S = int(graph.meta.get("state_dim", 0))
    seqs = [encode_text(vocab, t, graph.meta.get("text_codec")) for t in texts]
    B = len(seqs)
    T = max(len(s) for s in seqs)
    tok = np.zeros((B, T), dtype=np.int64)
    L = np.array([len(s) for s in seqs])
    for i, s in enumerate(seqs):
        tok[i, :len(s)] = s
    ctx = np.zeros((B, K, V))
    state = np.zeros((B, S))
    bits = np.zeros(B)
    correct = np.zeros(B, dtype=np.int64)
    failed = np.zeros(B, dtype=bool)
    it = Interpreter(graph)
    step_def = graph.entries["step"]
    per_pos_bits = np.zeros((B, T))
    for t in range(T - 1):
        rows = np.nonzero(L - 1 > t)[0]
        if rows.size == 0:
            break
        cur = tok[rows, t]
        ctx[rows] = np.roll(ctx[rows], -1, axis=1)
        ctx[rows, K - 1] = 0.0
        ctx[rows, K - 1, cur] = 1.0
        inp = {"ctx": ctx[rows], "prev": ctx[rows, K - 1], "step": np.full((rows.size, 1), float(t)),
               "state": state[rows]}
        out = it.run_bundle(step_def, inp, "/step")
        lg = np.asarray(out["logits"]).reshape(rows.size, -1)
        if lg.shape[1] != V:
            raise GraphError(f"logits width {lg.shape[1]} != vocab {V}")
        ns = np.asarray(out.get("state", np.zeros((rows.size, S)))).reshape(rows.size, -1)
        if ns.shape[1] != S:
            raise GraphError(f"state width {ns.shape[1]} != declared {S}")
        bad = ~np.all(np.isfinite(lg), axis=1) | ~np.all(np.isfinite(ns), axis=1)
        failed[rows[bad]] = True
        tgt = tok[rows, t + 1]
        lp = _log2_softmax(np.where(np.isfinite(lg), lg, 0.0))
        b = -lp[np.arange(rows.size), tgt]
        per_pos_bits[rows, t] = b
        bits[rows] += b
        correct[rows] += (np.argmax(np.where(np.isfinite(lg), lg, -np.inf), axis=1) == tgt)
        state[rows] = np.where(np.isfinite(ns), ns, 0.0)
    n_tok = L - 1
    failure_bits = max(FAILURE_BITS, float(np.log2(V) + 1)) if graph.meta.get("text_codec") == BYTE_CODEC else FAILURE_BITS
    bits = np.where(failed, failure_bits * n_tok, bits)
    if graph.meta.get("text_codec") == BYTE_CODEC:
        correct = np.where(failed, 0, correct)
    info = {"exec": it.stats.as_dict(), "per_pos_bits": per_pos_bits, "tokens": tok, "lengths": L,
            "token_unit": "utf8_byte" if graph.meta.get("text_codec") == BYTE_CODEC else "character",
            "failure_bits_per_token": failure_bits}
    return {"bits": bits, "n_tokens": n_tok, "correct": correct, "failed": failed}, info


def generate(graph: Graph, n: int, max_len: int = 200, temperature: float = 0.0, seed: int = 0,
             prompts: list[str] | None = None, *, backend: str = "reference",
             state_only_prefill: bool = False):
    """Free-running generation. temperature 0 = greedy. Sampling randomness lives in
    the driver (seeded), not in the model."""
    if backend not in ("reference", "compiled"):
        raise ValueError("backend must be reference or compiled")
    if state_only_prefill and backend != "compiled":
        raise ValueError("state-only prefill requires the compiled backend")
    vocab = graph.meta["vocab"]
    V = len(vocab)
    K = int(graph.meta.get("ctx_len", 8))
    S = int(graph.meta.get("state_dim", 0))
    rng = np.random.default_rng(seed)
    eos, bos = vocab.index(EOS), vocab.index(BOS)
    prompts = prompts or [""] * n
    if len(prompts) != n:
        raise ValueError("one prompt is required per generated sequence")
    ctx = np.zeros((n, K, V))
    state = np.zeros((n, S))
    out_tokens: list[list[int]] = [[] for _ in range(n)]
    done = np.zeros(n, dtype=bool)
    failure = [None] * n
    if backend == "compiled":
        from .compiled import CompiledInterpreter
        it = CompiledInterpreter(graph)
        it.assert_frozen()
    else:
        it = Interpreter(graph)
    feed = [[bos] + encode_content(vocab, p, graph.meta.get("text_codec")) for p in prompts]
    total = max(len(f) for f in feed) + max_len
    for t in range(total):
        rows = np.nonzero(~done)[0]
        if rows.size == 0:
            break
        cur = np.array([feed[i][t] if t < len(feed[i]) else out_tokens[i][-1] for i in rows])
        ctx[rows] = np.roll(ctx[rows], -1, axis=1)
        ctx[rows, K - 1] = 0.0
        ctx[rows, K - 1, cur] = 1.0
        inputs = {"ctx": ctx[rows], "prev": ctx[rows, K - 1],
                  "step": np.full((rows.size, 1), float(t)), "state": state[rows]}
        # Demand only state when every active row still consumes prompt bytes.
        # Mixed prompt/generation batches retain full eager execution. All state
        # dependencies execute; teacher-forced scoring is unchanged.
        demand_state = (state_only_prefill and all(t + 1 < len(feed[i]) for i in rows)
                        and "state" in graph.defs[graph.entries["step"]]["outputs"])
        if demand_state:
            out = it.run_bundle_outputs(graph.entries["step"], inputs, ["state"], "/gen")
        else:
            out = it.run_bundle(graph.entries["step"], inputs, "/gen")
        lg = None if demand_state else np.asarray(out["logits"]).reshape(rows.size, -1)
        ns = np.asarray(out.get("state", np.zeros((rows.size, S)))).reshape(rows.size, -1)
        state[rows] = np.where(np.isfinite(ns), ns, 0.0)
        for j, i in enumerate(rows):
            if t + 1 < len(feed[i]):
                continue  # still consuming the prompt
            row = lg[j]
            if not np.all(np.isfinite(row)):
                failure[i] = "nonfinite"
                done[i] = True
                continue
            row = row.copy()
            row[[vocab.index(PAD), bos]] = -np.inf
            if temperature <= 0:
                c = int(np.argmax(row))
            else:
                z = (row - row.max()) / temperature
                p = np.exp(z)
                p /= p.sum()
                c = int(rng.choice(V, p=p))
            if c == eos:
                done[i] = True
            else:
                out_tokens[i].append(c)
                if len(out_tokens[i]) >= max_len:
                    done[i] = True
                    failure[i] = failure[i] or "max_len"
    texts = []
    for i, tokens in enumerate(out_tokens):
        try:
            texts.append(decode_content(vocab, tokens, graph.meta.get("text_codec")))
        except UnicodeDecodeError:
            # Preserve malformed byte output as explicit evidence; the driver
            # does not repair the model's tokens into apparently valid text.
            texts.append(None)
            failure[i] = "invalid_utf8"
    return texts, {"failure": failure, "exec": it.stats.as_dict(), "token_ids": out_tokens,
                   "text_codec": graph.meta.get("text_codec", "characters"),
                   "backend": backend, "state_only_prefill": state_only_prefill,
                   "counter_scope": "demanded closure semantic operations" if state_only_prefill else "full eager semantic operations"}
