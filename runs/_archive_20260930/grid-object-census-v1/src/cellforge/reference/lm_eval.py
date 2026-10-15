"""Reference evaluation for the tinylm protocol family (protected surface).

Protocol ``lm-metrics/1``:
* bpc = total bits over all predicted tokens (characters + final <eos>)
  / number of predicted tokens.  Failed documents are charged FAILURE_BITS
  per token and counted in ``failures``.
* top1 = fraction of predicted tokens whose argmax equals the target.
* Generation metrics (final evaluation only, on the evaluation server):
  greedy and temperature-0.8 samples from <bos>; sentences are split on '.';
  ``grammatical_rate`` = grammatical complete sentences / complete sentences,
  checked by the private data-generating grammar; ``distinct_rate`` =
  distinct sentences / sentences.
"""
from __future__ import annotations

import time

import numpy as np

from ..ir.graph import Graph, GraphError
from ..runtime.lm import FAILURE_BITS, generate, score_texts

PROTOCOL = "lm-metrics/1"


def evaluate_lm(graph: Graph, texts: list[str], chunk: int = 128) -> dict:
    bits = 0.0
    n = 0
    correct = 0
    failures = 0
    per_doc_bits, per_doc_n = [], []
    t0 = time.perf_counter()
    macs = 0
    routes: dict = {}
    for i in range(0, len(texts), chunk):
        batch = texts[i:i + chunk]
        try:
            r, info = score_texts(graph, batch)
            macs += info["exec"]["macs"]
            for p, c in info["exec"]["route_counts"].items():
                for b, v in c.items():
                    routes.setdefault(p, {}).setdefault(b, 0)
                    routes[p][b] += v
        except (GraphError, ValueError, IndexError, FloatingPointError) as e:
            nt = np.array([len(t) + 1 for t in batch])
            r = {"bits": FAILURE_BITS * nt, "n_tokens": nt, "correct": np.zeros(len(batch), int),
                 "failed": np.ones(len(batch), bool)}
        bits += float(r["bits"].sum())
        n += int(r["n_tokens"].sum())
        correct += int(r["correct"].sum())
        failures += int(r["failed"].sum())
        per_doc_bits += r["bits"].tolist()
        per_doc_n += r["n_tokens"].tolist()
    return {"protocol": PROTOCOL, "n_docs": len(texts), "n_tokens": n, "total_bits": round(bits, 4),
            "bpc": bits / max(1, n), "top1": correct / max(1, n), "failures": failures,
            "latency_s_total": round(time.perf_counter() - t0, 3), "macs_total": int(macs),
            "route_counts": routes, "per_doc_bits": per_doc_bits, "per_doc_tokens": per_doc_n}


def generation_sample(graph: Graph, n: int = 64, temperature: float = 0.8, seed: int = 0, max_len: int = 200):
    greedy, gi = generate(graph, min(n, 8), max_len=max_len, temperature=0.0, seed=seed)
    sampled, si = generate(graph, n, max_len=max_len, temperature=temperature, seed=seed)
    return {"greedy": greedy, "sampled": sampled, "failures": [f for f in si["failure"] if f]}


def sentences(texts: list[str]) -> list[str]:
    out = []
    for t in texts:
        parts = t.split(".")
        for p in parts[:-1]:  # only sentences terminated by '.'
            p = p.strip()
            if p:
                out.append(p + ".")
    return out
