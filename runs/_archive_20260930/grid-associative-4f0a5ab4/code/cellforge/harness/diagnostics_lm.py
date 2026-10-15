"""Development diagnostics for character language models (agent-improvable)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..ir.graph import GraphError
from ..reference.lm_eval import generation_sample, sentences
from ..runtime.lm import score_texts

DIAGNOSTICS_VERSION = "dev-diag-lm/1"


def _context_class(prev: str, prev2: str) -> str:
    if prev == "<bos>":
        return "doc_start"
    if prev == ".":
        return "after_period"
    if prev == ",":
        return "after_comma"
    if prev == " ":
        return "after_period_space" if prev2 == "." else "word_start"
    return "in_word"


def run(graph, texts: list[str], lexicon: set[str], out_dir=None, n_gen: int = 16, seed: int = 0) -> dict:
    vocab = graph.meta["vocab"]
    try:
        r, info = score_texts(graph, texts)
    except (GraphError, ValueError, IndexError) as e:
        return {"version": DIAGNOSTICS_VERSION, "error": str(e)[:800]}
    ppb = info["per_pos_bits"]
    tok = info["tokens"]
    L = info["lengths"]
    cls_bits: Counter = Counter()
    cls_n: Counter = Counter()
    conf: Counter = Counter()
    worst = []
    for i in range(len(texts)):
        for t in range(L[i] - 1):
            prev = vocab[tok[i, t]]
            prev2 = vocab[tok[i, t - 1]] if t > 0 else ""
            c = _context_class(prev, prev2)
            cls_bits[c] += ppb[i, t]
            cls_n[c] += 1
            worst.append((ppb[i, t], i, t))
    worst.sort(reverse=True)
    n_tok = int(r["n_tokens"].sum())
    rep = {
        "version": DIAGNOSTICS_VERSION,
        "n_docs": len(texts), "n_tokens": n_tok,
        "bpc": round(float(r["bits"].sum()) / max(1, n_tok), 4),
        "top1": round(float(r["correct"].sum()) / max(1, n_tok), 4),
        "failures": int(r["failed"].sum()),
        "uniform_bpc_reference": round(float(np.log2(len(vocab))), 4),
        "bits_by_context": {c: {"bpc": round(cls_bits[c] / cls_n[c], 3), "n": cls_n[c],
                                "share_of_total_bits": round(cls_bits[c] / max(1e-9, sum(cls_bits.values())), 3)}
                            for c in sorted(cls_n)},
        "worst_positions": [{"doc": i, "pos": t, "bits": round(float(b), 2),
                             "context": texts[i][max(0, t - 30):t], "target": (texts[i] + "⏎")[t]}
                            for b, i, t in worst[:12]],
        "route_counts": info["exec"]["route_counts"],
        "exec": {k: v for k, v in info["exec"].items() if k != "route_counts"},
    }
    try:
        gen = generation_sample(graph, n=n_gen, seed=seed)
        sents = sentences(gen["sampled"])
        words = re.findall(r"[a-z]+", " ".join(gen["sampled"]))
        rep["generation"] = {
            "greedy": [g[:200] for g in gen["greedy"][:3]],
            "sampled_t0.8": [g[:200] for g in gen["sampled"][:6]],
            "complete_sentences": len(sents),
            "distinct_sentence_rate": round(len(set(sents)) / max(1, len(sents)), 3),
            "in_lexicon_word_rate": round(sum(w in lexicon for w in words) / max(1, len(words)), 3),
            "failures": gen["failures"][:5],
            "note": "grammaticality is measured only by the private final evaluator",
        }
    except (GraphError, ValueError, IndexError) as e:
        rep["generation"] = {"error": str(e)[:300]}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
