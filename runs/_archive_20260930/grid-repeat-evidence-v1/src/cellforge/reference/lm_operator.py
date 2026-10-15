"""Operator-side final evaluation for tinylm protocols (evaluation server only).

The data-generating grammar lives only in the private directory; it is loaded
from there at run time and never shipped to authoring machines.  Each run draws
a fresh secret seed, generates new documents, scores the artifact, checks the
grammaticality of the artifact's own free-running generations, and publishes
aggregates only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import secrets
import time
from pathlib import Path

import numpy as np

from ..ir.graph import Graph
from .lm_eval import evaluate_lm, generation_sample, sentences
from .protocols import PROTOCOLS


def load_grammar(private: Path):
    spec = importlib.util.spec_from_file_location("tinygrammar", private / "grammar" / "tinygrammar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--protocol", default="tinylm/1")
    ap.add_argument("--private", required=True)
    ap.add_argument("--public", required=True)
    ap.add_argument("--dev-train", required=True, help="author-visible training text, for overlap reporting")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    proto = PROTOCOLS[a.protocol]
    priv = Path(a.private)
    gram = load_grammar(priv)
    g = Graph.load(a.artifact)
    eval_id = time.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(4)
    seed = secrets.randbits(63)
    r = random.Random(seed)
    docs = [gram.document(r, 150) for _ in range(proto["final_n_docs"])]
    d = priv / "final_lm" / eval_id
    d.mkdir(parents=True)
    (d / "SEED").write_text(str(seed))
    (d / "docs.json").write_text(json.dumps(docs))
    res = evaluate_lm(g, docs)
    per_bits = np.array(res.pop("per_doc_bits"))
    per_n = np.array(res.pop("per_doc_tokens"))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(docs), size=(2000, len(docs)))
    boot = per_bits[idx].sum(1) / per_n[idx].sum(1)
    dev_sents = set(sentences(json.loads(Path(a.dev_train).read_text())))
    fin_sents = sentences(docs)
    gs = proto["final_gen"]
    gen = generation_sample(g, n=gs["n"], temperature=gs["temperature"], seed=seed % (2**32),
                            max_len=proto["gen_max_len"])
    samp = sentences(gen["sampled"])
    greedy_s = sentences(gen["greedy"])
    gram_ok = [gram.grammatical(s) for s in samp]
    novel_ok = [s for s, ok in zip(samp, gram_ok) if ok and s not in dev_sents]
    (d / "generations.json").write_text(json.dumps(gen))
    out = {
        "eval_id": eval_id, "label": a.label, "protocol": a.protocol, "model_hash": g.hash(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scoring": {k: res[k] for k in ("n_docs", "n_tokens", "bpc", "top1", "failures", "latency_s_total",
                                        "macs_total")} | {
            "bpc_bootstrap95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
            "route_counts": res["route_counts"]},
        "overlap": {"final_sentences": len(fin_sents),
                    "final_sentences_seen_in_dev_train": round(sum(s in dev_sents for s in fin_sents) / max(1, len(fin_sents)), 4),
                    "note": "finite grammar: short sentences recur; bpc is over new documents"},
        "generation": {"n_samples": gs["n"], "temperature": gs["temperature"],
                       "complete_sentences": len(samp),
                       "grammatical_rate": round(sum(gram_ok) / max(1, len(samp)), 4),
                       "grammatical_and_novel_rate": round(len(novel_ok) / max(1, len(samp)), 4),
                       "distinct_rate": round(len(set(samp)) / max(1, len(samp)), 4),
                       "greedy_grammatical": [gram.grammatical(s) for s in greedy_s[:8]],
                       "unterminated_or_failed": len([f for f in gen["failures"]]),
                       "mean_words_per_sentence": round(float(np.mean([len(re.findall(r'[a-z]+', s)) for s in samp])) if samp else 0.0, 2)},
    }
    pub = Path(a.public)
    with (pub / "final_results_lm.jsonl").open("a") as f:
        f.write(json.dumps(out, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
