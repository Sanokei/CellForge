"""Operator-side final evaluation for tinyreal protocols (evaluation server only).

Each evaluation draws fresh, previously unused stories from the private
final pool (usage tracked in ``used.json`` so no story is ever scored twice
unless the pool is exhausted, which raises).  Only aggregates leave the machine.
"""
from __future__ import annotations

import argparse
import json
import re
import secrets
import time
from pathlib import Path

import numpy as np

from ..ir.graph import Graph
from .lm_eval import evaluate_lm, generation_sample
from .protocols import PROTOCOLS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--protocol", default="tinyreal/1")
    ap.add_argument("--private", required=True)
    ap.add_argument("--public", required=True)
    ap.add_argument("--dev-train", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    proto = PROTOCOLS[a.protocol]
    priv = Path(a.private) / "realtext"
    pool = json.loads((priv / "final_pool.json").read_text())
    used_p = priv / "used.json"
    used = set(json.loads(used_p.read_text())) if used_p.exists() else set()
    free = [i for i in range(len(pool)) if i not in used]
    n = proto["final_n_docs"]
    if len(free) < n:
        raise SystemExit(f"final pool exhausted ({len(free)} unused < {n}); needs new held-out data")
    rng = np.random.default_rng(secrets.randbits(63))
    pick = sorted(rng.choice(free, size=n, replace=False).tolist())
    used_p.write_text(json.dumps(sorted(used | set(pick))))
    docs = [pool[i] for i in pick]
    g = Graph.load(a.artifact)
    eval_id = time.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(4)
    res = evaluate_lm(g, docs)
    per_bits = np.array(res.pop("per_doc_bits"))
    per_n = np.array(res.pop("per_doc_tokens"))
    idx = np.random.default_rng(0).integers(0, n, size=(2000, n))
    boot = per_bits[idx].sum(1) / per_n[idx].sum(1)
    train = json.loads(Path(a.dev_train).read_text())
    lex = set(re.findall(r"[a-z']+", " ".join(train)))
    gs = proto["final_gen"]
    gen = generation_sample(g, n=gs["n"], temperature=gs["temperature"], seed=secrets.randbits(31),
                            max_len=proto["gen_max_len"])
    words = re.findall(r"[a-z']+", " ".join(gen["sampled"]))
    sents = [s.strip() for t in gen["sampled"] for s in re.split(r"(?<=[.!?])\s", t) if s.strip()]
    train_sents = set(s.strip() for t in train for s in re.split(r"(?<=[.!?])\s", t) if s.strip())
    out = {"eval_id": eval_id, "label": a.label, "protocol": a.protocol, "model_hash": g.hash(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "scoring": {k: res[k] for k in ("n_docs", "n_tokens", "bpc", "top1", "failures", "latency_s_total",
                                           "macs_total")} | {
               "bpc_bootstrap95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
               "route_counts": res["route_counts"]},
           "pool_remaining": len(free) - n,
           "generation": {"n_samples": gs["n"], "temperature": gs["temperature"],
                          "in_lexicon_word_rate": round(sum(w in lex for w in words) / max(1, len(words)), 4),
                          "distinct_sentence_rate": round(len(set(sents)) / max(1, len(sents)), 4),
                          "copied_train_sentence_rate": round(sum(s in train_sents for s in sents) / max(1, len(sents)), 4),
                          "mean_words": round(len(words) / gs["n"], 1),
                          "failures": len(gen["failures"]),
                          "samples_for_inspection": [t[:240] for t in gen["sampled"][:3]]}}
    with (Path(a.public) / "final_results_real.jsonl").open("a") as f:
        f.write(json.dumps(out, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
