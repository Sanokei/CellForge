"""Operator-side final evaluation (runs on the evaluation server only).

Invoked as the ``evaluator`` OS user after an artifact is frozen::

    python -m cellforge.reference.operator --artifact <dir> --protocol digits-mono/1 \
        --private /srv/cellforge-private --public /srv/cellforge-public \
        --exclude-strings <dev_strings.txt>

Each invocation draws a fresh secret seed from the OS, renders the final
splits into the private directory, scores the artifact with the reference
evaluator, and publishes only aggregate metrics (no images, labels, seeds or
per-case outcomes) to the public results file.  Because every evaluation uses
new cases, a final result is never computed on previously exposed cases.
Author agents do not run on this machine; its private directory is mode 0700
and owned by ``evaluator``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from ..data.render import generate, save_dataset
from ..ir.graph import Graph
from .evaluator import evaluate
from .protocols import PROTOCOLS


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - r) / d, (c + r) / d)


def bootstrap_cer_ci(per_edits, per_chars, iters=2000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    e = np.asarray(per_edits, dtype=float)
    c = np.asarray(per_chars, dtype=float)
    n = len(e)
    idx = rng.integers(0, n, size=(iters, n))
    vals = e[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--private", required=True)
    ap.add_argument("--public", required=True)
    ap.add_argument("--exclude-strings")
    ap.add_argument("--label", default="")
    ap.add_argument("--splits", default="")
    a = ap.parse_args(argv)
    proto = PROTOCOLS[a.protocol]
    g = Graph.load(a.artifact)
    exclude = []
    if a.exclude_strings:
        exclude = [s for s in Path(a.exclude_strings).read_text().split() if len(s) >= 4]
    priv = Path(a.private)
    eval_id = time.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(4)
    out = {"eval_id": eval_id, "label": a.label, "protocol": a.protocol, "model_hash": g.hash(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "overlap_policy": "exact-string overlap with author-visible dev/anchor strings excluded for length>=4; "
                             "short strings (len<=3) cannot be made disjoint and are reported separately",
           "splits": {}}
    wanted = [s for s in a.splits.split(",") if s] or list(proto["final"])
    for split in wanted:
        spec = proto["final"][split]
        spec.exclude_strings = exclude
        seed = secrets.randbits(63)
        n = proto["final_n"][split]
        imgs, labels, man = generate(spec, n, seed, f"final:{split}")
        d = priv / "final" / eval_id / split
        save_dataset(d, imgs, labels, man)
        (d / "SEED").write_text(str(seed))
        res = evaluate(g, d, keep_per_case=True)
        per = res.pop("per_case_edits")
        preds = res.pop("predictions")
        (d / "per_case.json").write_text(json.dumps({"edits": per, "preds": preds}))
        lo, hi = wilson(res["exact_count"], res["n_cases"])
        clo, chi = bootstrap_cer_ci(per, [len(s) for s in labels])
        short = [i for i, s in enumerate(labels) if len(s) <= 3]
        long_ = [i for i, s in enumerate(labels) if len(s) >= 4]
        sub = lambda ids: {"n": len(ids), "exact": sum(preds[i] == labels[i] for i in ids) / max(1, len(ids)),
                           "cer": sum(per[i] for i in ids) / max(1, sum(len(labels[i]) for i in ids))}
        out["splits"][split] = {k: res[k] for k in ("n_cases", "exact_match", "exact_count", "cer", "total_edits",
                                                     "total_ref_chars", "failures", "failure_reasons",
                                                     "latency_ms_per_case", "dataset_manifest_sha256")} | {
            "exact_match_wilson95": [round(lo, 4), round(hi, 4)],
            "cer_bootstrap95": [round(clo, 4), round(chi, 4)],
            "len_le3": sub(short), "len_ge4_disjoint_strings": sub(long_)}
    pub = Path(a.public)
    pub.mkdir(parents=True, exist_ok=True)
    with (pub / "final_results.jsonl").open("a") as f:
        f.write(json.dumps(out, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
