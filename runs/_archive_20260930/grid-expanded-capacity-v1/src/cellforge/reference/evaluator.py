"""Reference evaluation of an exported graph on a dataset directory."""
from __future__ import annotations

import time

import numpy as np

from ..data.render import load_dataset
from ..ir.graph import Graph, GraphError, sha256_bytes
from ..runtime.decoder import decode
from . import metrics

EVALUATOR_VERSION = "ref-eval/1"


def predict_all(graph: Graph, imgs: np.ndarray, chunk: int = 256):
    preds, fails, execs = [], [], []
    t0 = time.perf_counter()
    for i in range(0, len(imgs), chunk):
        x = imgs[i:i + chunk].astype(np.float64) / 255.0
        try:
            p, info = decode(graph, x)
            fails.extend(info["failure"])
            execs.append(info["exec"])
        except (GraphError, FloatingPointError, MemoryError) as e:
            p = [None] * len(x)
            fails.extend([f"graph_error: {e}"[:200]] * len(x))
        preds.extend(p)
    dt = time.perf_counter() - t0
    return preds, fails, execs, dt


def evaluate(graph: Graph, dataset_dir, keep_per_case: bool = True) -> dict:
    imgs, labels, manifest = load_dataset(dataset_dir)
    preds, fails, execs, dt = predict_all(graph, imgs)
    res = metrics.score(preds, labels)
    reasons: dict[str, int] = {}
    for f in fails:
        if f:
            k = f.split(":")[0]
            reasons[k] = reasons.get(k, 0) + 1
    res.update({
        "evaluator": EVALUATOR_VERSION,
        "model_hash": graph.hash(),
        "dataset_manifest_sha256": sha256_bytes(open(f"{dataset_dir}/manifest.json", "rb").read()),
        "dataset_split": manifest["split"],
        "failure_reasons": reasons,
        "latency_s_total": round(dt, 4),
        "latency_ms_per_case": round(1000 * dt / max(1, len(labels)), 4),
        "macs_total": int(sum(e["macs"] for e in execs)),
    })
    if keep_per_case:
        res["predictions"] = preds
    else:
        res.pop("per_case_edits", None)
    return res
