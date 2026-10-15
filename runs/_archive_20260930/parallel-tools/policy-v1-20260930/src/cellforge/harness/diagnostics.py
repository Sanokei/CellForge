"""Development diagnostics (agent-improvable surface, versioned).

These reports are development feedback.  They reuse the reference metric
function for convenience, but nothing here can write reference results.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..ir.graph import GraphError
from ..reference.metrics import levenshtein, score
from ..runtime.decoder import decode

DIAGNOSTICS_VERSION = "dev-diag/1"


def _heat_rows(vec: np.ndarray, width: int = 128, height: int = 8) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size == 0 or not np.all(np.isfinite(v)):
        return np.zeros((height, width), dtype=np.uint8)
    lo, hi = v.min(), v.max()
    v = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    idx = np.minimum((np.arange(width) * v.size) // width, v.size - 1)
    row = (v[idx] * 255).astype(np.uint8)
    return np.tile(row, (height, 1))


def run(graph, imgs: np.ndarray, labels: list[str], out_dir: str | Path | None = None,
        viz_nodes: list[str] | None = None, trace_nodes: list[str] | None = None,
        max_examples: int = 24) -> dict:
    x = imgs.astype(np.float64) / 255.0
    want = set(viz_nodes or []) | set(trace_nodes or [])
    tf = (lambda p: any(p.endswith("/" + n) or p == n or ("/" + n) in p for n in want)) if want else None
    try:
        preds, info = decode(graph, x, trace=bool(want), trace_filter=tf)
    except (GraphError, ValueError, IndexError) as e:
        return {"version": DIAGNOSTICS_VERSION, "error": str(e)[:800]}
    s = score(preds, labels)
    by_len: dict[int, list[int]] = {}
    pred_len = Counter()
    conf = Counter()
    pos_ok = Counter()
    pos_n = Counter()
    for p, r in zip(preds, labels):
        by_len.setdefault(len(r), [0, 0])
        by_len[len(r)][1] += 1
        by_len[len(r)][0] += int(p == r)
        pred_len[(len(r), -1 if p is None else len(p))] += 1
        if p is not None and len(p) == len(r):
            for i, (a, b) in enumerate(zip(r, p)):
                pos_n[i] += 1
                pos_ok[i] += int(a == b)
                if a != b:
                    conf[f"{a}->{b}"] += 1
    rep = {
        "version": DIAGNOSTICS_VERSION,
        "n": len(labels),
        "exact_match": round(s["exact_match"], 4),
        "cer": round(s["cer"], 4),
        "total_edits": s["total_edits"], "total_ref_chars": s["total_ref_chars"],
        "failures": s["failures"],
        "failure_reasons": dict(Counter(f for f in info["failure"] if f)),
        "exact_by_true_length": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_len.items())},
        "length_error_rate": round(sum(c for (a, b), c in pred_len.items() if a != b) / max(1, len(labels)), 4),
        "top_length_confusions": [f"true{a}->pred{b}: {c}" for (a, b), c in pred_len.most_common(12) if a != b][:8],
        "per_position_char_acc_when_length_right": {i: round(pos_ok[i] / pos_n[i], 3) for i in sorted(pos_n)},
        "top_char_confusions": conf.most_common(15),
        "exec": {k: v for k, v in info["exec"].items() if k != "route_counts"},
        "route_counts": info["exec"]["route_counts"],
    }
    wrong = [i for i, (p, r) in enumerate(zip(preds, labels)) if p != r]
    rep["examples_wrong"] = [{"i": i, "true": labels[i], "pred": preds[i],
                              "edits": len(labels[i]) if preds[i] is None else levenshtein(preds[i], labels[i])}
                             for i in wrong[:max_examples]]
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        show = wrong[:max_examples] if wrong else list(range(min(8, len(labels))))
        for i in show:
            panels = [np.kron(imgs[i], np.ones((2, 2), dtype=np.uint8))]  # 2x upscale original
            if viz_nodes and "step_traces" in info:
                for st_i, st in enumerate(info["step_traces"]):
                    pos = np.nonzero(st["active"] == i)[0]
                    if pos.size == 0:
                        continue
                    for key, arr in st["trace"].items():
                        if any(key.split(".")[0].endswith("/" + n) for n in viz_nodes):
                            v = arr[pos[0]] if arr.shape[0] > 1 else arr[0]
                            panels.append(np.kron(_heat_rows(v), np.ones((1, 2), dtype=np.uint8)))
                            panels.append(np.zeros((2, 256), dtype=np.uint8))
            if viz_nodes and info.get("encode_trace"):
                for key, arr in info["encode_trace"].items():
                    if any(key.split(".")[0].endswith("/" + n) for n in viz_nodes):
                        v = arr[i] if arr.shape[0] > 1 else arr[0]
                        if v.ndim == 2 and v.shape[0] > 1:
                            hm = v.T if v.shape[0] > v.shape[1] else v
                            for r in range(min(hm.shape[0], 16)):
                                panels.append(np.kron(_heat_rows(hm[r], height=4), np.ones((1, 2), dtype=np.uint8)))
                        else:
                            panels.append(np.kron(_heat_rows(v), np.ones((1, 2), dtype=np.uint8)))
            Image.fromarray(np.vstack(panels)).save(out / f"case{i:05d}.png")
        if trace_nodes:
            summ = {}
            src = dict(info.get("encode_trace") or {})
            for st_i, st in enumerate(info.get("step_traces", [])):
                for k, v in st["trace"].items():
                    src[f"step{st_i}:{k}"] = v
            for k, v in src.items():
                if v.size:
                    summ[k] = {"shape": list(v.shape), "mean": float(np.mean(v)), "std": float(np.std(v)),
                               "min": float(np.min(v)), "max": float(np.max(v))}
                    np.save(out / (k.replace("/", "_").replace(":", "_").replace("[", "_").replace("]", "") + ".npy"),
                            v[:max_examples])
            rep["trace_summaries"] = summ
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
