"""Multiple-choice benchmark evaluation of character-level models (server only).

Benchmarks: MMLU (cais/mmlu test), ARC-Challenge test, HellaSwag validation.
Data live only in the private directory; aggregates are published.

Scoring (protocol ``mc-loglik/1``, lm-evaluation-harness style):
  * item text is normalized to the artifact's own character vocabulary
    (lowercase; curly quotes straightened; characters outside the vocabulary
    are removed; runs of spaces collapsed).  ``coverage`` reports the fraction
    of original characters kept.  Digits and capitals are therefore lost for
    the current lowercase vocabularies; this is a real limitation, reported.
  * for every choice the model scores ``prompt + " " + choice``; the score is
    the total bits over the choice characters only.
  * ``acc``: choice with fewest bits; ``acc_norm``: fewest bits per character.
  * a fixed random subsample of ``n`` items per benchmark (seeded once, stored
    in bench_subsets.json) is used so models are compared on identical items.
Chance: 25% for all three.  95% intervals are Wilson intervals over items.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from ..ir.graph import Graph
from ..runtime.lm import encode_text, score_texts
from ..runtime.text_codec import BYTE_CODEC, BYTE_VOCAB

PROTOCOL = "mc-loglik/1"
LETTERS = "abcd"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((c - r) / d, 4), round((c + r) / d, 4)]


def normalizer(vocab, codec=None):
    if codec == BYTE_CODEC:
        if vocab != BYTE_VOCAB:
            raise ValueError("byte benchmark requires canonical vocabulary")
        return lambda text: (text, 1.0)
    if codec is not None:
        raise ValueError(f"unsupported benchmark text codec: {codec}")
    keep = set(c for c in vocab if len(c) == 1)

    def norm(s: str):
        s0 = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        s0 = re.sub(r"\s+", " ", s0)
        out = "".join(c for c in s0 if c in keep)
        out = re.sub(r" +", " ", out).strip()
        return out, (len(out) / max(1, len(s.strip())))
    return norm


def load_items(bench_dir: Path, name: str):
    import pyarrow.parquet as pq
    if name == "mmlu":
        t = pq.read_table(bench_dir / "mmlu_test.parquet").to_pylist()
        return [{"prompt": r["question"], "choices": list(r["choices"]), "label": int(r["answer"]),
                 "group": r["subject"]} for r in t]
    if name == "arc_challenge":
        t = pq.read_table(bench_dir / "arc_c_test.parquet").to_pylist()
        out = []
        for r in t:
            labels = list(r["choices"]["label"])
            if r["answerKey"] not in labels or len(labels) != 4:
                continue
            out.append({"prompt": r["question"], "choices": list(r["choices"]["text"]),
                        "label": labels.index(r["answerKey"]), "group": "arc"})
        return out
    if name == "hellaswag":
        t = pq.read_table(bench_dir / "hellaswag_val.parquet").to_pylist()
        return [{"prompt": r["ctx"], "choices": [re.sub(r"\[.*?\]", " ", e) for e in r["endings"]],
                 "label": int(r["label"]), "group": r["activity_label"]} for r in t]
    raise ValueError(name)


def subset(items, n, seed_path: Path, name: str):
    """First n items of a fixed seeded permutation (stored once), so any n gives
    a nested, identical-across-models sample."""
    store = json.loads(seed_path.read_text()) if seed_path.exists() else {}
    key = name + "/perm/1"
    if key not in store:
        rng = np.random.default_rng(20260929 + len(name))
        store[key] = rng.permutation(len(items)).tolist()
        seed_path.write_text(json.dumps(store))
    return [items[i] for i in store[key][:n]]


def score_choices(graph: Graph, prompts, conts, chunk=48):
    """Return per-(prompt,cont) total bits over cont chars. Inputs are normalized strings."""
    texts = [p + " " + c for p, c in zip(prompts, conts)]
    byte_mode = graph.meta.get("text_codec") == BYTE_CODEC
    starts = [(len(p.encode("utf-8")) if byte_mode else len(p)) + 1 for p in prompts]
    bits = np.zeros(len(texts))
    for i in range(0, len(texts), chunk):
        batch = texts[i:i + chunk]
        r, info = score_texts(graph, batch)
        ppb = info["per_pos_bits"]
        for j, t in enumerate(batch):
            s, e = starts[i + j], len(t.encode("utf-8")) if byte_mode else len(t)
            bits[i + j] = ppb[j, s:e].sum() if e > s else np.inf
        if r["failed"].any():
            for j in np.nonzero(r["failed"])[0]:
                bits[i + j] = np.inf
    return bits


def eval_bench(graph: Graph, items, max_prompt_chars=600):
    vocab = graph.meta["vocab"]
    byte_mode = graph.meta.get("text_codec") == BYTE_CODEC
    norm = normalizer(vocab, graph.meta.get("text_codec"))
    prompts, conts, meta = [], [], []
    covs = []
    for ii, it in enumerate(items):
        p, cp = norm(it["prompt"])
        p = p[-max_prompt_chars:]
        covs.append(cp)
        for ci, ch in enumerate(it["choices"]):
            c, cc = norm(ch)
            covs.append(cc)
            prompts.append(p)
            conts.append(c)
            meta.append((ii, ci, len(c.encode("utf-8")) if byte_mode else len(c)))
    bits = score_choices(graph, prompts, conts)
    correct = correct_n = 0
    per_group: dict[str, list[int]] = {}
    for ii, it in enumerate(items):
        rows = [k for k, m in enumerate(meta) if m[0] == ii]
        b = np.array([bits[k] for k in rows])
        L = np.array([max(1, meta[k][2]) for k in rows])
        a = int(np.argmin(b))
        an = int(np.argmin(b / L))
        correct += a == it["label"]
        correct_n += an == it["label"]
        g = per_group.setdefault(it["group"], [0, 0])
        g[0] += an == it["label"]
        g[1] += 1
    n = len(items)
    return {"n": n, "acc": round(correct / n, 4), "acc_ci95": wilson(correct, n),
            "acc_norm": round(correct_n / n, 4), "acc_norm_ci95": wilson(correct_n, n),
            "chance": 0.25, "char_coverage": round(float(np.mean(covs)), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--private", required=True)
    ap.add_argument("--public", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--benches", default="mmlu,arc_challenge,hellaswag")
    ap.add_argument("--split", default="all", choices=["all", "sel", "sealed"],
                    help="sel = even positions of the stored subset (selection signal for breeding); "
                         "sealed = odd positions (never used for selection; final reporting only)")
    a = ap.parse_args(argv)
    g = Graph.load(a.artifact)
    bdir = Path(a.private) / "bench"
    out = {"protocol": "mc-byte-loglik/1" if g.meta.get("text_codec") == BYTE_CODEC else PROTOCOL,
           "label": a.label, "model_hash": g.hash(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "split": a.split, "results": {}}
    for name in a.benches.split(","):
        t0 = time.time()
        items = subset(load_items(bdir, name), a.n, bdir / "bench_subsets.json", name)
        if a.split == "sel":
            items = items[0::2]
        elif a.split == "sealed":
            items = items[1::2]
        res = eval_bench(g, items)
        res["seconds"] = round(time.time() - t0, 1)
        out["results"][name] = res
    with (Path(a.public) / "bench_results.jsonl").open("a") as f:
        f.write(json.dumps(out, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
