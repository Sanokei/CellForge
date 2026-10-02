"""Small author-visible compatibility probe, not a language benchmark."""
import hashlib
import json
from pathlib import Path

import numpy as np

from cellforge.governance.ledger import Ledger
from cellforge.ir.graph import Graph
from cellforge.runtime.lm import generate, score_texts


def main():
    root = Path("runs/unified-byte")
    child = Graph.load(root / "artifact")
    parent = Graph.load(Path("runs/real1/graphs") / child.meta["parent_hash"])
    raw = Path("data/tinyreal/dev_eval.json").read_bytes()
    texts = [text[:128] for text in json.loads(raw)[:3]]
    assert all(text.isascii() for text in texts)
    results = {}
    for label, graph in (("parent", parent), ("byte_lift", child)):
        result, info = score_texts(graph, texts)
        results[label] = {"model_hash": graph.hash(), "bits_per_token": float(result["bits"].sum() / result["n_tokens"].sum()),
                          "n_tokens": int(result["n_tokens"].sum()), "correct": int(result["correct"].sum()),
                          "failed_documents": int(result["failed"].sum()), "token_unit": info["token_unit"]}
    assert results["parent"]["n_tokens"] == results["byte_lift"]["n_tokens"]
    assert results["parent"]["correct"] == results["byte_lift"]["correct"]
    assert abs(results["parent"]["bits_per_token"] - results["byte_lift"]["bits_per_token"]) < 0.01
    prompts = ["Hello, what is 2 + 2?", "What is your name?", "你好 🦊"]
    generations, info = generate(child, len(prompts), max_len=24, prompts=prompts)
    broad, broad_info = score_texts(child, ["Hello 42!", "café 中文 🦊"])
    assert not np.any(broad["failed"])
    report = {"protocol": "unified-byte-compatibility-probe/1", "scope": "small development smoke; no claim of language improvement",
              "source_dev_sha256": hashlib.sha256(raw).hexdigest(), "sample": "first three development texts, first 128 ASCII characters each",
              "comparison": results, "stress_n_tokens": broad["n_tokens"].tolist(),
              "stress_failures": broad["failed"].tolist(), "stress_token_unit": broad_info["token_unit"],
              "generation": [{"prompt": p, "text": t, "failure": f} for p, t, f in zip(prompts, generations, info["failure"])],
              "decision": "interface artifact retained; repetitive generations show conversational ability unachieved"}
    destination = root / "compatibility-probe.json"
    if destination.exists():
        raise SystemExit("preserve the existing probe receipt")
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    Ledger(root / "ledger.jsonl").append("development_probe", report, "recorder")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
