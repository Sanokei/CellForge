"""Author a graph-native long-context repetition feature for the unified MoE.

Each historical three-token prefix is compared with the current three-token
prefix. Tokens that followed that prefix at least twice receive an authored
logit penalty. This is a context feature, not corpus fitting or a Python sampler
override. The same graph logits are used in scoring and generation.
"""
import argparse
import hashlib
import json
from pathlib import Path

from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.runtime.text_codec import BYTE_CODEC


def add_memory(parent, strength=2.0):
    if parent.meta.get("text_codec") != BYTE_CODEC or parent.meta.get("ctx_len", 0) < 8:
        raise ValueError("requires a full-byte language MoE with at least eight context positions")
    if not 0 < strength <= 20:
        raise ValueError("strength must be in (0,20]")
    graph = parent.copy()
    length, width = graph.meta["ctx_len"], len(graph.meta["vocab"])
    history = length - 3
    nodes = []
    for i in range(3):
        nodes.extend([
            {"id": f"s{i}", "use": "op:slice", "in": {"x": "$ctx"},
             "args": {"axis": 0, "start": i, "stop": i + history}},
            {"id": f"e{i}", "use": "op:expand", "in": {"x": f"s{i}"}, "args": {"axis": 1}}])
    nodes.extend([
        {"id": "windows", "use": "op:concat", "in": {f"x{i}": f"e{i}" for i in range(3)}, "args": {"axis": 1}},
        {"id": "flat", "use": "op:reshape", "in": {"x": "windows"}, "args": {"shape": [history, 3 * width]}},
        {"id": "last", "use": "op:slice", "in": {"x": "$ctx"}, "args": {"axis": 0, "start": history, "stop": length}},
        {"id": "query", "use": "op:reshape", "in": {"x": "last"}, "args": {"shape": [3 * width, 1]}},
        {"id": "similarity", "use": "op:matmul", "in": {"a": "flat", "b": "query"}},
        {"id": "threshold", "use": "op:bias", "in": {"x": "similarity"}, "args": {"c": -2}},
        {"id": "match", "use": "op:relu", "in": {"x": "threshold"}},
        {"id": "weights", "use": "op:transpose", "in": {"x": "match"}, "args": {"perm": [1, 0]}},
        {"id": "continuations", "use": "op:slice", "in": {"x": "$ctx"}, "args": {"axis": 0, "start": 3, "stop": length}},
        {"id": "counts", "use": "op:matmul", "in": {"a": "weights", "b": "continuations"}},
        {"id": "vector", "use": "op:reshape", "in": {"x": "counts"}, "args": {"shape": [width]}},
        {"id": "repeat", "use": "op:bias", "in": {"x": "vector"}, "args": {"c": -1}},
        {"id": "twice", "use": "op:step", "in": {"x": "repeat"}},
        {"id": "penalty", "use": "op:scale", "in": {"x": "twice"}, "args": {"c": strength}}])
    graph.defs["long_ngram_memory"] = {"kind": "bundle", "role": "cluster", "inputs": ["ctx"],
                                        "nodes": nodes, "outputs": {"penalty": "penalty"}}
    root = graph.defs[graph.entries["step"]]
    previous = root["outputs"]["logits"]
    root["nodes"].extend([
        {"id": "long_memory", "use": "long_ngram_memory", "in": {"ctx": "$ctx"}},
        {"id": "memory_logits", "use": "op:sub", "in": {"a": previous, "b": "long_memory"}}])
    root["outputs"]["logits"] = "memory_logits"
    graph.meta.update({"parent_hash": parent.hash(), "language_interface": "unified-byte-memory/1",
                       "long_context_feature": "three-token continuation repetition, full context",
                       "memory_strength": strength,
                       "limitations": "long context only detects repeated prefixes; no demonstrated long-range understanding"})
    graph.validate()
    check_profile(graph, "ladder-micro/1")
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strength", type=float, default=2)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("output must be new")
    graph = add_memory(Graph.load(args.parent), args.strength)
    digest = graph.save(output)
    card = {"model_hash": digest, "parent_hash": graph.meta["parent_hash"], "stats": graph.stats(),
            "profile": "ladder-micro/1", "context_tokens": graph.meta["ctx_len"],
            "feature": graph.meta["long_context_feature"], "strength": args.strength,
            "numeric_provenance": "authored threshold of two matches and penalty strength; no corpus statistics",
            "limitations": graph.meta["limitations"], "training": "none",
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "MODEL_CARD.json").write_text(json.dumps(card, indent=2))
    (output / "BUILDER.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
