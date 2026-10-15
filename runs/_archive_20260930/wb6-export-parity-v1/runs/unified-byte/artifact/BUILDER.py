"""Lift the existing text MoE into one canonical UTF-8 token-output graph.

Existing experts are retained. Unsupported bytes have an explicit probability
floor and project to '?' in inherited narrow expert features. This preserves
input bytes and exposes limitations instead of stripping text in an adapter.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.runtime.text_codec import BYTE_CODEC, BYTE_VOCAB


def lift_graph(parent, context_length=64):
    old_vocab = list(parent.meta["vocab"])
    if parent.meta.get("text_codec") is not None or "?" not in old_vocab:
        raise ValueError("lift expects an inherited character MoE with '?' fallback")
    old_context = parent.meta["ctx_len"]
    if context_length < old_context:
        raise ValueError("new context must cover inherited expert context")
    mapping = [i if i < 3 else ord(token) + 3 for i, token in enumerate(old_vocab)]
    if any(i >= 259 for i in mapping):
        raise ValueError("parent vocabulary must use ASCII characters")
    g = parent.copy()
    if "byte_projection" in g.defs:
        raise ValueError("byte projection namespace already exists")
    nodes, outputs = [], {}
    for port, axis in (("ctx", 1), ("prev", 0)):
        pieces = {}
        for i, byte_id in enumerate(mapping):
            name = f"{port}_{i}"
            nodes.append({"id": name, "use": "op:slice", "in": {"x": "$" + port},
                          "args": {"axis": axis, "start": byte_id, "stop": byte_id + 1}})
            pieces[f"x{i}"] = name
        joined = port + "_joined"
        nodes.append({"id": joined, "use": "op:concat", "in": pieces, "args": {"axis": axis}})
        for name, source in ((port + "_all", "$" + port), (port + "_known", joined)):
            nodes.append({"id": name, "use": "op:sum", "in": {"x": source},
                          "args": {"axis": axis, "keepdims": True}})
        missing = port + "_missing"
        nodes.append({"id": missing, "use": "op:sub",
                      "in": {"a": port + "_all", "b": port + "_known"}})
        padded = port + "_unknown"
        question = old_vocab.index("?")
        nodes.append({"id": padded, "use": "op:pad", "in": {"x": missing},
                      "args": {"axis": axis, "before": question, "after": len(old_vocab) - question - 1}})
        projected = port + "_projected"
        nodes.append({"id": projected, "use": "op:add", "in": {"a": joined, "b": padded}})
        if port == "ctx":
            nodes.append({"id": "recent", "use": "op:slice", "in": {"x": projected},
                          "args": {"axis": 0, "start": context_length - old_context, "stop": context_length}})
            outputs[port] = "recent"
        else:
            outputs[port] = projected
    g.defs["byte_projection"] = {"kind": "bundle", "role": "cluster", "inputs": ["ctx", "prev"],
                                 "nodes": nodes, "outputs": outputs}
    root = g.defs[g.entries["step"]]
    inherited = copy.deepcopy(root["nodes"])
    for node in inherited:
        for key, ref in node.get("in", {}).items():
            if ref in ("$ctx", "$prev"):
                node["in"][key] = "byte_features." + ref[1:]
    root["nodes"] = [{"id": "byte_features", "use": "byte_projection",
                      "in": {"ctx": "$ctx", "prev": "$prev"}}] + inherited
    legacy_logits = root["outputs"]["logits"]
    root["nodes"].extend([
        {"id": "byte_first", "use": "op:slice", "in": {"x": legacy_logits},
         "args": {"axis": 0, "start": 0, "stop": 1}},
        {"id": "byte_zero", "use": "op:scale", "in": {"x": "byte_first"}, "args": {"c": 0}},
        {"id": "byte_floor", "use": "op:bias", "in": {"x": "byte_zero"}, "args": {"c": -12}}])
    lookup = {byte_id: i for i, byte_id in enumerate(mapping)}
    parts = {}
    for byte_id in range(259):
        if byte_id in lookup:
            name = f"byte_logit_{byte_id}"
            i = lookup[byte_id]
            root["nodes"].append({"id": name, "use": "op:slice", "in": {"x": legacy_logits},
                                  "args": {"axis": 0, "start": i, "stop": i + 1}})
            parts[f"x{byte_id}"] = name
        else:
            parts[f"x{byte_id}"] = "byte_floor"
    root["nodes"].append({"id": "byte_logits", "use": "op:concat", "in": parts, "args": {"axis": 0}})
    root["outputs"]["logits"] = "byte_logits"
    g.meta.update({"vocab": BYTE_VOCAB, "text_codec": BYTE_CODEC, "ctx_len": context_length,
                   "parent_hash": parent.hash(), "language_interface": "unified-byte-moe/1",
                   "inherited_expert_context": old_context,
                   "limitations": "inherited experts use only last eight projected bytes; unsupported-byte floor is not language knowledge"})
    g.validate()
    check_profile(g, "ladder-micro/1")
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("preserve prior artifacts: output must be new")
    graph = lift_graph(Graph.load(args.parent))
    digest = graph.save(output)
    card = {"model_hash": digest, "parent_hash": graph.meta["parent_hash"], "stats": graph.stats(),
            "profile": "ladder-micro/1", "input": "lossless UTF-8 bytes", "output": "one shared 259-token logit head",
            "state_dim": graph.meta["state_dim"], "context_tokens_received": graph.meta["ctx_len"],
            "context_tokens_used_by_inherited_experts": graph.meta["inherited_expert_context"],
            "limitations": graph.meta["limitations"], "training": "none; parent weights retained",
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "MODEL_CARD.json").write_text(json.dumps(card, indent=2))
    (output / "BUILDER.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
