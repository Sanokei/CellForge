"""G0: recursive runtime engineering tests (not visual-discovery evidence)."""
import numpy as np
import pytest

from cellforge.construction.bootstrap import bootstrap
from cellforge.construction.transactions import (TransactionError, apply_ops, behavior_equal,
                                                 expand_pattern, validate_proposal)
from cellforge.ir.graph import Graph, GraphError, Limits
from cellforge.reference.metrics import levenshtein, score
from cellforge.runtime.decoder import decode
from cellforge.runtime.interp import Interpreter

RNG = np.random.default_rng(0)


def imgs(n=4):
    return RNG.random((n, 32, 128))


def cell(g, name, i, h, o, act="relu", shared=None):
    pre = shared or name
    if shared is None:
        g.params[f"{pre}.W1"] = RNG.normal(size=(h, i))
        g.params[f"{pre}.b1"] = RNG.normal(size=h)
        g.params[f"{pre}.W2"] = RNG.normal(size=(o, h))
        g.params[f"{pre}.b2"] = RNG.normal(size=o)
    g.defs[name] = {"kind": "cell", "act": act,
                    "params": {k: f"{pre}.{k}" for k in ("W1", "b1", "W2", "b2")}}


def nested_graph():
    """A contains cells; B contains A + cells; C contains B, another A, and a cell."""
    g = bootstrap()
    cell(g, "c1", 4, 4, 4)
    cell(g, "c2", 4, 4, 4)
    cell(g, "c3", 4, 4, 4)
    cell(g, "c5", 8, 2, 4)
    g.defs["A"] = {"kind": "bundle", "inputs": ["x"], "nodes": [
        {"id": "a1", "use": "c1", "in": {"x": "$x"}},
        {"id": "a2", "use": "c2", "in": {"x": "a1"}}], "outputs": {"y": "a2"}}
    g.defs["B"] = {"kind": "bundle", "inputs": ["x"], "nodes": [
        {"id": "inA", "use": "A", "in": {"x": "$x"}},
        {"id": "b3", "use": "c3", "in": {"x": "inA"}}], "outputs": {"y": "b3", "skip": "inA"}}
    g.defs["C"] = {"kind": "bundle", "inputs": ["x"], "nodes": [
        {"id": "b", "use": "B", "in": {"x": "$x"}},
        {"id": "a", "use": "A", "in": {"x": "$x"}},
        {"id": "cat", "use": "op:concat", "in": {"x0": "b.y", "x1": "a"}, "args": {"axis": -1}},
        {"id": "c5", "use": "c5", "in": {"x": "cat"}}], "outputs": {"y": "c5"}}
    g.defs["encode_root"] = {"kind": "bundle", "inputs": ["image"], "nodes": [
        {"id": "p", "use": "op:patchify", "in": {"x": "$image"}, "args": {"ph": 2, "pw": 2}},
        {"id": "feat", "use": "C", "in": {"x": "p"}},
        {"id": "pool", "use": "op:mean", "in": {"x": "feat"}, "args": {"axis": 0}},
        {"id": "pool2", "use": "op:mean", "in": {"x": "pool"}, "args": {"axis": 0}}],
        "outputs": {"colink": "pool2"}}
    g.params["proj"] = RNG.normal(size=(4, 13))
    g.defs["step_root"] = {"kind": "bundle", "inputs": ["colink", "step", "prev", "state"], "nodes": [
        {"id": "w", "use": "param:proj"},
        {"id": "lg", "use": "op:matmul", "in": {"a": "$colink", "b": "w"}},
        ], "outputs": {"logits": "lg", "state": "$state"}}
    return g


def test_bootstrap_emits_empty_and_scores_cer_one():
    g = bootstrap()
    g.validate()
    preds, info = decode(g, imgs())
    assert preds == [""] * 4
    s = score(preds, ["12", "3", "", "456"])
    assert s["cer"] == 1.0 and s["total_ref_chars"] == 6


def test_nested_mixed_bundles_execute_and_count():
    g = nested_graph()
    g.validate()
    g.params["proj"] = RNG.normal(size=(4, 13))
    preds, info = decode(g, imgs())
    assert len(preds) == 4
    st = g.stats()
    assert st["nesting_depth"] == 5  # encode_root > C > B > A > cell
    assert st["expanded_instances"] > len(g.defs) - 3


def test_recursion_rejected():
    g = nested_graph()
    g.defs["A"]["nodes"].append({"id": "loop", "use": "C", "in": {"x": "$x"}})
    with pytest.raises(GraphError, match="recursive"):
        g.validate()


def test_expansion_limit():
    g = nested_graph()
    g.params["proj"] = RNG.normal(size=(4, 13))
    # exponential doubling chain: D_k contains two D_{k-1}
    prev = "A"
    for k in range(20):
        name = f"D{k}"
        g.defs[name] = {"kind": "bundle", "inputs": ["x"], "nodes": [
            {"id": "l", "use": prev, "in": {"x": "$x"}},
            {"id": "r", "use": prev, "in": {"x": "l"}}], "outputs": {"y": "r"}}
        prev = name
    g.defs["encode_root"]["nodes"][1]["use"] = prev
    with pytest.raises(GraphError, match="expanded instances"):
        g.validate(Limits(max_expanded_instances=10_000))


def test_sharing_is_explicit_and_clone_on_write():
    g = nested_graph()
    g.params["proj"] = RNG.normal(size=(4, 13))
    cell(g, "c1_shared", 4, 4, 4, shared="c1")
    assert g.defs["c1_shared"]["params"] == g.defs["c1"]["params"]
    g2, pure = apply_ops(g, [{"op": "set_param", "name": "c1copy.W1", "pattern": {"copy": "c1.W1"}}])
    assert not pure and np.array_equal(g2.params["c1copy.W1"], g.params["c1.W1"])
    assert g2.params["c1copy.W1"] is not g2.params["c1.W1"]


def test_wrap_unwrap_preserve_behavior():
    g = nested_graph()
    g.params["proj"] = RNG.normal(size=(4, 13))
    x = imgs()
    g2, pure = apply_ops(g, [{"op": "wrap", "parent": "C", "nodes": ["a", "cat"],
                              "new_def": "Wr", "new_node": "wr"}])
    assert pure
    g2.validate()
    ok, why = behavior_equal(g, g2, x)
    assert ok, why
    g3, pure = apply_ops(g2, [{"op": "unwrap", "parent": "C", "node": "wr"}])
    assert pure
    ok, why = behavior_equal(g, g3, x)
    assert ok, why
    g4, _ = apply_ops(g, [{"op": "unwrap", "parent": "encode_root", "node": "feat"}])
    ok, why = behavior_equal(g, g4, x)
    assert ok, why


def test_validate_proposal_stale_base_and_nonfinite():
    g = bootstrap()
    x = imgs(2)
    r = validate_proposal(g, {"base": "0" * 64, "ops": [{"op": "set_state_dim", "value": 1}]}, x)
    assert not r["ok"] and "stale" in r["error"]
    r = validate_proposal(g, {"base": g.hash(), "ops": [
        {"op": "set_param", "name": "boot_logits", "value": [float("nan")] * 13}]}, x)
    assert not r["ok"]
    r = validate_proposal(g, {"base": g.hash(), "ops": [
        {"op": "put_def", "name": "bad", "def": {"kind": "bundle", "inputs": [], "nodes": [
            {"id": "q", "use": "nonexistent"}], "outputs": {"y": "q"}}}]}, x)
    assert not r["ok"]


def test_dispatch_is_lazy_and_ties_lowest():
    g = bootstrap()
    g.params["k0"] = np.array([1.0])
    g.params["k1"] = np.array([2.0])
    for i in (0, 1):
        g.defs[f"br{i}"] = {"kind": "bundle", "inputs": ["x"], "nodes": [
            {"id": "k", "use": f"param:k{i}"},
            {"id": "o", "use": "op:mul", "in": {"a": "$x", "b": "k"}}], "outputs": {"y": "o"}}
    g.defs["router"] = {"kind": "bundle", "inputs": ["x", "s"], "nodes": [
        {"id": "d", "use": "dispatch", "in": {"scores": "$s", "x": "$x"},
         "args": {"branches": ["br0", "br1"]}}], "outputs": {"y": "d.y"}}
    it = Interpreter(g)
    x = np.ones((3, 1))
    s = np.array([[0.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    out = it.run_bundle("router", {"x": x, "s": s})
    assert out["y"].ravel().tolist() == [2.0, 2.0, 1.0]
    it2 = Interpreter(g)
    it2.run_bundle("router", {"x": x, "s": np.array([[0, 1.0]] * 3)})
    assert it2.stats.route_counts["/d"] == {"br0": 0, "br1": 3}
    it3 = Interpreter(g)
    out = it3.run_bundle("router", {"x": x, "s": np.array([[np.nan, 1.0], [0, 1.0], [1.0, 0]])})
    assert np.isnan(out["y"][0, 0]) and out["y"][1, 0] == 2.0


def test_save_load_roundtrip_and_integrity(tmp_path):
    g = nested_graph()
    g.params["proj"] = RNG.normal(size=(4, 13))
    h = g.save(tmp_path / "a")
    g2 = Graph.load(tmp_path / "a")
    assert g2.hash() == h
    ok, _ = behavior_equal(g, g2, imgs())
    assert ok
    # tamper
    import json
    man = json.loads((tmp_path / "a" / "graph.json").read_text())
    man["params"]["proj"]["sha256"] = "0" * 64
    (tmp_path / "a" / "graph.json").write_text(json.dumps(man))
    with pytest.raises(GraphError, match="integrity"):
        Graph.load(tmp_path / "a")


def test_strict_microcells():
    g = nested_graph()
    g.params["proj"] = RNG.normal(size=(4, 13))
    g.validate(Limits(strict_microcells=True))  # 4-4-4 = 40 slots
    cell(g, "big", 8, 8, 8)
    with pytest.raises(GraphError, match="strict microcell"):
        g.validate(Limits(strict_microcells=True))


def test_patterns():
    g = bootstrap()
    a = expand_pattern({"ascii": ["#.", ".#"], "map": {"#": 2, ".": -1}}, g)
    assert a.tolist() == [[2, -1], [-1, 2]]
    b = expand_pattern({"sparse": [[0, 1, 3.0]], "shape": [2, 2]}, g)
    assert b[0, 1] == 3.0
    with pytest.raises(TransactionError):
        expand_pattern({"ascii": ["x"], "map": {"#": 1}}, g)


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    s = score([None, "ab", "abc"], ["ab", "ab", ""])
    assert s["total_edits"] == 2 + 0 + 3 and s["total_ref_chars"] == 4 and s["cer"] == 1.25
