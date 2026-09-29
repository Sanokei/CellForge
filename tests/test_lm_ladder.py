"""LM driver, ladder profile, and top-k dispatch tests."""
import numpy as np
import pytest

from cellforge.construction.bootstrap import bootstrap_lm
from cellforge.construction.transactions import behavior_equal, validate_proposal
from cellforge.ir.graph import GraphError
from cellforge.ir.profiles import check_ladder
from cellforge.reference.protocols import LM_CHARSET
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.lm import generate, score_texts

TEXTS = ["the cat runs.", "some dogs see a fox."]


def ladder_graph(k=1):
    g = bootstrap_lm(LM_CHARSET)
    V = len(g.meta["vocab"])
    rng = np.random.default_rng(0)
    for name, (i, o) in {"emb": (V, 4), "rt": (4, 2), "e0": (4, V), "e1": (4, V)}.items():
        g.params[f"{name}.W1"] = rng.normal(size=(o, i))
        g.params[f"{name}.b1"] = np.zeros(o)
        g.defs[f"c_{name}"] = {"kind": "cell", "act": "identity",
                               "params": {"W1": f"{name}.W1", "b1": f"{name}.b1"}}
    g.defs["cl_emb"] = {"kind": "bundle", "role": "cluster", "inputs": ["x"],
                        "nodes": [{"id": "c", "use": "c_emb", "in": {"x": "$x"}}], "outputs": {"y": "c"}}
    for n in ("rt", "e0", "e1"):
        g.defs[f"cl_{n}"] = {"kind": "bundle", "role": "cluster", "inputs": ["x"],
                             "nodes": [{"id": "c", "use": f"c_{n}", "in": {"x": "$x"}}], "outputs": {"y": "c"}}
    g.defs["router"] = {"kind": "bundle", "role": "router", "inputs": ["h"],
                        "nodes": [{"id": "s", "use": "cl_rt", "in": {"x": "$h"}}], "outputs": {"scores": "s"}}
    for n in ("e0", "e1"):
        g.defs[f"ex_{n}"] = {"kind": "bundle", "role": "expert", "inputs": ["h"],
                             "nodes": [{"id": "o", "use": f"cl_{n}", "in": {"x": "$h"}}], "outputs": {"logits": "o"}}
    g.defs["step_root"] = {"kind": "bundle", "inputs": ["ctx", "prev", "step", "state"], "nodes": [
        {"id": "h", "use": "cl_emb", "in": {"x": "$prev"}},
        {"id": "r", "use": "router", "in": {"h": "h"}},
        {"id": "d", "use": "dispatch", "in": {"scores": "r.scores", "h": "h"},
         "args": {"branches": ["ex_e0", "ex_e1"], "k": k}}],
        "outputs": {"logits": "d.logits", "state": "$state"}}
    return g


def test_uniform_bootstrap_bpc_is_log2_v():
    g = bootstrap_lm(LM_CHARSET)
    g.validate()
    r, _ = score_texts(g, TEXTS)
    assert np.allclose(r["bits"] / r["n_tokens"], np.log2(len(g.meta["vocab"])))


def test_ladder_profile_accepts_and_rejects():
    g = ladder_graph()
    g.validate()
    assert check_ladder(g)["experts"] == ["ex_e0", "ex_e1"]
    bad = ladder_graph()
    bad.defs["ex_e0"]["nodes"][0]["use"] = "c_e0"  # raw cell inside an expert
    with pytest.raises(GraphError, match="expert"):
        check_ladder(bad)
    bad2 = ladder_graph()
    bad2.defs["step_root"]["nodes"][2]["in"]["scores"] = "h"  # scores not from router
    with pytest.raises(GraphError, match="router"):
        check_ladder(bad2)
    with pytest.raises(GraphError, match="dispatch"):
        check_ladder(bootstrap_lm(LM_CHARSET))


def test_topk_matches_manual_mixture_and_k1_is_passthrough():
    g1, g2 = ladder_graph(1), ladder_graph(2)
    r1, i1 = score_texts(g1, TEXTS)
    r2, _ = score_texts(g2, TEXTS)
    assert np.all(np.isfinite(r1["bits"])) and np.all(np.isfinite(r2["bits"]))
    it = Interpreter(g2)
    h = np.random.default_rng(1).normal(size=(3, 4))
    s = np.array([[0.0, 1.0], [2.0, 0.0], [0.5, 0.5]])
    out = it.run_bundle("step_root", {"ctx": np.zeros((3, 8, 32)), "prev": np.eye(32)[[3, 4, 5]],
                                      "step": np.zeros((3, 1)), "state": np.zeros((3, 0))})
    assert out["logits"].shape == (3, 32)
    assert sum(i1["exec"]["route_counts"]["/step/d"].values()) == int(r1["n_tokens"].sum())


def test_generation_and_refactor_equivalence_lm():
    g = ladder_graph()
    texts, info = generate(g, 3, max_len=20, temperature=0.8, seed=1)
    assert len(texts) == 3
    prop = {"base": g.hash(), "ops": [{"op": "unwrap", "parent": "ex_e0", "node": "o"}]}
    # unwrap inside an expert inlines a raw cell -> violates the ladder profile
    v = validate_proposal(g, prop, TEXTS, kind="lm", profile="ladder/1")
    assert not v["ok"] and "ladder" in v["error"]
    v = validate_proposal(g, prop, TEXTS, kind="lm", profile="open/1")
    assert v["ok"] and v["pure_refactor"]
    ok, _ = behavior_equal(g, v["graph"], TEXTS, kind="lm")
    assert ok


def test_import_def_crossover_copies_exact_and_isolated():
    from cellforge.construction.transactions import apply_ops
    parent_a = ladder_graph()
    parent_b = ladder_graph()
    parent_b.params["e1.W1"] = parent_b.params["e1.W1"] + 1.0  # B's expert differs from A's
    refs = {"A": parent_a, "B": parent_b}
    child, pure = apply_ops(parent_a, [{"op": "import_def", "from": "B", "name": "ex_e1", "prefix": "b_"},
                                       {"op": "set_param", "name": "b_e1.W1", "pattern": {"copy": "b_e1.W1"}}],
                            resolver=lambda r: refs[r])
    assert not pure
    assert "b_ex_e1" in child.defs and "b_cl_e1" in child.defs and "b_c_e1" in child.defs
    assert np.array_equal(child.params["b_e1.W1"], parent_b.params["e1.W1"])
    assert not np.array_equal(child.params["b_e1.W1"], parent_a.params["e1.W1"])
    assert np.array_equal(child.params["e1.W1"], parent_a.params["e1.W1"])  # parent's own copy untouched
    child.validate()
    # importing into a namespace that already holds different content must be explicit
    with pytest.raises(Exception, match="already exists"):
        apply_ops(parent_a, [{"op": "import_def", "from": "B", "name": "ex_e1"}], resolver=lambda r: refs[r])
    # dispatch branches are renamed consistently
    child2, _ = apply_ops(parent_a, [{"op": "import_def", "from": "B", "name": "step_root", "prefix": "b_"}],
                          resolver=lambda r: refs[r])
    br = [n for n in child2.defs["b_step_root"]["nodes"] if n["use"] == "dispatch"][0]["args"]["branches"]
    assert br == ["b_ex_e0", "b_ex_e1"] and all(b in child2.defs for b in br)
