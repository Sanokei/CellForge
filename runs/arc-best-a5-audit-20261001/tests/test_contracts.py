"""Component contracts, using a 20-slot full-adder cell as the fixture."""
import itertools

import numpy as np
import pytest

from cellforge.construction.bootstrap import bootstrap_lm
from cellforge.construction.contracts import run_contract
from cellforge.ir.graph import GraphError
from cellforge.ir.profiles import check_micro
from cellforge.reference.protocols import LM_CHARSET


def adder_graph():
    g = bootstrap_lm(LM_CHARSET)
    g.params["fa.W1"] = np.ones((3, 3))
    g.params["fa.b1"] = np.array([-0.5, -1.5, -2.5])
    g.params["fa.W2"] = np.array([[1.0, -1.0, 1.0], [0.0, 1.0, 0.0]])  # [sum, carry]
    g.params["fa.b2"] = np.zeros(2)
    g.defs["full_adder"] = {"kind": "cell", "act": "step",
                            "params": {k: f"fa.{k}" for k in ("W1", "b1", "W2", "b2")}}
    g.defs["cl_adder"] = {"kind": "bundle", "role": "cluster", "inputs": ["x"],
                          "nodes": [{"id": "fa", "use": "full_adder", "in": {"x": "$x"}}], "outputs": {"y": "fa"}}
    return g


def truth_table():
    return {"component": "cl_adder", "output": "y", "cases": [
        {"in": {"x": list(bits)}, "expect": [sum(bits) % 2, int(sum(bits) >= 2)]}
        for bits in itertools.product([0, 1], repeat=3)]}


def test_full_adder_cell_satisfies_contract_and_is_micro():
    g = adder_graph()
    res = run_contract(g, truth_table())
    assert res["ok"] and res["passed"] == 8
    check_micro(g)  # 20 slots <= 40


def test_contract_detects_wrong_component():
    g = adder_graph()
    g.params["fa.b1"] = np.array([-0.5, -2.5, -2.5])  # broken carry threshold
    res = run_contract(g, truth_table())
    assert not res["ok"] and res["passed"] < 8


def test_micro_cap_rejects_big_cell():
    g = adder_graph()
    g.params["fa.W1"] = np.ones((3, 30))
    with pytest.raises(GraphError, match="slots"):
        check_micro(g)
