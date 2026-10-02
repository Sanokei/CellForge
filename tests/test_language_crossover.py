from pathlib import Path
import numpy as np
import pytest
from cellforge.ir.graph import Graph
from cellforge.construction.language_crossover import crossover_module
from cellforge.runtime.lm import score_texts


def test_module_crossover_preserves_parents_and_reconnects_dispatch():
    parent = Graph.load(Path("runs/unified-byte/artifact"))
    donor = parent.copy()
    donor.meta["donor_fixture"] = True
    before = parent.hash(), donor.hash()
    child, receipt = crossover_module(parent, donor, "ex_bos", "ex_bos", "breed_")
    assert (parent.hash(), donor.hash()) == before
    assert receipt["rewired_sites"]
    dispatch = next(n for n in child.defs[child.entries["step"]]["nodes"] if n["use"] == "dispatch")
    assert "breed_ex_bos" in dispatch["args"]["branches"]
    for key, value in parent.params.items():
        assert np.array_equal(child.params[key], value)
    a, _ = score_texts(parent, ["Hi"])
    b, _ = score_texts(child, ["Hi"])
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_module_crossover_rejects_contract_or_namespace_mismatch():
    parent = Graph.load(Path("runs/unified-byte/artifact"))
    donor = parent.copy()
    donor.meta["ctx_len"] += 1
    with pytest.raises(ValueError, match="ctx_len"):
        crossover_module(parent, donor, "router", "router", "breed_")
    with pytest.raises(ValueError, match="namespace"):
        crossover_module(parent, parent, "router", "router", "")
