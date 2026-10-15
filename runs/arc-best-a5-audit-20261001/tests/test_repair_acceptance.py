import copy
import pytest
from cellforge.construction.repair_acceptance import repair_admission


def evidence():
    return {label: dict(partition="development", manifest=label,
        source="public-training-real" if label == "anchor" else "authored-generator",
        outcomes={"a": True, "b": False}, false_positive_given_consistency=0.)
        for label in ("fresh", "archive", "anchor")}


def test_admission_requires_gain_and_each_regression_gate():
    parent=evidence(); child=copy.deepcopy(parent)
    child.update(caps_passed=True,fixtures_passed=True)
    assert not repair_admission(parent,child)["accepted"]
    child["fresh"]["outcomes"]["b"]=True
    assert repair_admission(parent,child)["accepted"]
    for label in ("fresh","archive","anchor"):
        bad=copy.deepcopy(child);bad[label]["outcomes"]["a"]=False
        assert not repair_admission(parent,bad)["accepted"]
        bad=copy.deepcopy(child);bad[label]["false_positive_given_consistency"]=None
        assert not repair_admission(parent,bad)["accepted"]


def test_population_and_provenance_cannot_be_substituted():
    parent=evidence();child=copy.deepcopy(parent)
    for key,value in (("manifest","different"),("partition","final"),("source","synthetic")):
        bad=copy.deepcopy(child);bad["anchor"][key]=value
        with pytest.raises(ValueError):repair_admission(parent,bad)
