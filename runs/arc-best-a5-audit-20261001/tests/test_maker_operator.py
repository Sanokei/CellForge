import pytest
from cellforge.ir.graph import Graph
from cellforge.construction.maker_operator import MakerOperator


def inputs():
    graph = Graph(meta={})
    graph.defs["root"] = {"kind": "bundle", "inputs": ["x"],
        "nodes": [{"id": "a", "use": "op:scale", "in": {"x": "$x"}, "args": {"c": 1}}],
        "outputs": {"y": "a"}}
    graph.entries = {"encode": "root", "step": "root"}
    fixture = [{"entry": "root", "inputs": {"x": [[1.]]}, "output_shapes": {"y": [1, 1]}}]
    return graph, fixture


def submit(operator, graph, fixtures, **kwargs):
    args = dict(diagnostics={"partition": "development", "evidence": "train-pairs", "remaining_error": 1},
        provenance={"actor": "assistant", "model": "recorded-session-model", "method": "manual-llm-maker"},
        ops=[], fixtures=fixtures)
    args.update(kwargs)
    return operator.submit(graph, **args)


def test_invalid_and_nochange_consume_budget(tmp_path):
    graph, fixtures = inputs()
    operator = MakerOperator(tmp_path / "ledger.jsonl", budget=2, profile="flat/1")
    _, first = submit(operator, graph, fixtures, ops=[{"op": "missing"}])
    _, second = submit(operator, graph, fixtures)
    assert first["status"] == "invalid" and second["status"] == "no-change"
    assert second["previous_receipt"] == first["receipt_hash"]
    with pytest.raises(ValueError, match="budget"):
        submit(operator, graph, fixtures)
    assert len(MakerOperator(operator.path, budget=2, profile="flat/1").rows) == 2


def test_reserved_diagnostics_and_shape_rejected(tmp_path):
    graph, fixtures = inputs()
    operator = MakerOperator(tmp_path / "ledger", budget=2, profile="flat/1")
    _, receipt = submit(operator, graph, fixtures, diagnostics={"partition": "development", "evidence": "train-pairs", "test_output": []})
    assert receipt["status"] == "invalid"
    fixtures[0]["output_shapes"]["y"] = [2, 2]
    _, receipt = submit(operator, graph, fixtures)
    assert receipt["status"] == "invalid"


def test_changed_candidate_score_and_sealed_once(tmp_path):
    graph, fixtures = inputs()
    operator = MakerOperator(tmp_path / "ledger", budget=1, profile="flat/1")
    definition = graph.copy().defs["root"]
    definition["nodes"][0]["args"]["c"] = 2
    child, receipt = submit(operator, graph, fixtures,
        ops=[{"op": "put_def", "name": "root", "def": definition}],
        evaluate=lambda candidate: {"remaining_error": 0}, accept=lambda metric: metric["remaining_error"] == 0)
    assert receipt["status"] == "accepted" and child.hash() != graph.hash()
    operator.freeze(child)
    assert operator.sealed_once(child, lambda candidate: {"score": 0}) == {"score": 0}
    with pytest.raises(ValueError, match="already called"):
        operator.sealed_once(child, lambda candidate: None)
    with pytest.raises(ValueError, match="frozen"):
        submit(operator, graph, fixtures)


def test_breed_repair_round_charges_all_arms_and_logs_counterexamples(tmp_path):
    graph, fixtures = inputs()
    operator = MakerOperator(tmp_path / "round.jsonl", budget=3, profile="flat/1")
    diagnostics = {"partition": "development", "evidence": "train-pairs",
        "counterexamples": [{"input": [[1]], "expected": [[2]],
            "predicted": [[1]], "cells_fired": ["root/a"]}]}
    repairs = {arm: {"ops": [], "actor": "recorded-author", "model": "recorded-model"}
        for arm in ("random_repair", "maker_repair")}
    scored = []
    result = operator.breed_then_repair_round(graph, diagnostics=diagnostics,
        repairs=repairs, fixtures=fixtures,
        evaluate=lambda candidate: scored.append(candidate.hash()) or {"solved": 0},
        accept=lambda metrics: False)
    assert len(scored) == 3 and len(operator.rows) == 3
    assert set(result) == {"breeding_only", "random_repair", "maker_repair"}
    assert all(row["diagnostics_shown"] == diagnostics for row in operator.rows)
    assert all(row["status"] == "rejected" for row in operator.rows)
    assert all(row["evaluation_calls"] == 1 for row in operator.rows)
    with pytest.raises(ValueError, match="budget"):
        operator.breed_then_repair_round(graph, diagnostics=diagnostics,
            repairs=repairs, fixtures=fixtures, evaluate=lambda _: {}, accept=lambda _: False)


def test_round_invalid_repair_consumes_attempt_and_other_arm_runs(tmp_path):
    graph, fixtures = inputs()
    operator = MakerOperator(tmp_path / "round", budget=3, profile="flat/1")
    diagnostics = {"partition": "development", "evidence": "train-pairs",
        "counterexamples": [{"input": [[1]], "expected": [[2]],
            "predicted": [[1]], "cells_fired": ["root/a"]}]}
    result = operator.breed_then_repair_round(graph, diagnostics=diagnostics,
        repairs={"random_repair": {}, "maker_repair": {"ops": [], "actor": "maker", "model": "recorded"}},
        fixtures=fixtures, evaluate=lambda _: {"solved": 0}, accept=lambda _: False)
    assert [row["status"] for row in operator.rows] == ["rejected", "invalid", "rejected"]
    assert [row["evaluation_calls"] for row in operator.rows] == [1, 0, 1]
    assert len(result) == 3
    reopened = MakerOperator(operator.path, budget=3, profile="flat/1")
    assert len(reopened.rows) == 3
    import json
    rows = [json.loads(line) for line in operator.path.read_text().splitlines()]
    rows[0]["status"] = "accepted"
    operator.path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="integrity"):
        MakerOperator(operator.path, budget=3, profile="flat/1")


def test_failed_evaluator_is_charged_and_timed(tmp_path):
    graph, fixtures = inputs()
    operator = MakerOperator(tmp_path / "failed", budget=1, profile="flat/1")
    def fail(candidate):
        raise RuntimeError("evaluation failed")
    _, receipt = submit(operator, graph, fixtures, evaluate=fail, evaluate_unchanged=True)
    assert receipt["status"] == "invalid" and receipt["evaluation_calls"] == 1
    assert receipt["evaluation_wall_seconds"] >= 0 and receipt["evaluation_cpu_seconds"] >= 0
    assert "evaluation failed" in receipt["error"]
    assert len(MakerOperator(operator.path, budget=1, profile="flat/1").rows) == 1


def test_admission_regression_cannot_be_overridden_by_callback(tmp_path):
    import copy
    graph, fixtures = inputs()
    parent = {label: dict(partition="development", manifest=label,
        source="public-training-real" if label == "anchor" else "authored-generator",
        outcomes={"a": True, "b": False}, false_positive_given_consistency=0.)
        for label in ("fresh", "archive", "anchor")}
    candidate = copy.deepcopy(parent)
    candidate.update(caps_passed=True, fixtures_passed=True)
    candidate["fresh"]["outcomes"]["b"] = True
    candidate["anchor"]["outcomes"]["a"] = False
    operator = MakerOperator(tmp_path / "admission", budget=2, profile="flat/1")
    _, receipt = submit(operator, graph, fixtures, evaluate_unchanged=True,
        evaluate=lambda _: candidate, accept=lambda _: True, admission_parent=parent)
    assert receipt["status"] == "rejected"
    assert receipt["admission"]["reasons"][0]["gate"] == "anchor_regression"
    assert receipt["admission_parent"] == parent
    candidate["anchor"]["outcomes"]["a"] = True
    _, receipt = submit(operator, graph, fixtures, evaluate_unchanged=True,
        evaluate=lambda _: candidate, accept=lambda _: True, admission_parent=parent)
    assert receipt["status"] == "accepted"
    assert len(MakerOperator(operator.path,budget=2,profile="flat/1").rows) == 2
