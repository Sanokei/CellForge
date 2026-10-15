"""Protocol tests; mocked generation is explicitly not an LLM result."""
import json
import pytest
import numpy as np

from cellforge.runtime import specialists
from cellforge.ir.graph import Graph
from test_grid_runtime import recolor_graph
from test_byte_language import uniform_byte_graph


@pytest.mark.parametrize("text", [
    '{"type":"respond","type":"call","text":"x"}',
    '{"type":"respond","text":NaN}',
    '{"type":"call","name":"arc_grid","arguments":{},"extra":1}',
    '[]', 'instructions before {"type":"respond","text":"x"}',
])
def test_strict_action_rejection(text):
    with pytest.raises(ValueError):
        specialists.parse_action(text)


def test_scripted_roundtrip_uses_observation(monkeypatch):
    call = {"type": "call", "name": "arc_grid", "arguments": {
        "demonstrations": [{"input": [[0]], "output": [[1]]}],
        "test_input": [[8, 9]]}}
    prompts = []
    def scripted(graph, n, prompts=None, **kwargs):
        # Test only host transcript flow. This is no learned policy.
        globals_prompt = prompts[0]
        seen.append(globals_prompt)
        if len(seen) == 1:
            action = call
        else:
            transcript = json.loads(globals_prompt.removesuffix("\nAction:"))
            attempts = transcript[-1]["content"]["result"]["attempts"]
            action = {"type": "respond", "text": json.dumps(attempts)}
        text = json.dumps(action)
        return [text], {"failure": [None], "token_ids": [[0] * len(text)]}
    seen = []
    monkeypatch.setattr(specialists, "generate", scripted)
    result = specialists.run_model_with_specialist(uniform_byte_graph(), "synthetic grid", recolor_graph())
    assert result["status"] == "responded"
    assert result["tool_calls"] == 1
    assert json.loads(result["answer"]) == [[[9, 0]]]
    assert "transcript" not in result


def test_target_leakage_and_invalid_calls_never_execute(monkeypatch):
    action = {"type": "call", "name": "arc_grid", "arguments": {
        "demonstrations": [{"input": [[0]], "output": [[1]]}],
        "test_input": [[2]], "test_output": [[3]]}}
    monkeypatch.setattr(specialists, "generate", lambda *a, **kw: (
        [json.dumps(action)], {"failure": [None], "token_ids": [[1]]}))
    monkeypatch.setattr(specialists.SpecialistBroker, "call", lambda *a: pytest.fail("forbidden execution"))
    result = specialists.run_model_with_specialist(uniform_byte_graph(), "x", recolor_graph(), max_steps=2)
    assert result["invalid_actions"] == 2
    assert result["tool_calls"] == 0


def test_context_budget_prevents_generation(monkeypatch):
    monkeypatch.setattr(specialists, "generate", lambda *a, **kw: pytest.fail("budget bypass"))
    result = specialists.run_model_with_specialist(uniform_byte_graph(), "x", recolor_graph(), max_context_bytes=1)
    assert result["status"] == "context_budget_exhausted"
    assert result["steps"] == 0


def test_disabled_tools_are_neither_advertised_nor_executed(monkeypatch):
    seen = []
    action = {"type": "call", "name": "arc_grid", "arguments": {
        "demonstrations": [{"input": [[0]], "output": [[1]]}],
        "test_input": [[2]]}}

    def scripted(*args, **kwargs):
        seen.append(json.loads(kwargs["prompts"][0].removesuffix("\nAction:")))
        return [json.dumps(action)], {"failure": [None], "token_ids": [[1]]}

    monkeypatch.setattr(specialists, "generate", scripted)
    monkeypatch.setattr(specialists.SpecialistBroker, "call",
                        lambda *args: pytest.fail("disabled tool executed"))
    result = specialists.run_model_with_specialist(
        uniform_byte_graph(), "x", recolor_graph(), enabled_tools=[], max_steps=1)
    assert "arc_grid" not in seen[0][0]["content"]
    assert result["invalid_actions"] == 1
    assert result["tool_calls"] == 0


def test_deduction_broker_executes_graph_and_rejects_hidden_targets():
    from pathlib import Path
    graph = Graph.load(Path(__file__).parents[1] / "runs/parallel-deduction/artifact-v1")
    broker = specialists.SpecialistBroker(deduction_graph=graph)
    arguments = {"facts": ["seed"], "rules": [
        {"if": ["seed"], "then": "middle"},
        {"if": ["middle"], "then": "finish"}], "query": "finish"}
    result = broker.call("deduction", arguments)
    assert result["status"] == "entailed"
    assert result["specialist_hash"] == graph.hash()
    assert broker.call("deduction", {**arguments, "query": "missing"})["status"] == "not_proven"
    for malformed in ({**arguments, "answer": True}, {**arguments, "query": "not_finish".replace("_", " ")},
                      {**arguments, "facts": ["!seed"]}):
        with pytest.raises(ValueError):
            broker.validate_call("deduction", malformed)
    assert "deduction" not in broker.contract([])
    graph.meta["policy"] = "changed"
    with pytest.raises(specialists.GraphError, match="changed"):
        broker.call("deduction", arguments)


def test_typed_policy_transfer_and_observation_ablation(tmp_path):
    # Unseen noninteger, signed payloads: transfer is generic arithmetic,
    # not memorized grids or language interpretation.
    graph = specialists.build_typed_observation_policy(7)
    graph.save(tmp_path)
    loaded = Graph.load(tmp_path)
    rng = np.random.default_rng(90217)
    for _ in range(32):
        payload = rng.normal(size=7)
        call = specialists.typed_observation_action(loaded, [1, 0, 0], np.zeros(7))
        response = specialists.typed_observation_action(loaded, [1, 1, 1], payload)
        invalid = specialists.typed_observation_action(loaded, [1, 1, 0], payload)
        absent = specialists.typed_observation_action(loaded, [0, 0, 0], payload)
        assert call["type"] == "call"
        assert response["type"] == "respond"
        assert np.array_equal(response["response_values"], payload)
        assert invalid["type"] == absent["type"] == "abstain"
        assert invalid["response_values"] is None
        ablated = specialists.typed_observation_action(loaded, [1, 1, 1], np.zeros(7))
        assert not np.array_equal(ablated["response_values"], payload)


def test_typed_policy_projection_is_graph_owned():
    graph = specialists.build_typed_observation_policy(3)
    graph.params["projection_W"] = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    result = specialists.typed_observation_action(graph, [1, 1, 1], [4., 7., 9.])
    assert result["response_values"] == [7., 9., 4.]
    with pytest.raises(ValueError):
        specialists.typed_observation_action(graph, [1, 1, 1], [float("nan"), 0, 0])
