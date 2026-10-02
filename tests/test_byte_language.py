import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cellforge.construction.bootstrap import bootstrap_lm
from cellforge.ir.graph import Graph
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.lm import encode_text, generate, score_texts
from cellforge.runtime.text_codec import BYTE_CODEC, BYTE_VOCAB, decode_content, encode_content
from cellforge.reference.bench_operator import normalizer, score_choices

spec = importlib.util.spec_from_file_location("lift_text_moe", Path(__file__).parents[1] / "scripts/lift_text_moe_to_bytes.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def parent_fixture():
    graph = bootstrap_lm(" ?ab")
    graph.meta["state_dim"] = 1
    width = len(graph.meta["vocab"])
    for i in range(2):
        graph.params[f"logits_{i}"] = np.arange(width, dtype=float) * (1 if i == 0 else -1)
        graph.defs[f"cluster_{i}"] = {"kind": "bundle", "role": "cluster", "inputs": [],
            "nodes": [{"id": "lg", "use": f"param:logits_{i}"}], "outputs": {"logits": "lg"}}
        graph.defs[f"expert_{i}"] = {"kind": "bundle", "role": "expert", "inputs": [],
            "nodes": [{"id": "lg", "use": f"cluster_{i}"}], "outputs": {"logits": "lg"}}
    graph.defs["router"] = {"kind": "bundle", "role": "router", "inputs": ["prev"], "nodes": [
        {"id": "a", "use": "op:slice", "in": {"x": "$prev"}, "args": {"axis": 0, "start": 5, "stop": 6}},
        {"id": "minus", "use": "op:neg", "in": {"x": "a"}},
        {"id": "other", "use": "op:bias", "in": {"x": "minus"}, "args": {"c": 1}},
        {"id": "scores", "use": "op:concat", "in": {"x0": "a", "x1": "other"}, "args": {"axis": 0}}],
        "outputs": {"scores": "scores"}}
    graph.defs["step_root"] = {"kind": "bundle", "inputs": ["ctx", "prev", "state", "step"], "nodes": [
        {"id": "router", "use": "router", "in": {"prev": "$prev"}},
        {"id": "dispatch", "use": "dispatch", "in": {"scores": "router.scores"}, "args": {"branches": ["expert_0", "expert_1"]}},
        {"id": "count", "use": "op:sum", "in": {"x": "$prev"}, "args": {"axis": 0, "keepdims": True}},
        {"id": "state", "use": "op:add", "in": {"a": "$state", "b": "count"}}],
        "outputs": {"logits": "dispatch.logits", "state": "state"}}
    del graph.params["boot_logits"]
    return graph


def uniform_byte_graph():
    graph = bootstrap_lm("")
    graph.meta.update({"vocab": BYTE_VOCAB, "text_codec": BYTE_CODEC})
    graph.params["boot_logits"] = np.zeros(259)
    return graph


def test_lossless_unicode_and_token_denominators():
    text = 'Hello 42! café 中文 🦊\n\x00'
    tokens = encode_content(BYTE_VOCAB, text, BYTE_CODEC)
    assert decode_content(BYTE_VOCAB, tokens, BYTE_CODEC) == text
    result, info = score_texts(uniform_byte_graph(), [text])
    assert result["n_tokens"].item() == len(text.encode("utf-8")) + 1
    assert result["bits"].item() / result["n_tokens"].item() == pytest.approx(np.log2(259))
    assert info["token_unit"] == "utf8_byte"


def test_lift_preserves_supported_logits_state_and_routing(tmp_path):
    parent = parent_fixture()
    lifted = builder.lift_graph(parent)
    lifted.save(tmp_path)
    lifted = Graph.load(tmp_path)
    for character in ("a", "b", "?"):
        inputs = {"ctx": np.zeros((1, 8, 7)), "prev": np.eye(7)[[parent.meta["vocab"].index(character)]],
                  "state": np.array([[4.0]]), "step": np.array([[0.0]])}
        expected = Interpreter(parent).run_bundle("step_root", inputs)
        inputs["ctx"] = np.zeros((1, 64, 259))
        inputs["prev"] = np.eye(259)[[ord(character) + 3]]
        actual = Interpreter(lifted).run_bundle("step_root", inputs)
        indices = [0, 1, 2] + [ord(c) + 3 for c in parent.meta["vocab"][3:]]
        np.testing.assert_array_equal(actual["logits"][:, indices], expected["logits"])
        np.testing.assert_array_equal(actual["state"], expected["state"])


def test_byte_continuation_offsets_include_multibyte_prompt():
    graph = uniform_byte_graph()
    assert normalizer(BYTE_VOCAB, BYTE_CODEC)("Ä2 中") == ("Ä2 中", 1.0)
    bits = score_choices(graph, ["é", "中"], ["🙂", "a"])
    np.testing.assert_allclose(bits, np.array([4, 1]) * np.log2(259))


def test_invalid_generated_utf8_is_reported_without_repair():
    graph = uniform_byte_graph()
    graph.params["boot_logits"][3 + 255] = 10
    text, info = generate(graph, 1, max_len=1, prompts=["任意 words 42"])
    assert text == [None]
    assert info["failure"] == ["invalid_utf8"]
    assert info["token_ids"] == [[258]]


def test_byte_failure_penalty_cannot_beat_uniform_model():
    graph = uniform_byte_graph()
    graph.params["boot_logits"][5] = np.nan
    result, info = score_texts(graph, ["x"])
    assert result["failed"].item()
    assert result["correct"].item() == 0
    assert result["bits"].item() / result["n_tokens"].item() > np.log2(259)


def test_malformed_byte_vocabulary_rejected():
    with pytest.raises(ValueError, match="canonical"):
        encode_text(BYTE_VOCAB[:-1], "x", BYTE_CODEC)
