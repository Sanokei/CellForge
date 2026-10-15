from pathlib import Path
import numpy as np
import pytest
from PIL import ImageFont
from cellforge.data.render import render_one
from cellforge.ir.graph import Graph, GraphError
from cellforge.runtime.text_codec import BYTE_CODEC, BYTE_VOCAB, decode_content
from cellforge.runtime.vision import OCRPredictor

ROOT = Path(__file__).resolve().parents[1]


def graph():
    return Graph.load(ROOT / "runs/p1/exports/r1-iter-b")


def test_actual_digit_graph_returns_image_conditioned_byte_observation():
    predictor = OCRPredictor(graph())
    image = render_one("123", ImageFont.truetype(str(ROOT / "fonts/DejaVuSansMono.ttf"), 20), 6, 0)
    result = predictor(image)
    assert result["text"] == "123"
    assert decode_content(BYTE_VOCAB, result["token_ids"], BYTE_CODEC) == "123"
    assert result["failure"] is None
    assert result["graph_hash"] == graph().hash()
    assert result["observation_kind"] == "image-conditioned-ocr-text"


@pytest.mark.parametrize("artifact", [
    "runs/p1/exports/r1-iter-b", "runs/parallel-vision/alphabet/candidate",
    "runs/parallel-vision/alphabet-crossovers/population/b_glyph_a",
    "runs/parallel-vision/alphabet-crossovers/population/b_glyph_oneshot",
    "runs/parallel-vision/alphabet-crossovers/population/a_glyph_oneshot",
])
def test_batch_matches_independent_crops_and_keeps_costs_aggregate(artifact):
    font = ImageFont.truetype(str(ROOT / "fonts/DejaVuSansMono.ttf"), 20)
    images = np.stack([render_one(text, font, 6, 0) for text in ("", "7", "123", "42")])
    predictor = OCRPredictor(Graph.load(ROOT / artifact))
    expected = [{k: v for k, v in predictor(image).items() if k != "exec"} for image in images]
    actual, costs = predictor.predict_batch(images)
    assert actual == expected
    assert costs["exec_scope"] == "whole-batch"
    assert costs["batch_size"] == 4
    assert all("exec" not in item for item in actual)
    reversed_results, _ = predictor.predict_batch(images[::-1])
    assert reversed_results == expected[::-1]


def test_batch_rejects_empty_and_changed_graph():
    predictor = OCRPredictor(graph())
    with pytest.raises(GraphError, match="nonempty"):
        predictor.predict_batch(np.zeros((0, 32, 128), dtype=np.uint8))
    predictor.graph.meta["max_len"] = 1
    with pytest.raises(GraphError, match="changed"):
        predictor.predict_batch(np.zeros((1, 32, 128), dtype=np.uint8))


@pytest.mark.parametrize("image", [np.zeros((32, 128)), np.zeros((128, 32), dtype=np.uint8), np.zeros((1, 32, 128), dtype=np.uint8)])
def test_rejects_ambiguous_pixel_inputs(image):
    with pytest.raises(GraphError, match="single uint8"):
        OCRPredictor(graph())(image)


def test_rejects_wrong_normalization_contract():
    artifact = graph()
    artifact.meta["pixel_contract"]["normalize"] = "already-normalized"
    with pytest.raises(GraphError, match="contract"):
        OCRPredictor(artifact)


def test_decoder_failure_never_becomes_language_content():
    artifact = graph()
    artifact.meta["max_len"] = 0
    result = OCRPredictor(artifact)(np.zeros((32, 128), dtype=np.uint8))
    assert result["failure"] == "missing_eos"
    assert result["text"] is None
    assert result["token_ids"] == []


def test_predictor_freezes_caller_graph():
    artifact = graph()
    predictor = OCRPredictor(artifact)
    artifact.meta["max_len"] = 0
    assert predictor.graph.hash() == predictor.graph_hash
    assert predictor(np.zeros((32, 128), dtype=np.uint8))["failure"] is None
