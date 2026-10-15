"""Frozen authored OCR as image-conditioned text observations, not image embeddings."""
from __future__ import annotations

import hashlib
import numpy as np

from ..ir.graph import Graph, GraphError
from .decoder import decode
from .text_codec import BYTE_CODEC, BYTE_VOCAB, encode_content


class OCRPredictor:
    """Single bright-ink uint8 raster -> authored OCR text and language byte IDs.

    No resizing, threshold repair, recognition outside the graph, or label input.
    The exported graph's vocabulary bounds recognition (legacy exports: digits).
    """

    def __init__(self, graph: Graph):
        graph.validate()
        contract = graph.meta.get("pixel_contract", {})
        if (contract.get("height"), contract.get("width"), contract.get("ink"),
                contract.get("normalize")) != (32, 128, "bright", "uint8/255"):
            raise GraphError("OCR requires exported bright-ink uint8/255 32x128 contract")
        if not {"encode", "step"}.issubset(graph.entries):
            raise GraphError("OCR graph needs encode and step entries")
        self.graph = graph.copy()
        self.graph_hash = self.graph.hash()
        self.last_exec = None

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray) or image.shape != (32, 128) or image.dtype != np.uint8:
            raise GraphError("OCR image must be a single uint8 array of shape (32, 128)")
        image_hash = hashlib.sha256(image.tobytes(order="C")).hexdigest()
        predictions, info = decode(self.graph, image[None].astype(np.float64) / 255.0)
        text = predictions[0]
        self.last_exec = info["exec"]
        return {
            "text": text,
            "token_ids": encode_content(BYTE_VOCAB, text, BYTE_CODEC) if text is not None else [],
            "token_codec": BYTE_CODEC,
            "observation_kind": "image-conditioned-ocr-text",
            "graph_hash": self.graph_hash,
            "image_sha256": image_hash,
            "failure": info["failure"][0],
            "exec": self.last_exec,
        }
