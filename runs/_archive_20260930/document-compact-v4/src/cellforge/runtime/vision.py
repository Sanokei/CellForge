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

    def predict_batch(self, images: np.ndarray):
        """Decode independent crops together; execution costs belong to the batch.

        Each crop starts with its own zero recurrent state. No cross-crop context
        or target information is supplied. Batch equivalence requires validation
        for each candidate graph before using this execution path in evaluation.
        """
        if (not isinstance(images, np.ndarray) or images.dtype != np.uint8 or
                images.ndim != 3 or images.shape[1:] != (32, 128) or not len(images)):
            raise GraphError("OCR batch must be a nonempty uint8 array of shape (B, 32, 128)")
        if self.graph.hash() != self.graph_hash:
            raise GraphError("OCR graph changed after initialization")
        predictions, info = decode(self.graph, images.astype(np.float64) / 255.0)
        self.last_exec = info["exec"]
        observations = []
        for image, text, failure in zip(images, predictions, info["failure"]):
            observations.append({
                "text": text,
                "token_ids": encode_content(BYTE_VOCAB, text, BYTE_CODEC) if text is not None else [],
                "token_codec": BYTE_CODEC,
                "observation_kind": "image-conditioned-ocr-text",
                "graph_hash": self.graph_hash,
                "image_sha256": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
                "failure": failure,
            })
        return observations, {"exec": self.last_exec, "exec_scope": "whole-batch", "batch_size": len(images)}
