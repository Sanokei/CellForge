"""Labeled bootstrap artifact: emits EOS immediately (CER = 1 by construction).

This is the explicitly recorded starting point; every later parameter or
topology change must trace to an accepted author proposal.
"""
from __future__ import annotations

import numpy as np

from ..ir.graph import Graph
from ..runtime.decoder import BOS, EOS, PAD

BOOTSTRAP_ID = "bootstrap/eos-constant/1"


def vocab_for(charset: str) -> list[str]:
    return [PAD, BOS, EOS] + list(charset)


def bootstrap(charset: str = "0123456789", max_len: int = 12) -> Graph:
    vocab = vocab_for(charset)
    eos_logits = np.zeros(len(vocab))
    eos_logits[vocab.index(EOS)] = 1.0
    g = Graph()
    g.params["boot_logits"] = eos_logits
    g.defs["encode_root"] = {
        "kind": "bundle", "inputs": ["image"],
        "nodes": [{"id": "ink", "use": "op:mean", "in": {"x": "$image"}, "args": {"axis": 0}}],
        "outputs": {"colink": "ink"},
    }
    g.defs["step_root"] = {
        "kind": "bundle", "inputs": ["colink", "step", "prev", "state"],
        "nodes": [{"id": "logits", "use": "param:boot_logits"},
                  {"id": "lg", "use": "op:broadcast_to", "in": {"x": "logits", "like": "$prev"}}],
        "outputs": {"logits": "lg", "state": "$state"},
    }
    g.entries = {"encode": "encode_root", "step": "step_root"}
    g.meta = {"vocab": vocab, "state_dim": 0, "max_len": max_len,
              "pixel_contract": {"height": 32, "width": 128, "normalize": "uint8/255", "ink": "bright"},
              "primitive_version": "cellforge-ops/1", "bootstrap": BOOTSTRAP_ID}
    return g


LM_BOOTSTRAP_ID = "bootstrap/uniform-logits/1"


def bootstrap_lm(charset: str, ctx_len: int = 8) -> Graph:
    """Uniform next-token logits over the vocabulary (bpc = log2 V by construction)."""
    vocab = vocab_for(charset)
    g = Graph()
    g.params["boot_logits"] = np.zeros(len(vocab))
    g.defs["step_root"] = {
        "kind": "bundle", "inputs": ["ctx", "prev", "step", "state"],
        "nodes": [{"id": "logits", "use": "param:boot_logits"},
                  {"id": "lg", "use": "op:broadcast_to", "in": {"x": "logits", "like": "$prev"}}],
        "outputs": {"logits": "lg", "state": "$state"},
    }
    g.entries = {"step": "step_root"}
    g.meta = {"task": "lm", "vocab": vocab, "state_dim": 0, "ctx_len": ctx_len,
              "primitive_version": "cellforge-ops/1", "bootstrap": LM_BOOTSTRAP_ID}
    return g
