"""Frozen-population lineage/output-contract checks, not recognition benchmarks."""
import json
from pathlib import Path
import numpy as np
import pytest
from cellforge.ir.graph import Graph, canonical_json
from cellforge.runtime.vision import OCRPredictor
import hashlib

ROOT = Path(__file__).resolve().parents[1]
POPULATION = ROOT/'runs/parallel-vision/alphabet-crossovers'


@pytest.mark.parametrize('name',['b_glyph_a','b_glyph_oneshot','a_glyph_oneshot'])
def test_frozen_extended_child_preserves_donor_digit_and_alignment_parameters(name):
    parent = Graph.load(ROOT/'runs/parallel-vision/breeding/population'/name)
    child = Graph.load(POPULATION/'population'/name)
    proposal = json.loads((POPULATION/'proposals'/(name+'.json')).read_text())
    assert parent.hash() == proposal['parent_hash']
    assert child.hash() == proposal['child_hash']
    assert proposal['contract_hash'] == hashlib.sha256(canonical_json(proposal['contract'])).hexdigest()
    assert set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:;-!?/').issubset(child.meta['vocab'])
    keys = ('tmpl.W1','tmpl.b1') if name.startswith('b_') else ('glyph.W1','glyph.b1')
    for key in keys:
        assert np.array_equal(parent.params[key][:10],child.params[key][:10])
    for key in parent.params:
        if key not in keys:
            assert np.array_equal(parent.params[key],child.params[key])
    observation = OCRPredictor(child)(np.zeros((32,128),dtype=np.uint8))
    assert observation['failure'] is None
    assert observation['text'] == ''


def test_identical_authored_alphabet_head_and_vocab_across_family():
    children = [Graph.load(POPULATION/'population'/name)
                for name in ['b_glyph_a','b_glyph_oneshot','a_glyph_oneshot']]
    assert children[0].meta['vocab'] == children[1].meta['vocab'] == children[2].meta['vocab']
    for suffix in ('W1','b1'):
        base = children[0].params['tmpl.'+suffix][10:-1]
        assert np.array_equal(base,children[1].params['tmpl.'+suffix][10:-1])
        assert np.array_equal(base,children[2].params['glyph.'+suffix][10:])
