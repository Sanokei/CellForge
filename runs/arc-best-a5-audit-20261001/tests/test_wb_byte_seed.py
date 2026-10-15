from pathlib import Path
import hashlib
import numpy as np
import pytest
from cellforge.ir.graph import Graph
from cellforge.construction.wb_byte_seed import wb_byte_seed, reference_probabilities, context_key
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter
from cellforge.runtime.lm import encode_text
from cellforge.runtime.text_codec import BYTE_CODEC, BYTE_VOCAB
from cellforge.runtime.ops import op_csr_lookup


def fixture():
    founder = Graph.load(Path('runs/unified-byte/artifact'))
    texts = ['abracadabra abc', 'abracadabra xyz', 'café 中文']
    partition = {'partition':'dev_train_selection','source_split':'dev_train',
                 'corpus_sha256':hashlib.sha256('fixture'.encode()).hexdigest(),
                 'items':[{'family':hashlib.sha256(t.encode()).hexdigest(),'text':t} for t in texts]}
    return wb_byte_seed(founder,partition)


def test_exact_context_key_has_no_leading_zero_ambiguity():
    assert context_key(b'') == 0
    assert context_key(b'\x00') == 1
    assert context_key(b'\x00\x00') == 258
    assert context_key(bytes([255])*6) < 2**53


def test_wb_export_matches_all_byte_probabilities_and_resets():
    seed, provenance, counts = fixture()
    assert provenance['artifact_class'] == 'fitted' and provenance['residual_weight'] == 0
    reference, compiled = Interpreter(seed), CompiledInterpreter(seed)
    for text in ['abracadabra', 'unseen 🦊', 'a']:
        raw = text.encode(); tokens = encode_text(BYTE_VOCAB,text,BYTE_CODEC)
        ctx = np.zeros((1,64,259))
        for t in range(len(raw)+1):
            ctx = np.roll(ctx,-1,axis=1);ctx[:,-1]=0;ctx[0,-1,tokens[t]]=1
            inputs = {'ctx':ctx,'prev':ctx[:,-1],'step':np.array([[float(t)]])}
            expected = reference_probabilities(counts,raw[:t])
            a = reference.run_bundle('cfwb_trunk',inputs)['probs'][0]
            b = compiled.run_bundle('cfwb_trunk',inputs)['probs'][0]
            np.testing.assert_allclose(a,expected,atol=2e-16,rtol=2e-13)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(a.sum(),1.,atol=1e-14)


def test_csr_lookup_shared_and_per_example_tables_missing_keys():
    table = {'keys':np.array([[1.,4.]]),'offsets':np.array([[0.,1.,2.]]),
             'columns':np.array([[0.,2.]]),'values':np.array([[.5,.9]]),
             'query':np.array([[4.],[3.],[1.]])}
    out = op_csr_lookup(table,{'width':3})
    np.testing.assert_array_equal(out,[[0,0,.9],[0,0,0],[.5,0,0]])
    table = {k:np.repeat(v,3,axis=0) if k!='query' else v for k,v in table.items()}
    np.testing.assert_array_equal(op_csr_lookup(table,{'width':3}),out)


def test_wb_rejects_reporting_fit():
    with pytest.raises(ValueError,match='dev_train'):
        wb_byte_seed(Graph.load(Path('runs/unified-byte/artifact')),{'partition':'report'})
