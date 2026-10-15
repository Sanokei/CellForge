"""Transport/isolation tests; constant answer fixture is not document reasoning."""
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import ImageFont
from cellforge.data.render import render_one
from cellforge.ir.graph import Graph, GraphError
from cellforge.runtime.text_codec import BYTE_CODEC, BYTE_VOCAB
from cellforge.runtime.document import DocumentPredictor, DocumentBudget

ROOT = Path(__file__).resolve().parents[1]


def system(budget=None):
    bias = np.zeros(259)
    bias[2] = 1
    lm = Graph(defs={'head': {'kind':'cell','act':'identity','params':{'W1':'W','b1':'b'}},
                     'step': {'kind':'bundle','inputs':['ctx','prev','step','state'],
                              'nodes':[{'id':'head','use':'head','in':{'x':'$prev'}}],
                              'outputs':{'logits':'head.y','state':'$state'}}},
               params={'W':np.zeros((259,259)), 'b':bias}, entries={'step':'step'},
               meta={'task':'lm','text_codec':BYTE_CODEC,'vocab':BYTE_VOCAB,'ctx_len':2,'state_dim':0})
    return DocumentPredictor(lm, Graph.load(ROOT/'runs/p1/exports/r1-iter-b'), budget)


def image():
    return 255-render_one('123', ImageFont.truetype(str(ROOT/'fonts/DejaVuSansMono.ttf'),20),6,0)


def test_actual_graph_transport_and_page_ocr():
    result = system()({'pages':[image()], 'question':'Which digits?'}, keep_trace=True)
    assert result['status'] == 'responded'
    assert result['answer'] == ''  # constant-EOS fixture, not evidence of answering
    assert result['observations'][0]['observation']['text'] == '123'
    assert json.loads(result['language_prompt'])['question'] == 'Which digits?'
    assert result['crop_count'] == 1


def test_order_padding_and_withheld_observations():
    page = np.full((33,129),255,dtype=np.uint8)
    result = system()({'pages':[page], 'question':'q'}, observation_mode='withheld', keep_trace=True)
    assert [o['crop_xy'] for o in result['observations']] == [[0,0],[128,0],[0,32],[128,32]]
    assert result['observations'][-1]['source_hw'] == [1,1]
    assert all(o['observation']['text'] == '' for o in result['observations'])
    assert json.loads(result['language_prompt'])['observations'] == [{'status':'observations_withheld'}]


@pytest.mark.parametrize('extra', ['answer','evidence_pages','target','labels'])
def test_no_groundtruth_or_evidence_inputs(extra):
    with pytest.raises(ValueError,match='pages and question only'):
        system()({'pages':[image()], 'question':'q',extra:'hidden'})


def test_crop_budget_rejects_whole_input_without_partial_page_selection():
    result = system(DocumentBudget(max_crops=1))({'pages':[np.zeros((33,128),dtype=np.uint8)],'question':'q'})
    assert result['status'] == 'crop_budget_exhausted'
    assert result['required_crops'] == 2
    assert result['crop_count'] == 0


def test_context_overflow_cannot_generate_answer():
    result = system(DocumentBudget(max_context_bytes=1))({'pages':[image()],'question':'q'})
    assert result['status'] == 'context_budget_exhausted'
    assert result['answer'] is None


def test_page_contract_rejects_normalized_or_rgb():
    for page in [np.zeros((32,128)),np.zeros((32,128,3),dtype=np.uint8)]:
        with pytest.raises(ValueError, match='grayscale raster'):
            system()({'pages':[page],'question':'q'})


def test_frozen_components_checked_before_ocr_or_budget():
    predictor = system(DocumentBudget(max_crops=1))
    predictor.ocr.graph.meta['max_len'] = 0
    with pytest.raises(GraphError, match='changed'):
        predictor({'pages':[np.zeros((64,128),dtype=np.uint8)],'question':'q'})


def test_seeded_shuffle_keeps_coordinates_and_moves_content():
    font = ImageFont.truetype(str(ROOT/'fonts/DejaVuSansMono.ttf'),20)
    pages = [255-render_one(s,font,6,0) for s in ['123','42','7']]
    result = system()({'pages':pages,'question':'q'}, observation_mode='shuffled',shuffle_seed=3,keep_trace=True)
    prompt = json.loads(result['language_prompt'])
    assert [o['page_index'] for o in prompt['observations']] == [0,1,2]
    assert [o['observation']['text'] for o in prompt['observations']] == [
        result['observations'][i]['observation']['text'] for i in result['shuffle_permutation']]
    assert result['shuffle_permutation'] != [0,1,2]
    assert result['shuffle_seed'] == 3


def test_batch_size_does_not_change_language_input_or_recognition():
    page = np.full((33, 129), 255, dtype=np.uint8)
    inputs = {'pages': [page], 'question': 'Which digits?'}
    serial = system(DocumentBudget(ocr_batch_size=1))(inputs, keep_trace=True)
    batched = system(DocumentBudget(ocr_batch_size=3))(inputs, keep_trace=True)
    assert serial['observations'] == batched['observations']
    assert serial['language_prompt'] == batched['language_prompt']
    assert serial['answer'] == batched['answer']
    assert len(serial['ocr_batches']) == 4
    assert [batch['batch_size'] for batch in batched['ocr_batches']] == [3, 1]
    assert 'node_execs' not in batched['language_prompt']


def test_compact_format_keeps_every_crop_and_question_without_provenance_tokens():
    predictor = system()
    inputs = {'pages': [np.full((33, 129), 255, dtype=np.uint8)], 'question': 'What is shown?'}
    full = predictor(inputs, keep_trace=True)
    compact_predictor = DocumentPredictor(predictor.language_graph, predictor.ocr.graph,
                                          observation_format='compact')
    compact = compact_predictor(inputs, keep_trace=True)
    payload = json.loads(compact['language_prompt'])
    assert list(payload)[-1] == 'question'
    assert payload['question'] == inputs['question']
    assert len(payload['observations']) == full['crop_count'] == 4
    assert compact['observations'] == full['observations']
    for row, original in zip(payload['observations'], full['observations']):
        assert row['text'] == original['observation']['text']
        assert row['failure'] == original['observation']['failure']
        assert row['crop_xy'] == original['crop_xy']
        assert 'token_ids' not in row and 'image_sha256' not in row
    assert compact['context_bytes'] < full['context_bytes']
    assert compact['observation_schema'] == 'document-ocr-observation/3'


def test_compiled_document_prefill_preserves_answer_and_observations():
    predictor = system()
    inputs = {'pages': [image()], 'question': 'Which digits?'}
    baseline = predictor(inputs, keep_trace=True)
    accelerated = DocumentPredictor(predictor.language_graph, predictor.ocr.graph,
                                    backend='compiled', state_only_prefill=True)(inputs, keep_trace=True)
    assert accelerated['answer'] == baseline['answer']
    assert accelerated['status'] == baseline['status']
    assert accelerated['observations'] == baseline['observations']
    assert accelerated['language_prompt'] == baseline['language_prompt']
    assert accelerated['language_backend'] == 'compiled'
