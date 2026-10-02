"""Expand identical literal alphabet onto three frozen OCR crossover children.

No ranking, weight fitting, document labels, or known six-word probe selection.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph, canonical_json, array_hash
from cellforge.construction.transactions import apply_ops
from cellforge.governance.ledger import Ledger
from cellforge.runtime.vision import OCRPredictor
from author_alphabet_ocr import build, GLYPHS, patch


def build_table_parent(parent):
    """Keep recipient alignment/digit/presence rows, expand content classes."""
    chars = list(GLYPHS)
    templates = np.stack([patch(char).reshape(-1) for char in chars])
    new_weights = 2*(2*templates-1)/180
    new_bias = (180-2*templates.sum(1))/180
    weights = np.concatenate([parent.params['tmpl.W1'][:10],new_weights,parent.params['tmpl.W1'][10:]])
    bias = np.concatenate([parent.params['tmpl.b1'][:10],new_bias,parent.params['tmpl.b1'][10:]])
    content = 10+len(chars)
    classes = content+1
    vocab_width = content+3
    ops = [{'op':'set_param','name':'tmpl.W1','value':weights.tolist()},
           {'op':'set_param','name':'tmpl.b1','value':bias.tolist()}]
    reader = copy.deepcopy(parent.defs['reader'])
    for node in reader['nodes']:
        if node['id'] == 'c0d':
            node['args']['stop'] = content
        elif node['id'] == 'allf':
            node['args']['shape'] = [4,8*classes]
        elif node['id'] == 'selr':
            node['args']['shape'] = [8,classes]
    table = copy.deepcopy(parent.defs['to_table'])
    for node in table['nodes']:
        if node['id'] == 'dig':
            node['args']['stop'] = content
        elif node['id'] == 'pres':
            node['args'].update({'start':content,'stop':classes})
    # Pure graph-owned suffix presence: preserve the original ink detector,
    # but a blank interior cell may emit space while later ink exists.
    table['nodes'].extend([
        {'id':'future_map','use':'param:alphabet.future','in':{}},
        {'id':'future','use':'op:matmul','in':{'a':'future_map','b':'pr'}},
        {'id':'future_bias','use':'op:bias','in':{'x':'future'},'args':{'c':-0.5}},
        {'id':'future_on','use':'op:step','in':{'x':'future_bias'}},
        {'id':'future_penalty','use':'op:scale','in':{'x':'future_on'},'args':{'c':-1000.0}},
        {'id':'dynamic_eos','use':'op:bias','in':{'x':'future_penalty'},'args':{'c':500.0}},
        {'id':'tail_eos','use':'op:slice','in':{'x':'eos'},'args':{'axis':0,'start':8,'stop':12}},
        {'id':'all_eos','use':'op:concat','in':{'x0':'dynamic_eos','x1':'tail_eos'},'args':{'axis':0}},
    ])
    # Move final concat after its new EOS dependencies (typed DAG order).
    concat = next(node for node in table['nodes'] if node['id']=='tab')
    table['nodes'].remove(concat)
    concat['in']['x1'] = 'all_eos'
    table['nodes'].append(concat)
    step = copy.deepcopy(parent.defs['step_root'])
    for node in step['nodes']:
        if node['id']=='lgr':
            node['args']['shape'] = [vocab_width]
    ops += [{'op':'set_param','name':'alphabet.future','value':[[float(j>=i) for j in range(8)] for i in range(8)]},
            {'op':'put_def','name':'reader','def':reader},
            {'op':'put_def','name':'to_table','def':table},
            {'op':'put_def','name':'step_root','def':step}]
    child,_ = apply_ops(parent,ops)
    child.meta['vocab'] = parent.meta['vocab']+chars
    child.meta['alphabet_hypothesis'] = 'direct-authored5x7-strokes/1'
    child.validate()
    return child,ops


def main():
    root = Path('runs/parallel-vision')
    output = root/'alphabet-crossovers'
    ledger = Ledger(output/'ledger.jsonl')
    rows = []
    for name in ['b_glyph_a','b_glyph_oneshot','a_glyph_oneshot']:
        parent = Graph.load(root/'breeding/population'/name)
        if name.startswith('b_'):
            child,ops = build_table_parent(parent)
            assert np.array_equal(child.params['tmpl.W1'][:10],parent.params['tmpl.W1'][:10])
            assert np.array_equal(child.params['tmpl.b1'][:10],parent.params['tmpl.b1'][:10])
            assert np.array_equal(child.params['tmpl.W1'][-1],parent.params['tmpl.W1'][-1])
        else:
            child,ops = build(parent)
            assert np.array_equal(child.params['glyph.W1'][:10],parent.params['glyph.W1'][:10])
            assert np.array_equal(child.params['glyph.b1'][:10],parent.params['glyph.b1'][:10])
        unchanged = {k:array_hash(v) for k,v in parent.params.items() if k not in ('glyph.W1','glyph.b1','tmpl.W1','tmpl.b1')}
        assert all(array_hash(child.params[k])==v for k,v in unchanged.items())
        artifact = output/'population'/name
        child.save(artifact)
        contract = {'protocol':'alphabet-expanded-ocr-family/1','pixel_contract':child.meta['pixel_contract'],
                    'vocab':child.meta['vocab'],'entries':['encode','step'],'observation_codec':'utf8-bytes/1',
                    'capability_claim':'authored alphabet hypothesis only; no document-development winner selected'}
        original = json.loads((root/'breeding/proposals'/(name+'.json')).read_text())
        proposal = {'protocol':'ocr-alphabet-crossover-extension/1','parent_hash':parent.hash(),'child_hash':child.hash(),
                    'original_crossover':original,'alphabet_source_sha256':hashlib.sha256(Path('scripts/author_alphabet_ocr.py').read_bytes()).hexdigest(),
                    'alphabet_strokes_sha256':hashlib.sha256(canonical_json(GLYPHS)).hexdigest(),
                    'contract':contract,'contract_hash':hashlib.sha256(canonical_json(contract)).hexdigest(),
                    'unchanged_recipient_parameter_hashes':unchanged,'digit_rows_preserved':True,'ops':ops,
                    'metadata_changes':child.meta,'no_ranking_or_final_data':True}
        (output/'proposals').mkdir(parents=True,exist_ok=True)
        (output/'proposals'/(name+'.json')).write_text(json.dumps(proposal,indent=2)+'\n')
        ledger.append('alphabet_crossover_extension',proposal,'ocr-author')
        observation = OCRPredictor(child)(np.zeros((32,128),dtype=np.uint8))
        fixture = {'fixture':'allzero bright-ink raster','purpose':'execution/schema only; not accuracy selection',
                   'result':observation,'digit_rows_preserved':True}
        (output/'fixtures').mkdir(parents=True,exist_ok=True)
        (output/'fixtures'/(name+'.json')).write_text(json.dumps(fixture,indent=2)+'\n')
        ledger.append('alphabet_schema_fixture',fixture,'ocr-author')
        rows.append({'id':name,'hash':child.hash(),'artifact':str(artifact),'contract_hash':proposal['contract_hash']})
    manifest = {'protocol':'alphabet-expanded-ocr-family/1','population':rows,'selection':'none; pending matched document development',
                'numeric_source':'inherited digit parameters plus identical authorliteral alphabet formulas; no1D arrays, font fitting or pretrained weights'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':
    main()
