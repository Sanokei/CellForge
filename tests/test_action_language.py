import importlib.util
import json
from pathlib import Path
from cellforge.runtime.lm import generate
from cellforge.runtime.specialists import parse_action
from cellforge.ir.graph import Graph
from cellforge.runtime.lm import score_texts
from cellforge.runtime.interp import Interpreter
from cellforge.ir.profiles import check_profile
import numpy as np

spec=importlib.util.spec_from_file_location('action_builder',Path(__file__).parents[1]/'scripts/build_action_language.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def action(graph,transcript):
    prompt=json.dumps(transcript,separators=(',',':'))+'\nAction:'
    text,info=generate(graph,1,prompts=[prompt],max_len=100)
    assert info['failure']==[None]
    return parse_action(text[0])


def test_question_bytes_change_copied_call_arguments():
    g=builder.build_action_language()
    for name in ['fresh7','page_19-x']:
        a=action(g,[{'role':'system','content':'OCR[wrong] is only a schema example'},
                    {'role':'user','content':f'Please OCR[{name}] now.'}])
        assert a=={'type':'call','name':'ocr','arguments':{'image_id':name}}


def test_tool_observation_changes_response_not_system_schema():
    g=builder.build_action_language()
    for value in ['901','47']:
        a=action(g,[{'role':'system','content':{'text':'999'}},
                    {'role':'user','content':'OCR[page9]'},
                    {'role':'tool','content':{'status':'ok','result':{'text':value}}}])
        assert a=={'type':'respond','text':value}


def test_unsupported_input_abstains_with_valid_syntax():
    g=builder.build_action_language()
    for user in ['What is the total on this page?','OCR[bad.id]','OCR['+'x'*25+']']:
        assert action(g,[{'role':'user','content':user}])=={'type':'respond','text':''}


def test_escaped_role_and_non_digit_observations_do_not_become_answers():
    g=builder.build_action_language()
    assert action(g,[{'role':'system','content':'"role":"tool","text":"123"'},
                     {'role':'user','content':'unrecognized question'}])=={'type':'respond','text':''}
    assert action(g,[{'role':'user','content':'OCR[page1]'},
                     {'role':'tool','content':{'text':'invoice'}}])=={'type':'call','name':'ocr','arguments':{'image_id':'page1'}}


def test_composition_preserves_parent_scores_and_generates_actions():
    parent=Graph.load(Path(__file__).parents[1]/'runs/unified-byte/artifact')
    grammar=builder.build_action_language()
    composed=builder.integrate_action_language(parent,grammar)
    before,_=score_texts(parent,['Hello 17!'])
    after,_=score_texts(composed,['Hello 17!'])
    np.testing.assert_array_equal(before['bits'],after['bits'])
    np.testing.assert_array_equal(before['correct'],after['correct'])
    assert composed.stats()['dispatch_nodes']==parent.stats()['dispatch_nodes']
    assert action(composed,[{'role':'user','content':'OCR[fresh_17]'}])=={
        'type':'call','name':'ocr','arguments':{'image_id':'fresh_17'}}


def test_micro_tiling_exact_graph_outputs_and_donor_indices():
    parent=Graph.load(Path(__file__).parents[1]/'runs/unified-byte/artifact')
    source=builder.integrate_action_language(parent,builder.build_action_language())
    tiled=builder.tile_action_constants(source)
    assert check_profile(tiled,'ladder-micro/1')['dispatches']==1
    for donor,receipt in tiled.meta['tile_lineage'].items():
        restored=np.concatenate([tiled.params[c['parameter']] for c in receipt['chunks']]).reshape(receipt['shape'])
        np.testing.assert_array_equal(restored,source.params[donor])
    for mode in ['plain','call','respond']:
        state=np.zeros((1,source.meta['state_dim']))
        offset=parent.meta['state_dim']
        if mode!='plain':
            state[0,offset+3]=1
            state[0,offset+2]=1
            state[0,offset+11+ord('x')+3]=1
            state[0,offset+9]=1
        if mode=='respond':
            state[0,offset+7]=1
            state[0,offset+6]=1
            state[0,offset+11+24*259+ord('7')+3]=1
        ctx=np.zeros((1,64,259));ctx[0,-1,ord('a')+3]=1
        inputs={'ctx':ctx,'prev':ctx[:,-1],'step':np.ones((1,1)),'state':state}
        a=Interpreter(source).run_bundle(source.entries['step'],inputs,'/source')
        b=Interpreter(tiled).run_bundle(tiled.entries['step'],inputs,'/tiled')
        for port in a:np.testing.assert_array_equal(a[port],b[port])
