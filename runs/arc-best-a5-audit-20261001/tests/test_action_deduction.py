import importlib.util
import json
from pathlib import Path
from cellforge.runtime.lm import generate
from cellforge.runtime.specialists import parse_action
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.runtime.lm import score_texts
import numpy as np

spec=importlib.util.spec_from_file_location('deduction_builder',Path(__file__).parents[1]/'scripts/build_action_deduction.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def action(graph,question,observation=None):
    transcript=[{'role':'system','content':'schema Fact wrong. If wrong then hidden. Is hidden entailed?'},
                {'role':'user','content':question}]
    if observation is not None:transcript.append({'role':'tool','content':observation})
    text,info=generate(graph,1,prompts=[json.dumps(transcript,separators=(',',':'))+'\nAction:'],max_len=512)
    assert info['failure']==[None]
    return parse_action(text[0])


def test_novel_atom_names_clause_order_and_query_change_call():
    g=builder.build_deduction_action()
    for question in ['Fact seed_x. If seed_x then next_7. If next_7 then goal. Is goal entailed?',
                     'If next_7 then goal. Fact seed_x. If seed_x then next_7. Is goal entailed?']:
        a=action(g,question)
        assert a['type']=='call' and a['name']=='deduction'
        assert a['arguments']['facts']==['seed_x']
        assert {tuple(r['if'])+(r['then'],) for r in a['arguments']['rules']}=={('seed_x','next_7'),('next_7','goal')}
        assert a['arguments']['query']=='goal'


def test_empty_rule_fact_ablation_and_actual_tool_result():
    g=builder.build_deduction_action()
    assert action(g,'Fact novel. Is novel entailed?')['arguments']=={'facts':['novel'],'rules':[],'query':'novel'}
    assert action(g,'If novel then goal. Is goal entailed?')['arguments']=={'facts':[],'rules':[{'if':['novel'],'then':'goal'}],'query':'goal'}
    for status,value in [('entailed',True),('not_proven',False)]:
        actual=action(g,'Fact novel. Is novel entailed?',{'status':'ok','result':{'status':status,'entailed':value}})
        assert actual=={'type':'respond','text':status}


def test_invalid_tool_status_never_becomes_entailed():
    g=builder.build_deduction_action()
    for observation in [{'status':'invalid_action','result':{'status':'entailed','entailed':True}},
                        {'status':'ok','result':{'status':'entailed','entailed':False}},
                        {'status':'observation_withheld'}]:
        assert action(g,'Fact novel. Is novel entailed?',observation)['type']=='call'


def test_composed_micro_preserves_language_and_ocr_paths():
    parent=Graph.load(Path(__file__).parents[1]/'runs/action-language/composed-micro-preserved-artifact')
    g=builder.compose_deduction_action(parent,builder.build_deduction_action())
    assert check_profile(g,'ladder-micro/1')['dispatches']==1
    a,_=score_texts(parent,['Hello 17!'])
    b,_=score_texts(g,['Hello 17!'])
    for k in ('bits','correct','failed'):np.testing.assert_array_equal(a[k],b[k])
    assert action(g,'OCR[fresh17]')=={'type':'call','name':'ocr','arguments':{'image_id':'fresh17'}}
