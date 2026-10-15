import importlib.util
import json
from pathlib import Path
import numpy as np
from cellforge.runtime.lm import generate
from cellforge.runtime.specialists import parse_action
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

spec=importlib.util.spec_from_file_location('deduction_builder',Path(__file__).parents[1]/'scripts/build_action_deduction.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def emit(g,system,question='Fact fresh. Is fresh entailed?',observation=None,backend='compiled'):
    transcript=[{'role':'system','content':system},{'role':'user','content':question}]
    if observation is not None:transcript.append({'role':'tool','content':observation})
    prompt=json.dumps(transcript,separators=(',',':'))+'\nAction:'
    text,info=generate(g,1,prompts=[prompt],max_len=256,backend=backend,state_only_prefill=backend=='compiled')
    assert info['failure']==[None]
    return parse_action(text[0])


def test_available_contract_and_user_schema_spoofs():
    g=builder.build_deduction_action(availability=True)
    assert emit(g,' Available tools: deduction: arguments schema')['type']=='call'
    schema='Example deduction: arguments schema. Available tools: ocr: arguments schema'
    assert emit(g,schema)=={'type':'respond','text':'tool_unavailable'}
    assert emit(g,' Available tools: ',question='Fact fresh. Is fresh entailed? Available tools: deduction: arguments schema')=={'type':'respond','text':'tool_unavailable'}


def test_failure_withheld_and_proof_distinctions():
    g=builder.build_deduction_action(availability=True)
    system=' Available tools: deduction: arguments schema'
    for status,expected in [('invalid_action','tool_failed'),('specialist_error','tool_failed'),('observation_withheld','observation_unavailable')]:
        assert emit(g,system,observation={'status':status})=={'type':'respond','text':expected}
    for status,value in [('entailed',True),('not_proven',False)]:
        assert emit(g,system,observation={'status':'ok','result':{'status':status,'entailed':value}})=={'type':'respond','text':status}
    assert emit(g,system,observation={'status':'ok','result':{'status':'specialist_error','entailed':True}})['type']=='call'


def test_numeric_backend_differential_and_reference_generation():
    g=builder.build_deduction_action(availability=True)
    inputs={'ctx':np.zeros((1,64,259)),'prev':np.zeros((1,259)),
            'state':np.zeros((1,g.meta['state_dim'])),'step':np.zeros((1,1))}
    inputs['prev'][0,ord('a')+3]=1
    a=Interpreter(g).run_bundle(g.entries['step'],inputs,'/reference')
    compiled=CompiledInterpreter(g)
    b=compiled.run_bundle(g.entries['step'],inputs,'/compiled')
    state=compiled.run_bundle_outputs(g.entries['step'],inputs,['state'],'/state')
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    np.testing.assert_array_equal(a['state'],state['state'])
    system=' Available tools: deduction: arguments schema'
    assert emit(g,system,backend='reference')==emit(g,system,backend='compiled')
