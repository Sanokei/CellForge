"""Exact reference differential evidence, including traces and virtual counters."""
from pathlib import Path
import numpy as np
import pytest
from cellforge.ir.graph import Graph, GraphError
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

ROOT = Path(__file__).resolve().parents[1]


def equal(a,b):
    if isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else:
        assert a==b


def random_graph(seed,k):
    rng=np.random.default_rng(seed)
    defs={}
    params={}
    for i in range(2):
        defs[f'cell{i}']={'kind':'cell','act':'tanh','params':{'W1':f'W{i}','b1':f'b{i}','W2':f'V{i}','b2':f'c{i}'}}
        params.update({f'W{i}':rng.normal(size=(5,4)),f'b{i}':rng.normal(size=5),f'V{i}':rng.normal(size=(4,5)),f'c{i}':rng.normal(size=4)})
        defs[f'branch{i}']={'kind':'bundle','inputs':['x'],'nodes':[{'id':'cell','use':f'cell{i}','in':{'x':'$x'}}],'outputs':{'y':'cell.y'}}
    defs['root']={'kind':'bundle','inputs':['x','scores'],'nodes':[
        {'id':'coord','use':'op:coords','args':{'n':4}},
        {'id':'scale','use':'op:scale','in':{'x':'coord'},'args':{'c':0.125}},
        {'id':'combine','use':'op:add','in':{'a':'$x','b':'scale'}},
        {'id':'route','use':'dispatch','in':{'scores':'$scores','x':'combine'},'args':{'branches':['branch0','branch1'],'k':k}}],
        'outputs':{'value':'route.y','constant':'scale'}}
    return Graph(defs=defs,params=params,entries={'step':'root'},meta={'task':'lm'})


@pytest.mark.parametrize('seed',range(6))
@pytest.mark.parametrize('k',[1,2])
def test_random_nested_cells_dispatch_constants_traces_and_reordered_batches(seed,k):
    graph=random_graph(seed,k)
    reference=Interpreter(graph,trace=True,trace_filter=lambda path:'coord' not in path)
    compiled=CompiledInterpreter(graph,trace=True,trace_filter=lambda path:'coord' not in path)
    rng=np.random.default_rng(seed+19)
    x=rng.normal(size=(4,4))
    scores=np.array([[0.,0.],[1.,2.],[np.nan,1.],[2.,1.]])
    for order in [[0,1,2,3],[3,2,1,0],[1,1,0,3]]:
        inputs={'x':x[order].copy(),'scores':scores[order].copy()}
        equal(reference.run_bundle('root',inputs,'/root'),compiled.run_bundle('root',inputs,'/root'))
        equal(reference.stats.as_dict(),compiled.stats.as_dict())
        equal(reference.trace,compiled.trace)
        inputs['x'][:]=99  # must not contaminate any later constant reuse
    out=compiled.run_bundle('root',{'x':x,'scores':scores})
    with pytest.raises(ValueError):
        out['constant'][0,0]=999
    graph.params['W0'][:]=0  # compiler owns a separate parameter snapshot
    compiled.assert_frozen()
    compiled.g.meta['changed']=True
    with pytest.raises(GraphError,match='snapshot changed'):
        compiled.assert_frozen()


@pytest.mark.parametrize('artifact',['runs/p1/exports/r1-iter-a','runs/p1/exports/r1-iter-b',
    'runs/parallel-vision/alphabet-crossovers/population/b_glyph_a',
    'runs/parallel-vision/alphabet-crossovers/population/b_glyph_oneshot',
    'runs/parallel-vision/alphabet-crossovers/population/a_glyph_oneshot'])
def test_actual_ocr_decode_predictions_all_intermediate_arrays_and_stats(artifact,monkeypatch):
    from cellforge.runtime import decoder
    graph=Graph.load(ROOT/artifact)
    images=np.random.default_rng(77).random((2,32,128))
    original=decoder.decode(graph,images,trace=True)
    monkeypatch.setattr(decoder,'Interpreter',CompiledInterpreter)
    accelerated=decoder.decode(graph,images,trace=True)
    equal(original,accelerated)


@pytest.mark.parametrize('artifact',['runs/unified-byte/artifact','runs/unified-byte-memory/artifact'])
def test_actual_language_prompt_state_routes_and_generated_tokens(artifact,monkeypatch):
    from cellforge.runtime import lm
    graph=Graph.load(ROOT/artifact)
    original=lm.generate(graph,1,prompts=['Question: 42?\nOCR: 123'],max_len=4)
    monkeypatch.setattr(lm,'Interpreter',CompiledInterpreter)
    equal(original,lm.generate(graph,1,prompts=['Question: 42?\nOCR: 123'],max_len=4))


def test_actual_deduction_chain_and_nonproof_differential(monkeypatch):
    from scripts.build_deduction_specialist import build_deduction
    from cellforge.runtime import deduction
    graph=build_deduction()
    cases=[(['a'],[{'if':['a'],'then':'b'},{'if':['b'],'then':'c'}],'c'),
           ([],[{'if':['a'],'then':'b'},{'if':['b'],'then':'a'}],'b')]
    original=[deduction.HornDeductionPredictor(graph)(*case) for case in cases]
    monkeypatch.setattr(deduction,'Interpreter',CompiledInterpreter)
    equal(original,[deduction.HornDeductionPredictor(graph)(*case) for case in cases])


@pytest.mark.parametrize('artifact',['runs/unified-byte/artifact','runs/unified-byte-memory/artifact'])
def test_state_only_prefill_matches_every_state_and_final_generation(artifact):
    graph=Graph.load(ROOT/artifact)
    reference=Interpreter(graph)
    compiled=CompiledInterpreter(graph)
    vocab=len(graph.meta['vocab'])
    ctx=np.zeros((1,graph.meta['ctx_len'],vocab))
    state=np.zeros((1,graph.meta.get('state_dim',0)))
    tokens=[1]+[byte+3 for byte in ('Question: é?\nOrdered OCR: abc 42\n'*4).encode('utf-8')]
    for t,token in enumerate(tokens):
        ctx=np.roll(ctx,-1,axis=1)
        ctx[:,-1]=0
        ctx[:,-1,token]=1
        inputs={'ctx':ctx,'prev':ctx[:,-1],'state':state,'step':np.array([[float(t)]])}
        original=reference.run_bundle(graph.entries['step'],inputs,'/gen')
        accelerated=(compiled.run_bundle_outputs(graph.entries['step'],inputs,['state'],'/gen')
                     if t+1<len(tokens) else compiled.run_bundle(graph.entries['step'],inputs,'/gen'))
        equal(original['state'],accelerated['state'])
        state=np.where(np.isfinite(original['state']),original['state'],0)
    equal(original,accelerated)
    # Continue generated tokens: both models start from identical state/context.
    for index in range(4):
        logits=original['logits'].reshape(1,-1).copy()
        logits[:,[0,1]]=-np.inf
        token=int(logits.argmax(-1)[0])
        ctx=np.roll(ctx,-1,axis=1)
        ctx[:,-1]=0
        ctx[:,-1,token]=1
        inputs={'ctx':ctx,'prev':ctx[:,-1],'state':state,'step':np.array([[float(len(tokens)+index)]])}
        original=reference.run_bundle(graph.entries['step'],inputs,'/gen')
        accelerated=compiled.run_bundle(graph.entries['step'],inputs,'/gen')
        equal(original,accelerated)
        state=np.where(np.isfinite(original['state']),original['state'],0)
    assert compiled.stats.node_execs<reference.stats.node_execs


@pytest.mark.parametrize('seed',range(3))
def test_demand_does_not_prune_logits_when_state_depends_on_them(seed):
    graph=random_graph(seed,2)
    graph.defs['root']['outputs']={'state':'route.y','logits':'combine'}
    inputs={'x':np.random.default_rng(seed).normal(size=(3,4)),'scores':np.array([[1,2],[2,1],[0,0.]])}
    original=Interpreter(graph).run_bundle('root',inputs)
    compiled=CompiledInterpreter(graph)
    equal(original['state'],compiled.run_bundle_outputs('root',inputs,['state'])['state'])
    assert compiled.stats.route_counts


def test_nonfinite_parameter_scoring_failure_matches_reference(monkeypatch):
    from cellforge.runtime import lm
    # Use an existing byte artifact whose execution is still well-formed.
    graph=Graph.load(ROOT/'runs/unified-byte/artifact')
    key=next(iter(graph.params))
    graph.params[key].flat[0]=np.nan
    original=lm.score_texts(graph,['Question: x'])
    monkeypatch.setattr(lm,'Interpreter',CompiledInterpreter)
    equal(original,lm.score_texts(graph,['Question: x']))


@pytest.mark.parametrize('temperature', [0.0, 0.7])
def test_integrated_prefill_mixed_length_batch_preserves_tokens(temperature):
    from cellforge.runtime.lm import generate
    graph = Graph.load(ROOT / 'runs/unified-byte/artifact')
    arguments = dict(prompts=['é?', 'Longer prompt with 42.'], max_len=5,
                     temperature=temperature, seed=193)
    reference = generate(graph, 2, **arguments)
    accelerated = generate(graph, 2, backend='compiled', state_only_prefill=True, **arguments)
    assert reference[0] == accelerated[0]
    for key in ('token_ids', 'failure', 'text_codec'):
        equal(reference[1][key], accelerated[1][key])
    assert accelerated[1]['counter_scope'] == 'demanded closure semantic operations'


def test_prefill_requires_compiled_backend():
    from cellforge.runtime.lm import generate
    graph = Graph.load(ROOT / 'runs/unified-byte/artifact')
    with pytest.raises(ValueError, match='compiled'):
        generate(graph, 1, state_only_prefill=True)
