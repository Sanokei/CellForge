import json
from pathlib import Path
import numpy as np
import pytest
from cellforge.ir.graph import Graph
from cellforge.reference.evolution_fitness import EvolutionFitness,compiled_content_fitness
from cellforge.runtime.lm import score_texts
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.text_codec import BYTE_CODEC,BYTE_VOCAB


def recurrent_fixture():
    return Graph(defs={'step':{'kind':'bundle','inputs':['ctx','prev','state','step'],
       'nodes':[{'id':'prev_slice','use':'op:slice','in':{'x':'$prev'},'args':{'axis':0,'start':0,'stop':1}},
                {'id':'delta','use':'op:scale','in':{'x':'prev_slice'},'args':{'c':.125}},
                {'id':'next','use':'op:add','in':{'a':'$state','b':'delta'}},
                {'id':'weight','use':'param:weight'},
                {'id':'logits','use':'op:mul','in':{'a':'$prev','b':'weight'}}],
       'outputs':{'logits':'logits','state':'next'}}},params={'weight':np.ones(259)},
       entries={'step':'step'},meta={'task':'lm','text_codec':BYTE_CODEC,'vocab':BYTE_VOCAB,'state_dim':1,'ctx_len':3})


def reference(graph,items):
    scores,info=score_texts(graph,[x['text'] for x in items])
    lengths=[len(x['text'].encode()) for x in items]
    bits=[float(info['per_pos_bits'][j,:n].sum()) for j,n in enumerate(lengths)]
    return scores,info,bits


def test_windows_keep_full_context_state_and_exact_cached_input_trajectory():
    g=recurrent_fixture();items=[{'text':'a'*70},{'text':'é and more'}]
    e=EvolutionFitness(g,items)
    it=Interpreter(g)
    for j in range(len(items)):
        previous=np.zeros((1,1))
        for record in [r for r in e.records if r[0]==j]:
            np.testing.assert_array_equal(record[4]['state'],previous)
            out=it.run_bundle('step',record[4]);previous=out['state']
            np.testing.assert_array_equal(record[5][('next','out')],previous)
    for candidate in (g,g.copy()):
        candidate.params['weight'][100]=1.5
        expected=reference(candidate,items)[2]
        cached=e.evaluate(candidate)
        full=compiled_content_fitness(candidate,items)
        np.testing.assert_allclose(cached['next_byte']['per_document_content_bits'],expected,rtol=0,atol=1e-12)
        assert cached['fitness_bits_per_byte']==pytest.approx(full['fitness_bits_per_byte'],abs=1e-13)
        assert 'cache' in cached['cache_mode']
    # State-update constants, recurrent reads and context shape are unsafe.
    state=g.copy();state.defs['step']['nodes'][1]['args']['c']=.25
    assert e.evaluate(state)['cache_mode']=='compiled_full_unsafe_prefix_fallback'
    read=g.copy();read.defs['step']['nodes'][2]['in']['b']='$step'
    assert e.evaluate(read)['cache_mode']=='compiled_full_unsafe_prefix_fallback'
    context=g.copy();context.meta['ctx_len']=4
    assert e.evaluate(context)['cache_mode']=='compiled_full_unsafe_prefix_fallback'


def test_nonfinite_logit_failure_charges_entire_document_and_eos_checked():
    g=recurrent_fixture();items=[{'text':'é '+('x'*66)}];e=EvolutionFitness(g,items)
    bad=g.copy();bad.params['weight'][100]=np.inf
    with np.errstate(invalid='ignore'):
        result=e.evaluate(bad)
    assert not result['eligible_for_selection']
    assert result['next_byte']['failed_documents']==1
    assert result['fitness_bits_per_byte']==pytest.approx(np.log2(259)+1)
    assert result['timing']['evaluated_bytes']==len(items[0]['text'].encode())+1


def test_actual_nested_router_expert_mutations_match_reference_and_state_mutation_falls_back():
    g=Graph.load(Path(__file__).parents[1]/'runs/unified-byte/artifact')
    items=[{'text':'Hello, there. A traveler wrote a note. '+('x'*32)},{'text':'héllo world'}]
    e=EvolutionFitness(g,items)
    variants=[g]
    for name in ('cl_router.sp.W1','cl_bigram.W10'):
        child=g.copy();child.params[name].flat[0]+=.125;variants.append(child)
    for child in variants:
        expected=reference(child,items)[2]
        result=e.evaluate(child)
        assert result['cache_mode']=='state_ancestor_cache_cold'
        np.testing.assert_allclose(result['next_byte']['per_document_content_bits'],expected,rtol=0,atol=1e-10)
        warm=e.evaluate(child)
        assert warm['cache_mode']=='state_ancestor_cache_warm'
        assert warm['next_byte']==result['next_byte']
    state=g.copy();state.params['cl_name_write.e0'].flat[0]+=.125
    result=e.evaluate(state)
    assert result['cache_mode']=='compiled_full_unsafe_prefix_fallback'
    np.testing.assert_allclose(result['next_byte']['per_document_content_bits'],reference(state,items)[2],rtol=0,atol=1e-10)


def test_selected_byte_ranges_keep_unscored_prefix_and_charge_selected_bytes_on_failure():
    g=recurrent_fixture();text=('a'*70)+'é'
    items=[{'text':text,'score_byte_ranges':[[70,72]]}]
    e=EvolutionFitness(g,items,window_bytes=64)
    _,info,_=reference(g,items)
    expected=float(info['per_pos_bits'][0,70:72].sum())
    for result in (e.evaluate(g),compiled_content_fitness(g,items)):
        assert result['next_byte']['content_bytes']==2
        assert result['next_byte']['full_content_bytes']==72
        assert result['timing']['evaluated_bytes']==73
        assert result['next_byte']['content_bits']==pytest.approx(expected,abs=1e-12)
    bad=g.copy();bad.params['weight'][100]=np.inf
    with np.errstate(invalid='ignore'):
        result=e.evaluate(bad)
    assert not result['eligible_for_selection']
    assert result['next_byte']['content_bits']==pytest.approx(2*(np.log2(259)+1))
    for ranges in ([],[[0,0]],[[0,73]],[[2,5],[4,6]],[[2,5],[0,1]],[[0.,1]]):
        with pytest.raises(ValueError):EvolutionFitness(g,[{'text':text,'score_byte_ranges':ranges}])


def test_reused_nested_constants_refresh_after_parameter_change():
    from cellforge.runtime.compiled import CompiledInterpreter
    g=recurrent_fixture();g.defs['step']['nodes'].insert(3,{'id':'constant','use':'op:scale','in':{'x':'weight'},'args':{'c':2}})
    # Put the parameter before its constant consumer in valid topological order.
    nodes=g.defs['step']['nodes'];constant=nodes.pop(3);nodes.insert(4,constant)
    g.defs['step']['nodes'][-1]['in']['b']='constant'
    parent=CompiledInterpreter(g)
    child=g.copy();child.params['weight'][100]=2
    reused=CompiledInterpreter(child,reuse_from=parent)
    fresh=CompiledInterpreter(child)
    inp={'ctx':np.zeros((1,3,259)),'prev':np.ones((1,259)),'state':np.zeros((1,1)),'step':np.zeros((1,1))}
    np.testing.assert_array_equal(reused.run_bundle('step',inp)['logits'],fresh.run_bundle('step',inp)['logits'])
    assert reused.run_bundle('step',inp)['logits'][0,100]==4


def test_csr_batch_rows_are_independent_and_fitted_table_mutation_scores_match():
    from cellforge.runtime.ops import op_csr_lookup
    from cellforge.construction.wb_byte_seed import wb_byte_seed
    import hashlib
    rng=np.random.default_rng(908)
    query=rng.choice([1.,4.,7.,999.],size=(64,1))
    arrays={'keys':np.array([[1.,4.,7.]]),'offsets':np.array([[0.,2.,3.,5.]]),
            'columns':np.array([[0.,0.,2.,1.,2.]]),'values':np.array([[.2,.3,.9,.1,.7]])}
    batch=op_csr_lookup(dict(arrays,query=query),{'width':3})
    for j in range(64):
        np.testing.assert_array_equal(batch[j:j+1],op_csr_lookup(dict(arrays,query=query[j:j+1]),{'width':3}))
    per_example={k:np.repeat(v,64,axis=0) for k,v in arrays.items()}
    per_example['values'][0]*=2
    separate=op_csr_lookup(dict(per_example,query=query),{'width':3})
    for j in range(64):
        row={k:v[j:j+1] for k,v in per_example.items()}
        np.testing.assert_array_equal(separate[j:j+1],op_csr_lookup(dict(row,query=query[j:j+1]),{'width':3}))
    founder=Graph.load(Path(__file__).parents[1]/'runs/unified-byte/artifact')
    partition={'partition':'dev_train_selection','source_split':'dev_train','corpus_sha256':hashlib.sha256(b'tiny synthetic fixture').hexdigest(),
               'items':[{'family':'fixture-a','text':'abracadabra abc'},{'family':'fixture-b','text':'café 中文'}]}
    fitted,_,_=wb_byte_seed(founder,partition)
    items=[{'text':'abracadabra '+('xyz '*16),'score_byte_ranges':[[0,3],[65,70]]},{'text':'unseen café 中文'}]
    e=EvolutionFitness(fitted,items)
    for index,child in enumerate((fitted,fitted.copy())):
        child.params['cfwb.values'].flat[0]+=.001*(index+1)
        result=e.evaluate(child)
        scores,info=score_texts(child,[i['text'] for i in items])
        expected=[]
        for j,item in enumerate(items):
            ranges=item.get('score_byte_ranges',[[0,len(item['text'].encode())]])
            expected.append(sum(float(info['per_pos_bits'][j,start:stop].sum()) for start,stop in ranges))
        assert result['cache_mode']=='state_ancestor_cache_cold'
        np.testing.assert_allclose(result['next_byte']['per_document_content_bits'],expected,rtol=0,atol=1e-10)
