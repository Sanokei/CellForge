import copy
import sys
from pathlib import Path
import numpy as np
import pytest
from cellforge.ir.graph import Graph
from cellforge.construction.graph_niche_archive import GraphNicheArchive
from cellforge.runtime.grid import pack_grid
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))

def graph():
    g=Graph.load(Path(__file__).resolve().parents[1]/'runs/priority-a-v2/batch-03/775101-maker_repair-frozen-candidate')
    return g

def metrics(outcomes):
    return dict(fixtures_passed=True,caps_passed=True,**{label:dict(partition='development',manifest=label,
        source='public-training-real' if label=='anchor' else 'authored-generator',
        outcomes=dict(outcomes),false_positive_given_consistency=0.) for label in ('fresh','archive','anchor')})

def fixture(g):
    x,_,h,w=pack_grid([[1]])
    inputs=dict(input=x,input_height=h,input_width=w,output=x,output_height=h,output_width=w,
        state=np.zeros((1,g.meta['state_dim'])),phase=np.zeros((1,1)))
    return [dict(entry=g.entries['grid_step'],inputs={k:v.tolist() for k,v in inputs.items()},
        output_shapes=dict(state=[1,g.meta['state_dim']],color_logits=[1,2,30,30,10],height_logits=[1,2,30],width_logits=[1,2,30]))]

def test_real_graph_literal_proposer_and_mortality(tmp_path):
    g=graph();a=GraphNicheArchive(tmp_path/'ledger',budget=3,capacity=2,lifespan=1)
    a.seed(g,metrics({'a':True,'b':False}),niche='crop-fill')
    shown=dict(partition='development',evidence='train-pairs',graph_hash=g.hash(),counterexamples=[])
    observed=[]
    def proposer(parent,diagnostics,rng):
        observed.append((parent.hash(),diagnostics['graph_hash']))
        return dict(base=parent.hash(),scope='joint',rationale='fixture-only literal gate edit',
            actor='test',model='no-model-call',ops=[dict(op='set_param',name='context_mode.W',value=[[2.]])])
    result=a.mutate(g.hash(),proposer,diagnostics=shown,fixtures=fixture(g),evaluate=lambda _:metrics({'a':True,'b':True}),rng=np.random.default_rng(1))
    assert observed==[(g.hash(),g.hash())]
    assert result['receipt']['status']=='accepted'
    child=result['candidate'].hash()
    assert child in a.living and g.hash() not in a.living
    assert g.hash() in a.archive and child in a.archive
    assert not hasattr(a.archive[g.hash()],'graph')
    assert a.donor(g.hash()).hash()==g.hash()
    assert a.sample_parent(np.random.default_rng(2))==child
    assert a.shared_rewards()=={child:2.}

def test_invalid_proposer_charged_and_niche_can_die(tmp_path):
    g=graph();a=GraphNicheArchive(tmp_path/'ledger',budget=2,lifespan=1)
    a.seed(g,metrics({'a':True}),niche='x')
    shown=dict(partition='development',evidence='train-pairs',graph_hash=g.hash())
    result=a.mutate(g.hash(),lambda *args:None,diagnostics=shown,fixtures=[],evaluate=None,rng=np.random.default_rng(1))
    assert result['receipt']['status']=='invalid'
    assert len(a.maker.operator.rows)==1
    assert not a.living and g.hash() in a.archive
    with pytest.raises(ValueError,match='extinct'):a.sample_parent(np.random.default_rng(2))

def test_unpaired_food_sources_rejected(tmp_path):
    g=graph();a=GraphNicheArchive(tmp_path/'ledger',budget=2)
    a.seed(g,metrics({'a':True}),niche='x')
    other=g.copy();other.meta['test_variant']=1
    with pytest.raises(ValueError,match='paired'):a.seed(other,metrics({'b':True}),niche='y')

def test_real_numeric_breeding_then_maker_charges_one_offspring(tmp_path):
    from priority_a_numeric_breeding import breed_numeric_champion
    left=graph();right=left.copy();right.params['bank_adjacency_recolor.marker.b']+=1
    a=GraphNicheArchive(tmp_path/'ledger',budget=3,lifespan=5)
    a.seed(left,metrics({'a':True,'b':False}),niche='crop')
    a.seed(right,metrics({'a':False,'b':True}),niche='adjacency')
    seen=[]
    def propose(base,shown,rng):
        seen.append(shown['reproduction'])
        return dict(base=base.hash(),scope='joint',rationale='literal fixture-only gate change',
            actor='fixture',model='no-model-call',ops=[dict(op='set_param',name='context_mode.W',value=[[2.]])])
    evaluated=[]
    def evaluate(g):
        evaluated.append(g.hash());return metrics({'a':True,'b':True})
    result=a.breed_and_mutate(left.hash(),right.hash(),breed_numeric_champion,propose,
        diagnostics_for=lambda g:dict(partition='development',evidence='train-pairs',graph_hash=g.hash()),
        fixtures_for=fixture,evaluate=evaluate,rng=np.random.default_rng(2),breeding_seed=5)
    assert result['receipt']['status']=='accepted'
    assert len(a.maker.operator.rows)==len(evaluated)==1
    assert seen[0]['parent_hashes']==[left.hash(),right.hash()]
    child=result['candidate'].hash()
    assert seen[0]['bred_hash']!=child and seen[0]['bred_hash'] not in a.living
    assert a.living[child].parents==(left.hash(),right.hash())
    assert result['candidate'].meta['expanded_genome']['parent_hashes']==[left.hash(),right.hash()]

def test_broken_breeder_is_charged_without_scoring(tmp_path):
    left=graph();right=left.copy();right.meta['variant']=1
    a=GraphNicheArchive(tmp_path/'ledger',budget=2)
    for g in (left,right):a.seed(g,metrics({'a':True}),niche='x')
    def broken(*args):raise ValueError('invalid wiring')
    result=a.breed_and_mutate(left.hash(),right.hash(),broken,lambda *args:None,
        diagnostics_for=lambda _:None,fixtures_for=fixture,evaluate=lambda _:pytest.fail('must not score'),
        rng=np.random.default_rng(1),breeding_seed=1)
    assert result['receipt']['status']=='invalid'
    assert result['receipt']['evaluation_calls']==0 and len(a.maker.operator.rows)==1

def test_archive_loading_detects_corruption_and_restart_cannot_reset_ages(tmp_path):
    g=graph();a=GraphNicheArchive(tmp_path/'ledger',budget=2)
    a.seed(g,metrics({'a':True}),niche='x')
    path=a.archive[g.hash()].path/'graph.json'
    import json
    data=json.loads(path.read_text());data['meta']['tampered']=True;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash mismatch'):a.donor(g.hash())
    shown=dict(partition='development',evidence='train-pairs',graph_hash=g.hash())
    a.mutate(g.hash(),lambda *args:None,diagnostics=shown,fixtures=[],evaluate=None,rng=np.random.default_rng(1))
    with pytest.raises(ValueError,match='resumption'):GraphNicheArchive(tmp_path/'ledger',budget=2)
