import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_expanded_genome import build_expanded_genome
from measure_grid_bank_capacity import curriculum
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.ir.profiles import check_profile


@pytest.mark.parametrize('backend',['reference','compiled'])
def test_expanded_bank_shapes_adaptation_and_reset(backend):
    donor=Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/scaling-two-slot-v2/base')
    g=build_expanded_genome(donor)
    assert g.meta['state_dim']==1515 and len(g.meta['expanded_genome']['alleles'])==7
    assert check_profile(g,'ladder-micro/1')['dispatches']==2
    assert g.stats()['unique_param_storage']==273
    tasks=curriculum(3017000)
    predictor=GridPredictor(g,backend=backend)
    for index in (11,15,17,19,21,23,11):
        t=tasks[index];q=t['test'][0]
        assert q['output'] in predictor(t['train'],q['input'])
        assert q['output'] in predictor(list(reversed(t['train'])),q['input'])
    with pytest.raises(ValueError):build_expanded_genome(donor,['geometry_0']*6)


def test_expanded_measurement_family_balanced_without_scaling_fallbacks():
    from measure_grid_expanded_capacity import family
    tasks=family()
    names=[t.get('family',t.get('metadata',{}).get('family')) for t in tasks]
    assert len(tasks)==120 and len(set(names))==12
    assert all(names.count(f)==10 for f in names)
    for t in tasks:
        if t.get('family') in ('repeat2','repeat3'):
            factor=int(t['family'][-1])
            for pair in t['train']+t['test']:
                assert len(pair['output'])==factor*len(pair['input'])<=30
                assert len(pair['output'][0])==factor*len(pair['input'][0])<=30


def test_nine_slot_contract_and_breeding():
    from expanded_grid_genetics import crossover,mutate,canonical
    from cellforge.construction.transactions import apply_ops
    donor=Graph.load(Path(__file__).parents[1]/'runs/joint-layout-repair-v1/maker_repair')
    alleles=donor.meta['expanded_genome']['alleles']
    graph=build_expanded_genome(donor,alleles+['scaling_bank2_repeat.gene','scaling_bank3_repeat.gene'])
    assert graph.meta['state_dim']==1717
    assert graph.meta['expanded_genome']['slots']==list(range(8,17))
    control,_=apply_ops(graph,[dict(op='set_slot_gene',slot=16,gene='geometry_0')])
    canonical(graph,control)
    for seed in range(8):
        child=crossover(graph,control,seed)
        canonical(graph,mutate(child,seed))
        assert len(child.meta['expanded_genome']['donor_per_slot'])==9
    with pytest.raises(ValueError):build_expanded_genome(donor,alleles+['geometry_0']*3)


def test_frozen_nine_slot_build_recipe_replays_lineage_hashes():
    import json
    root=Path(__file__).parents[1]
    recipe=json.loads((root/'runs/nine-slot-capacity-v1/BUILD_REPLAY.json').read_text())
    parent=Graph.load(root/'runs/joint-layout-repair-v1/maker_repair')
    assert parent.hash()==recipe['parent_hash']
    for row in recipe['reproduced_exactly']:
        graph=build_expanded_genome(parent,row['alleles'],
            pre_rank_limitations=recipe['pre_rank_limitations'])
        graph.meta['limitations']=recipe['post_rank_limitations']
        assert graph.hash()==row['hash']


def test_nine_slot_random_control_and_fourteen_family_selection():
    from run_grid_population_matched import random_genotype,frozen_expanded_corpora,evaluate_selection
    from expanded_grid_genetics import canonical
    root=Path(__file__).parents[1]
    graph=Graph.load(root/'runs/nine-slot-capacity-v1/nine-active')
    candidate=random_genotype(graph,773501,expanded_bank=True)
    assert len(candidate.meta['expanded_genome']['alleles'])==9
    canonical(graph,candidate)
    selection,report,paths,commitments=frozen_expanded_corpora(
        1,4597000,4607000,2,include_marker=True,include_numeric=True)
    assert len(selection)==14 and len(report)==28 and paths==[]
    assert all(set(task)=={'train'} for task in selection)
    assert commitments['selection_demo_pairs']==39
    result=evaluate_selection(candidate,selection,rank_consistent=True)
    assert result['predictor_calls']==39 and not result['failed']


def test_cached_selection_preserves_solve_vector_and_physical_evaluation_calls():
    from run_grid_population_matched import frozen_expanded_corpora,evaluate_selection
    graph=Graph.load(Path(__file__).parents[1]/'runs/nine-slot-capacity-v1/nine-active')
    selection,_,_,_=frozen_expanded_corpora(1,4617000,4627000,2,True,True)
    plain=evaluate_selection(graph,selection,rank_consistent=True)
    cached=evaluate_selection(graph,selection,rank_consistent=True,cache_demo_state=True)
    for field in ('solves','solved','tasks','failed','errors','predictor_calls'):
        assert plain[field]==cached[field]
    assert cached['predictor_calls']==39
    assert cached['runtime_steps']==78
    assert cached['demo_state_cache_hits']==25
