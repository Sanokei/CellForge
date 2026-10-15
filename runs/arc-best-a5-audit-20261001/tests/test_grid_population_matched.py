import importlib.util
from pathlib import Path
import pytest
from cellforge.construction.living_population import LivingPopulation


@pytest.fixture
def runner(monkeypatch):
    scripts=Path(__file__).parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('matched_population',scripts/'run_grid_population_matched.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def base():
    from cellforge.ir.graph import Graph
    # Frozen authored artifacts only; no task data are loaded.
    path=Path(__file__).parents[1]/'runs/arc-grid-genes/two-slot-genome-v1/base'
    if not path.exists():
        candidates=list((Path(__file__).parents[1]/'runs/arc-grid-genes').glob('**/INVENTORY.json'))
        for candidate in candidates:
            if (candidate.parent/'base/graph.json').exists(): return Graph.load(candidate.parent/'base')
        pytest.skip('two-slot authored base fixture not available')
    return Graph.load(path)


def test_selection_cannot_switch_output_rank_between_demonstrations(runner):
    selection=[{'train':[{'input':[[1]],'output':[[3]]},
                         {'input':[[2]],'output':[[4]]}]}]
    calls=[]
    def predictor(demos,grid):
        calls.append(grid)
        return [[[3]],[[9]]] if grid==[[1]] else [[[9]],[[4]]]
    ordinary=runner.evaluate_selection(None,selection,predictor_factory=lambda _:predictor)
    strict=runner.evaluate_selection(None,selection,predictor_factory=lambda _:predictor,rank_consistent=True)
    assert ordinary['solved']==1 and strict['solved']==0
    assert ordinary['predictor_calls']==strict['predictor_calls']==2
    assert not strict['failed'] and len(calls)==4
    def stable(demos,grid):return [[[3 if grid==[[1]] else 4]],[[9]]]
    correct=runner.evaluate_selection(None,selection,predictor_factory=lambda _:stable,rank_consistent=True)
    assert correct['solved']==1 and not correct['failed']


def test_canonical_identity_ignores_lineage_but_validates_declared_function(runner,base):
    a=runner.instantiate_genome(base,runner.GENES[:2],operator_seed=1)
    b=runner.instantiate_genome(base,runner.GENES[:2],operator_seed=99)
    assert a.hash()!=b.hash() and runner.canonical_genotype(base,a)==runner.canonical_genotype(base,b)
    bad=a.copy();bad.meta['two_slot_genome']['alleles'][0]=runner.GENES[2]
    with pytest.raises(ValueError): runner.canonical_genotype(base,bad)
    bad=a.copy();bad.meta['grid_attempts']=1 if base.meta['grid_attempts']==2 else 2
    with pytest.raises(ValueError): runner.canonical_genotype(base,bad)


def test_neutral_genotype_age_does_not_reset_at_death_boundary(runner):
    pop=LivingPopulation(capacity=4,max_age=3,task_manifest='fixed',n_tasks=1)
    evaluation={'genotype_hash':'same-functional-genome','task_manifest':'fixed','solves':[True]}
    pop.admit([evaluation]);pop.advance_age()
    runner.tick_with_evaluation(pop,evaluation);assert pop.members[0].age==2
    runner.tick_with_evaluation(pop,evaluation);assert not pop.members
    assert len(pop.archive)==3
    assert pop.events[-1]['event']=='age_eviction'


def test_random_stream_roles_are_reproducible_distinct_and_legacy_preserved(runner):
    seen=set()
    for index in range(64):
        legacy=runner.proposal_seeds(772101,index)
        assert legacy=={'parent':772101+index,'proposal':772101+index}
        values=runner.proposal_seeds(772101,index,True)
        assert values==runner.proposal_seeds(772101,index,True)
        assert values['parent']!=values['proposal']
        assert not seen.intersection(values.values());seen.update(values.values())
    assert len(seen)==128


def test_expanded_corpus_requires_no_native_files_and_no_query_selection(runner):
    selection,report,paths,policy=runner.frozen_expanded_corpora(2)
    assert len(selection)==len(report)==24 and paths==[]
    assert policy['native_content_manifest'] is None
    assert policy['selection_demo_pairs']==70
    assert all(set(t)=={'train'} for t in selection)
    assert len({t.get('family',t.get('metadata',{}).get('family')) for t in report})==12


def test_expanded_canonical_dispatch_uses_validated_seven_slots(runner):
    from cellforge.ir.graph import Graph
    import expanded_grid_genetics as genetics
    base=Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/expanded-breeding-v2/base')
    child=genetics.mutate(base,1)
    assert runner.canonical_genotype(base,child)==genetics.canonical(base,child)


def test_expanded_runner_end_to_end_budget_and_no_native_access(runner,tmp_path,monkeypatch,capsys):
    import json,sys,types
    class IdentityFixture:
        def __init__(self,graph,**kwargs):pass
        def __call__(self,train,x):return [x,x]
    monkeypatch.setattr(runner,'GridPredictor',IdentityFixture)
    # Protocol test on Windows: production measurements use Linux resource.
    monkeypatch.setitem(sys.modules,'resource',types.SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda _:types.SimpleNamespace(ru_maxrss=0)))
    inventory=Path(__file__).parents[1]/'runs/arc-grid-genes/expanded-breeding-v2/INVENTORY.json'
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['runner','--inventory',str(inventory),'--out',str(out),
                                  '--expanded-bank','--private',str(tmp_path/'does-not-exist'),
                                  '--per-family','2','--report-per-family','10','--selection-seed','3097000',
                                  '--validation-seed','3107000','--max-age','8','--independent-streams'])
    runner.main();capsys.readouterr()
    r=json.loads((out/'result.json').read_text())
    assert r['physical_selection_evaluations']==18 and r['equal_physical_calls']
    assert all(c['physical_evaluations']==6 and c['predictor_calls']==420 for c in r['costs'].values())
    assert r['policy']['native_tasks_loaded']==0 and r['policy']['native_content_manifest'] is None
    assert not r['policy']['native_query_scoring']
    assert all(x['split']=='fresh-synthetic-validation' for x in r['rows'])
    assert all(x['tasks']==120 for x in r['rows'])
    assert r['policy']['selection_seed_base']==3097000 and r['policy']['validation_seed_base']==3107000
    assert r['ledger_verified']==36


def test_all_demo_pairs_executed_after_mismatch_and_no_query_fields(runner):
    calls=[]
    def factory(_):
        def predict(train,input_grid):
            calls.append((train,input_grid));return [[[9]]]
        return predict
    selection=[{'train':[{'input':[[i]],'output':[[i]]} for i in range(3)]}]
    result=runner.evaluate_selection(None,selection,predictor_factory=factory)
    assert result['predictor_calls']==3 and len(calls)==3 and result['solved']==0
    assert not result['failed']
    with pytest.raises(ValueError): runner.evaluate_selection(None,[dict(selection[0],test=[{'input':[[1]],'output':[[1]]}])],predictor_factory=factory)


def test_predictor_errors_are_failed_falsevector_and_all_calls_consume_budget(runner):
    calls=[]
    def factory(_):
        def predict(train,input_grid): calls.append(1);raise RuntimeError('fixture failure')
        return predict
    selection=[{'train':[{'input':[[i]],'output':[[i]]} for i in range(3)]}]
    result=runner.evaluate_selection(None,selection,predictor_factory=factory)
    assert result['failed'] and result['solves']==[False] and result['predictor_calls']==3
    assert len(calls)==len(result['errors'])==3
