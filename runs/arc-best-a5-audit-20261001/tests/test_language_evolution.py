import importlib.util
from pathlib import Path
import numpy as np
from cellforge.construction.bootstrap import bootstrap_lm

spec = importlib.util.spec_from_file_location("evolution",Path(__file__).parents[1]/"scripts/evolve_language_islands.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_primary_rejects_adaptive_mixer_metadata_and_module():
    import pytest
    graph=bootstrap_lm('ab')
    module.reject_adaptive_mixer(graph)
    graph.meta['adaptive_mixer_recipe']={}
    with pytest.raises(ValueError,match='excluded'):module.reject_adaptive_mixer(graph)
    graph.meta.pop('adaptive_mixer_recipe')
    graph.defs['mixer.posterior']={'nodes':[]}
    with pytest.raises(ValueError,match='excluded'):module.reject_adaptive_mixer(graph)


def test_discrete_mutation_reproducible_and_parent_unchanged():
    parent = bootstrap_lm("ab")
    parent.params["boot_logits"][:] = 1.
    original = parent.hash()
    a,pa = module.mutate(parent,0,77)
    b,pb = module.mutate(parent,0,77)
    assert a.hash()==b.hash() and pa==pb
    assert parent.hash()==original
    assert sum(np.count_nonzero(parent.params[n]!=a.params[n]) for n in parent.params)==1
    assert not pa["gradient_estimator"]


def test_nonzero_and_zero_activation_both_change_one_coordinate():
    parent = bootstrap_lm("ab")
    for seed in range(10):
        child,proposal = module.mutate(parent,1,seed)
        assert proposal["new"]!=proposal["old"]
        assert sum(np.count_nonzero(parent.params[n]!=child.params[n]) for n in parent.params)==1


def test_adaptive_floor_and_two_batch_acceptance():
    stats={op:{'tried':100,'accepted':0} for op in module.OPERATORS}
    stats['row_scale']['accepted']=100
    p=module.adaptive_probabilities(stats)
    assert np.isclose(p.sum(),1) and np.all(p>=.05)
    assert p[0]>p[1]
    score=lambda x:{'eligible_for_selection':True,'fitness_bits_per_byte':x}
    assert module.accept_two_batches([score(3),score(4)],[score(2),score(3)])
    assert not module.accept_two_batches([score(3),score(4)],[score(2),score(4)])
    assert not module.accept_two_batches([],[])
    p=dict(score(3),composite_bpb=4.)
    better=dict(score(3.02),composite_bpb=3.9)
    assert module.accept_two_batches([p,p],[better,better])
    bad_corpus=dict(score(3.04),composite_bpb=3.8)
    assert not module.accept_two_batches([p,p],[bad_corpus,bad_corpus])
    rounding=dict(score(3),composite_bpb=4.-1e-12)
    assert not module.accept_two_batches([p,p],[rounding,rounding])


def test_structured_operators_frozen_parent_and_reproducible():
    from cellforge.ir.graph import Graph
    parent=Graph.load(Path(__file__).parents[1]/'runs/unified-byte/artifact')
    original=parent.hash()
    for op in module.OPERATORS:
        if op in ('trunk_tile','mixing_gate','recipe_update','recipe_transfer'): continue
        a,lineage=module.structured_mutate(parent,parent,op,902)
        b,other=module.structured_mutate(parent,parent,op,902)
        assert a.hash()==b.hash() and lineage==other
        assert parent.hash()==original
        assert lineage['operator']==op and not lineage['gradient_estimator']
        if op=='typed_module':
            assert lineage['details']['runtime_admission_required']


def test_four_arm_engine_keeps_pure_lineage_and_no_report(tmp_path,monkeypatch):
    import json,sys
    from cellforge.ir.graph import Graph
    graph=Graph.load(Path(__file__).parents[1]/'runs/unified-byte/artifact')
    graph.save(tmp_path/'founder')
    fitted=graph.copy(); fitted.meta['fitted_provenance']={'method':'fixture-only','partition':'selection','corpus_sha256':'fixture'}
    fitted.save(tmp_path/'fitted')
    for name,text in [('a','ab'),('b','cd')]:
        (tmp_path/(name+'.json')).write_text(json.dumps([{'id':name,'family':name,'kind':'next_byte','text':text}]))
    def evaluator(graph,items,**kwargs):
        return {'fitness_bits_per_byte':3.,'eligible_for_selection':True,'next_byte':{'content_bytes':2,'content_bits':6.,'failed_documents':0}}
    monkeypatch.setattr(module,'evaluate_language_curriculum',evaluator)
    import cellforge.reference.evolution_fitness as fast
    class FixtureFitness:
        def __init__(self,founder,items): self.items=items
        def evaluate(self,graph): return evaluator(graph,self.items)
    monkeypatch.setattr(fast,'EvolutionFitness',FixtureFitness)
    for name,text in [('ca','ef'),('cb','gh')]:
        (tmp_path/(name+'.json')).write_text(json.dumps([{'id':name,'family':name,'kind':'next_byte','text':text,'score_byte_ranges':[[1,2]],'dependency_distance_bytes':9}]))
    monkeypatch.setattr(sys,'argv',['evolve','--artifact',str(tmp_path/'founder'),'--fitted-artifact',str(tmp_path/'fitted'),'--batch-a',str(tmp_path/'a.json'),'--batch-b',str(tmp_path/'b.json'),'--context-a',str(tmp_path/'ca.json'),'--context-b',str(tmp_path/'cb.json'),'--qualified-fast-fitness','--operator-plan','structured','--out',str(tmp_path/'result.json'),'--steps','2'])
    module.main()
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['policy']['report_evaluations']==0
    assert result['pure_hash']==graph.hash()
    assert len(result['records'])==2
    assert set(result['records'][0]['arms'])=={'fitted_evolution','fitted_random','founder_evolution'}
    assert result['policy']['gradient_steps']==0





def test_archive_sampling_includes_failures_and_correct_ranks():
    archive=[{'performance':1.,'novelty':1.},{'performance':2.,'novelty':1.},{'performance':float('inf'),'novelty':0.,'failed':True}]
    p=module.archive_probabilities(archive)
    assert np.isclose(p.sum(),1) and np.all(p>0)
    assert p[0]>p[1]>p[2]


def test_recipe_simplex_mutation_preserves_structural_arrays():
    parent=bootstrap_lm('ab')
    parent.params['mixer.prior']=np.array([.8,.15,.05])
    parent.params['cfwb.keys']=np.array([1.,2.])
    parent.meta['evolution_recipe_parameters']=[{'parameter':'mixer.prior','minimum':1e-9,'maximum':1.,'normalization':'simplex'}]
    child,lineage=module.structured_mutate(parent,parent,'recipe_update',333)
    assert np.all(child.params['mixer.prior']>0) and np.isclose(child.params['mixer.prior'].sum(),1)
    assert np.array_equal(parent.params['cfwb.keys'],child.params['cfwb.keys'])
    assert lineage['operator']=='recipe_update'


def test_exporter_manifests_reject_report_and_wb_operator_mapping():
    import pytest
    items=[{'id':'x','kind':'next_byte','text':'public fixture'}]
    assert module.selection_items({'partition':'selection','items':items})==items
    assert module.selection_items(items)==items
    for label in ('report','audit','final',None):
        with pytest.raises(ValueError):module.selection_items({'partition':label,'items':items})
    parent=bootstrap_lm('ab')
    assert module.mapped_operator(parent,'trunk_tile')[0]=='block_scale'
    assert module.mapped_operator(parent,'mixing_gate')[0]=='block_scale'
    fitted=parent.copy();fitted.params['cfwb.values']=np.array([.5]);fitted.meta['fitted_provenance']={'fixture':True}
    assert module.mapped_operator(fitted,'trunk_tile')[0]=='mixing_gate'


def test_dictionary_four_arm_ledger_curves_and_failure_messages(tmp_path,monkeypatch):
    import json,sys
    from cellforge.ir.graph import Graph
    from cellforge.governance.ledger import Ledger
    graph=Graph.load(Path(__file__).parents[1]/'runs/unified-byte/artifact');graph.save(tmp_path/'founder')
    fitted=graph.copy();fitted.meta['fitted_provenance']={'method':'WB fixture','source_split':'dev_train'};fitted.params['cfwb.values']=np.array([.5])
    fitted.defs[fitted.entries['step']]['nodes'].append({'id':'authored_residual','use':'op:scale','in':{'x':fitted.defs[fitted.entries['step']]['outputs']['logits']},'args':{'c':0}})
    fitted.save(tmp_path/'fitted')
    for name,text in [('a','ab'),('b','cd'),('ca','ef'),('cb','gh')]:
        item={'id':name,'family':name,'kind':'next_byte','text':text}
        if name.startswith('c'):item.update(score_byte_ranges=[[1,2]],dependency_distance_bytes=9)
        (tmp_path/(name+'.json')).write_text(json.dumps({'partition':'selection','items':[item]}))
    import cellforge.reference.evolution_fitness as fast
    class FixtureFitness:
        def __init__(self,founder,items):pass
        def evaluate(self,graph):return {'fitness_bits_per_byte':3.,'eligible_for_selection':True,'next_byte':{'content_bytes':2,'content_bits':6.,'failed_documents':0}}
    monkeypatch.setattr(fast,'EvolutionFitness',FixtureFitness)
    def fail(*args):raise ValueError('explicit fixture proposal failure')
    monkeypatch.setattr(module,'structured_mutate',fail)
    argv=['evolve','--artifact',str(tmp_path/'founder'),'--fitted-artifact',str(tmp_path/'fitted'),
          '--batch-a',str(tmp_path/'a.json'),'--batch-b',str(tmp_path/'b.json'),'--context-a',str(tmp_path/'ca.json'),'--context-b',str(tmp_path/'cb.json'),
          '--qualified-fast-fitness','--out',str(tmp_path/'result.json'),'--steps','2']
    monkeypatch.setattr(sys,'argv',argv);module.main()
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['proposal_counts']=={'fitted_evolution':2,'fitted_random':2,'founder_evolution':2}
    assert result['policy']['operator_plan']=='structured'
    assert result['report_evaluations']==0 and len(result['evaluation_curve'])==2
    for record in result['records']:
        for arm in record['arms'].values():
            assert arm['failure_message']=='explicit fixture proposal failure' and arm['reserved_batch_slots']==4
    progress=json.loads((tmp_path/'result-progress.json').read_text())
    assert progress['status']=='complete_bounded_development'
    ledger=Ledger(tmp_path/'result-ledger.jsonl');assert ledger.verify()==result['ledger_entries_verified']
    assert ledger.entries()[0]['type']=='policy_frozen'
    assert len(ledger.find('proposal_failed'))==6
