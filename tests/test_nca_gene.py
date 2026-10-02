import numpy as np
from cellforge.construction.nca_gene import build_nca_gene,mutate_nca_gene,crossover_nca_gene
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter

def test_identity_shared_microcells_all_unrolls():
    grids=[[[1,2,3],[4,5,6]],[[9]],(np.arange(900).reshape(30,30)%10).tolist()]
    for t in (2,4,8):
        graph=build_nca_gene(t); predictor=GridPredictor(graph)
        assert graph.stats()['unique_param_storage']==1488
        assert sum(d['kind']=='cell' for d in graph.defs.values())==48
        for grid in grids: assert predictor([{'input':grid,'output':grid}],grid)==[grid]

def test_local_neighborhood_mask_and_reset():
    graph=build_nca_gene(2,'neighbor_majority'); predictor=GridPredictor(graph)
    grid=[[0,1,0],[0,0,0],[0,0,0]]
    first=predictor([{'input':[[7]],'output':[[8]]}],grid)
    assert len(first[0])==3 and len(first[0][0])==3
    assert first==predictor([{'input':[[2]],'output':[[3]]}],grid)
    x,m,h,w=pack_grid([[1]])
    result=Interpreter(graph).run_bundle('nca_gene',{'input':x,'input_mask':m,'input_height':h,'input_width':w})
    assert not result['colors'][0,1:].any() and not result['colors'][0,:,1:].any()

def test_mutation_and_import_are_reproducible_without_parent_changes():
    parent=build_nca_gene(); original=parent.hash(); a=mutate_nca_gene(parent,55); b=mutate_nca_gene(parent,55)
    assert a.hash()==b.hash() and parent.hash()==original
    child=crossover_nca_gene(parent,build_nca_gene(seed='neighbor_majority'),3,'donor_')
    assert parent.hash()==original and child.meta['nca_crossover']['runtime_admission_required']
    assert GridPredictor(child)([{'input':[[1]],'output':[[1]]}],[[1]])


def test_fixed_slots_survive_all_discrete_mutation_choices():
    from cellforge.construction.nca_gene import nca_genome_contract
    parent=build_nca_gene(); operators=set()
    for seed in range(20):
        child=mutate_nca_gene(parent,seed); operators.add(child.meta['nca_mutation']['details']['operator'])
        assert nca_genome_contract(child)['active_cells']==48
    assert operators=={'row_block','cell_activation','typed_wire_swap'}


def test_training_operator_rejects_test_targets_and_runs_small_fixture(tmp_path,monkeypatch):
    import importlib.util,json,sys,pytest
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('nca_evolution',Path(__file__).parents[1]/'scripts/evolve_nca_gene.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    items=[{'task_hash':'1'*64,'train':[{'input':[[1,2]],'output':[[1,2]]}]}]
    module.validate_items(items)
    with pytest.raises(ValueError): module.validate_items([dict(items[0],test=[{'output':[[2]]}])])
    path=tmp_path/'items.json'; path.write_text(json.dumps(items))
    monkeypatch.setattr(sys,'argv',['evolve','--items',str(path),'--private-out',str(tmp_path/'private'),'--public',str(tmp_path/'public.json'),'--proposals-per-arm','2','--seconds','2'])
    module.main(); result=json.loads((tmp_path/'public.json').read_text())
    assert result['proposals_per_arm']==2 and result['policy']['test_evaluations']==0
    assert (tmp_path/'private/winners-frozen.json').exists()
    assert len(result['proposals'])==4
