import importlib.util
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.ir.profiles import check_profile

spec=importlib.util.spec_from_file_location('grid_gene_builder',Path(__file__).parents[1]/'scripts/build_grid_gene_pool.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_ranked_two_assoc_reconstruction_and_reset():
    parent=Graph.load(Path(__file__).parents[1]/'runs/grid-associative/artifact')
    child=builder.ranked_two(parent);check_profile(child,'ladder-micro/1')
    predict=GridPredictor(child)
    train=[{'input':[[0,1,2],[3,4,5]],'output':[[6,7],[8,9],[1,2]]}]
    x=[[0,1,2],[3,4,5]]
    outputs=predict(train,x)
    assert len(outputs)==2
    assert outputs[0]==train[0]['output']
    # New task resets learned associations and graph-ranked state.
    plain=[{'input':[[1,2],[3,4]],'output':[[1,2],[3,4]]}]
    assert predict(plain,plain[0]['input'])[0]==plain[0]['output']
    assert predict(train,x)==outputs


def test_ranked_two_geometry_and_fixed_single_coverage_control():
    parent=Graph.load(Path(__file__).parents[1]/'runs/grid20/artifact')
    graph=builder.ranked_two(parent);predict=GridPredictor(graph)
    train=[{'input':[[1,2,3],[4,5,6]],'output':[[1,4],[2,5],[3,6]]}]
    outputs=predict(train,train[0]['input'])
    assert len(outputs)==2 and outputs[0]==train[0]['output']
    single=builder.fixed_single_gene(parent,0)
    assert GridPredictor(single)(train,train[0]['input'])==[train[0]['input']]
    assert single.meta['gene_pool']['fixed_gene']=='expert_0'


def test_pair_crossover_preserves_both_donor_outputs_and_recurrent_reset():
    assoc=Graph.load(Path(__file__).parents[1]/'runs/grid-associative/artifact')
    geometry=Graph.load(Path(__file__).parents[1]/'runs/grid20/artifact')
    left=builder.fixed_single_gene(assoc,0);right=builder.fixed_single_gene(geometry,10)
    child=builder.breed_fixed_pair(left,right);check_profile(child,'ladder-micro/1')
    train=[{'input':[[0,1,2],[3,4,5]],'output':[[6,7,8],[9,1,2]]}]
    query=[[5,4,3],[2,1,0]]
    expected=GridPredictor(left)(train,query)+GridPredictor(right)(train,query)
    predictor=GridPredictor(child)
    assert predictor(train,query)==expected
    assert child.meta['state_dim']==left.meta['state_dim']+right.meta['state_dim']
    assert predictor(train,query)==expected
    # Source donors remain untouched by import/mutation of the child.
    assert child.meta['gene_pool']['parent_hashes']==[left.hash(),right.hash()]
