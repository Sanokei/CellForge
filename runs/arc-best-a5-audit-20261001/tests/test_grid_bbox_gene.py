import importlib.util
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.runtime.grid import GridPredictor

spec=importlib.util.spec_from_file_location('bbox_gene',Path(__file__).parents[1]/'scripts/build_grid_bbox_gene.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_graph_bbox_crop_assoc_transfer_and_empty_identity():
    parent=Graph.load(Path(__file__).parents[1]/'runs/grid-associative/artifact')
    pool=builder.append_bbox_gene(parent);single=builder.pool.fixed_single_gene(pool,8)
    check_profile(single,'ladder-micro/1')
    train=[{'input':[[0,0,0,0],[0,5,0,0],[0,0,5,0],[0,0,0,0]],'output':[[7,0],[0,7]]}]
    query=[[0,0,0,0,0],[0,0,0,0,0],[0,0,5,5,0],[0,0,0,5,0]]
    expected=[[[7,7],[0,7]]]
    assert GridPredictor(single)(train,query)==expected
    assert GridPredictor(single,backend='compiled')(train,query)==expected
    empty=[[0,0,0],[0,0,0]]
    assert GridPredictor(single)([{'input':empty,'output':empty}],empty)==[empty]


def test_bbox_ranked_pool_prefers_crop_when_demo_shape_requires_it():
    graph=builder.pool.ranked_two(builder.append_bbox_gene(Graph.load(Path(__file__).parents[1]/'runs/grid-associative/artifact')))
    train=[{'input':[[0,0,0],[0,2,0],[0,0,0]],'output':[[2]]}]
    assert GridPredictor(graph)(train,[[0,2,2],[0,0,0]])[0]==[[2,2]]
