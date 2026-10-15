import importlib.util
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.ir.profiles import check_profile

spec=importlib.util.spec_from_file_location('ranked_pairs',Path(__file__).parents[1]/'scripts/build_grid_ranked_pairs.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_normalized_graph_ranking_swaps_donor_order_and_matches_compiled():
    repo=Path(__file__).parents[1]
    assoc=builder.fixed.fixed_single_gene(Graph.load(repo/'runs/grid-associative/artifact'),0)
    identity=builder.fixed.fixed_single_gene(Graph.load(repo/'runs/grid20/artifact'),0)
    train=[{'input':[[1,2]],'output':[[3,4]]}];query=[[2,1]]
    for left,right in ((identity,assoc),(assoc,identity)):
        graph=builder.ranked_pair(left,right);check_profile(graph,'ladder-micro/1')
        ref=GridPredictor(graph);compiled=GridPredictor(graph,backend='compiled')
        expected=[[[4,3]],[[2,1]]]
        assert ref(train,query)==expected
        assert compiled(train,query)==expected
        # New task resets scores and all donor state.
        second=[{'input':[[1,2]],'output':[[1,2]]}]
        assert ref(second,query)[0]==query
        assert ref(train,query)==expected


def test_shape_errors_rank_dimension_changing_gene_and_query_does_not_update_scores():
    repo=Path(__file__).parents[1]
    parent=Graph.load(repo/'runs/grid20/artifact')
    left=builder.fixed.fixed_single_gene(parent,0)
    right=builder.fixed.fixed_single_gene(parent,10)
    graph=builder.ranked_pair(left,right)
    train=[{'input':[[1,2,3],[4,5,6]],'output':[[1,4],[2,5],[3,6]]}]
    query=[[6,5,4],[3,2,1]]
    assert GridPredictor(graph)(train,query)==[[[6,3],[5,2],[4,1]],query]
