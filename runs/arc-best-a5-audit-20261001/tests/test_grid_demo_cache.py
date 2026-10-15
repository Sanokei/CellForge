"""Opt-in adaptation reuse must preserve task and query isolation."""
from pathlib import Path
import pytest
from cellforge.ir.graph import Graph,GraphError
from cellforge.runtime.grid import GridPredictor


def test_demo_cache_ranked_parity_reset_input_mutation_and_graph_guard():
    graph=Graph.load(Path(__file__).parents[1]/'runs/nine-slot-capacity-v1/nine-active')
    plain=GridPredictor(graph,backend='compiled')
    cached=GridPredictor(graph,backend='compiled',cache_demo_state=True)
    for background in (3,9,3):
        demos=[{'input':[[1,0]],'output':[[1,background]]},
               {'input':[[0,1]],'output':[[background,1]]}]
        for index,query in enumerate(([[0,1]],[[1],[0]],[[0,1]])):
            assert cached(demos,query)==plain(demos,query)
            assert cached.last_demo_cache_hit==(index>0)
            assert cached.last_runtime_steps==(1 if index>0 else 3)
            assert not cached._demo_cache[1].flags.writeable
        demos[0]['output'][0][1]=8
        assert cached(demos,[[0,1]])==plain(demos,[[0,1]])
        assert not cached.last_demo_cache_hit
    graph.params[next(iter(graph.params))].flat[0]+=1
    with pytest.raises(GraphError,match='changed'):cached(demos,[[1]])


def test_demo_cache_requires_compiled_backend_and_nonempty_demos():
    graph=Graph.load(Path(__file__).parents[1]/'runs/nine-slot-capacity-v1/nine-active')
    with pytest.raises(ValueError):GridPredictor(graph,cache_demo_state=True)
    predictor=GridPredictor(graph,backend='compiled',cache_demo_state=True)
    with pytest.raises(GraphError):predictor([],[[1]])


def test_query_state_updates_never_enter_demonstration_cache():
    import numpy as np
    from test_grid_runtime import recolor_graph
    graph=recolor_graph()
    graph.meta['state_dim']=1
    graph.defs['root']['nodes'].append(dict(id='advance',use='op:bias',
        **{'in':{'x':'$state'},'args':{'c':1}}))
    graph.defs['root']['outputs']['state']='advance'
    predictor=GridPredictor(graph,backend='compiled',cache_demo_state=True)
    demos=[{'input':[[1]],'output':[[2]]},{'input':[[3]],'output':[[4]]}]
    for query in ([[2]],[[5]],[[2]]):
        predictor(demos,query)
        np.testing.assert_array_equal(predictor._demo_cache[1],[[2]])
