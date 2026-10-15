import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from build_grid_rule_genes import RULES, build_rule_gene
from cellforge.runtime.grid import GridPredictor, pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter


@pytest.mark.parametrize('rule', RULES)
def test_every_rule_backend_agreement_and_padding(rule):
    g = build_rule_gene(rule, iterations=8, table={0:5, 2:6})
    x = [[2,4,0,2],[2,0,2,2],[0,2,4,0]]
    colors, _, h, w = pack_grid(x)
    inputs = {'input':colors,'input_height':h,'input_width':w}
    a = Interpreter(g).run_bundle('rule.gene',inputs,'/rule')
    b = CompiledInterpreter(g).run_bundle('rule.gene',inputs,'/rule')
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    assert not a['colors'][0,len(x):].any()
    assert not a['colors'][0,:,len(x[0]):].any()
    np.testing.assert_array_equal(a['colors'].sum(axis=-1),colors.sum(axis=-1))


@pytest.mark.parametrize('rule,x,expected,options', [
    ('touching',[[2,4,2,2]],[[3,4,3,2]],{}),
    ('not_touching',[[2,4,2,2]],[[2,4,2,3]],{}),
    ('isolated',[[2,0,2,2]],[[3,0,2,2]],{}),
    ('interior',[[2,2,2],[2,2,2],[2,2,2]],[[2,2,2],[2,3,2],[2,2,2]],{}),
    ('boundary',[[2,2,2],[2,2,2],[2,2,2]],[[3,3,3],[3,2,3],[3,3,3]],{}),
    ('border',[[2,2,2],[2,2,2],[2,2,2]],[[3,3,3],[3,2,3],[3,3,3]],{}),
    ('not_border',[[2,2,2],[2,2,2],[2,2,2]],[[2,2,2],[2,3,2],[2,2,2]],{}),
    ('row',[[2,2],[2,2]],[[2,2],[3,3]],{}),
    ('column',[[2,2],[2,2]],[[2,3],[2,3]],{}),
    ('top',[[2,2],[2,2]],[[3,3],[2,2]],{}),
    ('left',[[2,2],[2,2]],[[3,2],[3,2]],{}),
    ('column_parity',[[2,2,2]],[[2,3,2]],{}),
    ('checkerboard',[[2,2],[2,2]],[[2,3],[3,2]],{}),
    ('color_size',[[2,0,2]],[[3,0,3]],{'value':2}),
    ('row_minority',[[2,4,4]],[[3,4,4]],{}),
    ('column_majority',[[2,4],[2,4],[4,2]],[[3,4],[3,4],[4,2]],{}),
    ('column_minority',[[2,4],[4,2],[4,2]],[[3,4],[4,2],[4,2]],{}),
    ('neighbor_majority',[[2,2,4]],[[3,3,4]],{}),
    ('component_border',[[0,0,0],[0,2,0],[0,0,2]],[[0,0,0],[0,2,0],[0,0,3]],{}),
    ('component_not_border',[[0,0,0],[0,2,0],[0,0,2]],[[0,0,0],[0,3,0],[0,0,2]],{}),
    ('neighbor_count',[[4,2,4],[0,2,0]],[[4,3,4],[0,2,0]],{'value':2}),
    ('row_parity',[[2,2],[2,2],[2,2]],[[3,3],[2,2],[3,3]],{'value':0}),
    ('bottom',[[2,2],[2,2]],[[2,2],[3,3]],{}),
    ('right',[[2,2],[2,2]],[[2,3],[2,3]],{}),
    ('majority',[[2,2,4]],[[3,3,4]],{}),
    ('minority',[[2,4,4]],[[3,4,4]],{}),
    ('minority',[[2,4]],[[3,4]],{}),
    ('row_majority',[[2,2,4],[2,4,4]],[[3,3,4],[2,4,4]],{}),
    ('neighbor_minority',[[4,2,4],[0,2,0]],[[4,3,4],[0,3,0]],{}),
    ('local_table',[[2,4,2,2]],[[6,4,6,5]],{'table':{0:5,1:6}}),
    ('flood_marker',[[4,2,2,0,2]],[[4,3,3,0,2]],{}),
    ('flood_border',[[0,0,0],[0,2,0],[0,0,0]],[[0,0,0],[0,2,0],[0,0,0]],{}),
    ('enclosed',[[4,4,4],[4,2,4],[4,4,4]],[[4,4,4],[4,3,4],[4,4,4]],{}),
    ('component_size',[[2,2,0,2]],[[3,3,0,2]],{'value':2}),
    ('component_parity',[[2,2,0,2]],[[2,2,0,3]],{'value':1}),
    ('component_touching',[[4,2,2,0,2]],[[4,3,3,0,2]],{}),
    ('component_not_touching',[[4,2,2,0,2]],[[4,2,2,0,3]],{}),
    ('component_largest',[[2,2,0,2]],[[3,3,0,2]],{}),
    ('component_smallest',[[2,2,0,2]],[[2,2,0,3]],{}),
    ('contained_bbox',[[4,0,4],[2,2,2],[4,0,4]],[[4,0,4],[2,3,2],[4,0,4]],{}),
    ('touching',[[2]],[[2]],{'marker':0}),
    ('touching',[[0,4]],[[3,4]],{'source':0}),
    ('flood_marker',[[4,2,2,2,2]],[[4,2,2,2,2]],{'iterations':1}),
    ('component_size',[[2,2,2,2]],[[2,2,2,2]],{'iterations':1,'value':4}),
])
def test_independent_semantic_examples(rule,x,expected,options):
    g = build_rule_gene(rule,**options)
    for backend in ('reference','compiled'):
        assert GridPredictor(g,backend=backend)([{'input':[[0]],'output':[[0]]}],x) == [expected]


def test_diagonal_connectivity():
    x = [[4,0,0],[0,2,0],[0,0,2]]
    for connectivity, expected in [(4,x),(8,[[4,0,0],[0,3,0],[0,0,3]])]:
        assert GridPredictor(build_rule_gene('flood_marker',connectivity=connectivity))([{'input':[[0]],'output':[[0]]}],x) == [expected]


@pytest.mark.parametrize('options', [{'source':True},{'target':10},{'connectivity':6},
    {'iterations':0},{'value':-1},{'table':{5:1}}])
def test_invalid_configuration(options):
    with pytest.raises(ValueError): build_rule_gene('touching',**options)


def test_clusters_compose_and_artifact_roundtrip(tmp_path):
    from build_grid_rule_genes import add_rule_cluster
    from cellforge.ir.graph import Graph
    from build_grid_local_fill_gene import Nodes
    g = Graph(meta={'task':'grid'})
    first = add_rule_cluster(g,'touching',source=2,target=3,marker=4,prefix='first')
    second = add_rule_cluster(g,'color_size',source=3,target=5,value=2,prefix='second')
    n = Nodes()
    a = n.add(first,{p:'$'+p for p in ('input','input_height','input_width')})
    b = n.add(second,{'input':a+'.colors','input_height':a+'.height','input_width':a+'.width'})
    g.defs['chain'] = n.bundle('cluster',['input','input_height','input_width'],{'colors':b+'.colors'})
    g.entries = {'grid_step':'chain'}
    g.validate()
    x,_,h,w = pack_grid([[2,4,2]])
    result = CompiledInterpreter(g).run_bundle('chain',{'input':x,'input_height':h,'input_width':w},'/chain')
    assert result['colors'][0,0,:3].argmax(axis=-1).tolist() == [5,4,5]
    artifact = build_rule_gene('local_table',table={0:5,1:6})
    artifact.save(tmp_path/'gene')
    loaded = Graph.load(tmp_path/'gene')
    assert artifact.hash() == loaded.hash()
