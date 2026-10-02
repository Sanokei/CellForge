import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from demonstration_mode_blueprint import materialize_mode_blueprint
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from expanded_grid_genetics import instantiate
from priority_a_breeding import breed_champion


@pytest.mark.parametrize('backend',['reference','compiled'])
def test_same_query_has_different_demo_conditioned_transform(backend):
    directory=Path(__file__).parents[1]/'runs/priority-a-batch02-v1/775121-maker_repair-selected'
    if not (directory/'graph.json').exists():pytest.skip('research graph unavailable')
    g=materialize_mode_blueprint(Graph.load(directory),'a02_dense_crop_fill.gene')
    # Disable unrelated object/crop genes so another expert cannot satisfy
    # this test on behalf of the new context-controlled compound cell.
    alleles=['geometry_0']*9;alleles[2]='a02_dense_crop_fill.gene'
    g=instantiate(g,alleles)
    p=GridPredictor(g,backend=backend)
    query=[[0,0,0,0,0],[0,0,0,0,0],[0,0,4,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    crop_demo=dict(input=[[0,0,0,0],[0,2,3,0],[0,0,0,0]],output=[[2,3]])
    crop=p([crop_demo],query)
    assert [[4]] in crop
    fill_demo=dict(input=query,output=[[0,0,0,0,0],[0,0,4,0,0],[0,4,4,4,0],[0,0,4,0,0],[0,0,0,0,0]])
    assert fill_demo['output'] in p([fill_demo],query)
    assert [[4]] in p([crop_demo],query)  # state resets across tasks


def test_context_blueprint_breeds_with_original_archive_and_rejects_tampering():
    root=Path(__file__).parents[1]/'runs'
    path=root/'priority-a-batch02-v1/775121-maker_repair-selected'
    if not (path/'graph.json').exists():pytest.skip('research graph unavailable')
    champion=materialize_mode_blueprint(Graph.load(path),'a02_dense_crop_fill.gene')
    mate=Graph.load(root/'priority-a-v2/right')
    child=breed_champion(champion,mate,775131)
    assert child.defs['root']==champion.defs['root']
    assert child.meta['expanded_genome']['parent_hashes']==[champion.hash(),mate.hash()]
    bad=champion.copy();bad.defs['root']['nodes'][0]['args']['start']=99
    with pytest.raises(ValueError,match='semantics differ'):
        breed_champion(bad,mate,775131)
