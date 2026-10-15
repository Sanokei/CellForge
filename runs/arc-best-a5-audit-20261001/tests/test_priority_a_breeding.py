import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from priority_a_breeding import breed_champion
from cellforge.ir.graph import Graph
from cellforge.construction.transactions import apply_ops
from build_grid_periodic_marker_gene import build_periodic_marker_gene


def test_new_cell_weights_survive_two_parent_crossover():
    folder=Path(__file__).parents[1]/'runs/nine-slot-population-inventory-v1'
    if not (folder/'base/graph.json').exists():pytest.skip('local research inventory unavailable')
    mate=Graph.load(folder/'base');donor=build_periodic_marker_gene()
    champ,_=apply_ops(mate,[dict(op='import_slot_gene',**{'from':donor.hash()},name='periodic.gene',prefix='a01_',slot=8)],resolver=lambda _:donor)
    hashes=[champ.hash(),mate.hash()]
    retained=lost=0
    for seed in range(10):
        child=breed_champion(champ,mate,seed)
        assert child.meta['expanded_genome']['parent_hashes']==hashes
        for name,value in champ.params.items():np.testing.assert_array_equal(child.params[name],value)
        retained+=child.meta['expanded_genome']['alleles'][0]=='a01_periodic.gene'
        lost+=child.meta['expanded_genome']['alleles'][0]!= 'a01_periodic.gene'
    assert retained and lost
    assert [champ.hash(),mate.hash()]==hashes


def test_shared_weight_change_rejects_mate():
    folder=Path(__file__).parents[1]/'runs/nine-slot-population-inventory-v1'
    if not (folder/'base/graph.json').exists():pytest.skip('local research inventory unavailable')
    mate=Graph.load(folder/'base');champ=mate.copy()
    name=next(iter(champ.params));champ.params[name]=champ.params[name]+1
    with pytest.raises(ValueError,match='numerical semantics'):
        breed_champion(champ,mate,0)
