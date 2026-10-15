import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from priority_a_numeric_breeding import breed_numeric_champion
from priority_a_breeding import breed_champion
from cellforge.ir.graph import Graph

def test_literal_cell_weight_mutations_can_breed_with_real_lineage():
    mate=Graph.load(Path(__file__).parents[1]/'runs/priority-a-v2/right')
    champion=mate.copy();names=['bank_adjacency_recolor.marker.W','bank_adjacency_recolor.marker.b']
    for name in names:champion.params[name]+=1
    hashes=[champion.hash(),mate.hash()]
    with pytest.raises(ValueError,match='numerical semantics'):breed_champion(champion,mate,1)
    observed=set()
    for seed in range(8):
        child=breed_numeric_champion(champion,mate,seed)
        group=next(g for g in child.meta['expanded_genome']['numeric_donors'] if names[0] in g['parameters'])
        assert names[1] in group['parameters']
        observed.add(group['donor'])
        source=(champion,mate)[group['donor']]
        for name in names:np.testing.assert_array_equal(child.params[name],source.params[name])
        assert child.meta['expanded_genome']['parent_hashes']==hashes
        assert child.meta['expanded_genome']['numeric_donors']==breed_numeric_champion(champion,mate,seed).meta['expanded_genome']['numeric_donors']
    assert observed=={0,1}
    assert [champion.hash(),mate.hash()]==hashes

def test_shape_change_is_rejected():
    mate=Graph.load(Path(__file__).parents[1]/'runs/priority-a-v2/right');champion=mate.copy()
    champion.params['bank_adjacency_recolor.marker.W']=np.ones((1,11))
    with pytest.raises(ValueError):breed_numeric_champion(champion,mate,1)
