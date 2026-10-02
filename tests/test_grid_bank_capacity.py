import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from measure_grid_bank_capacity import curriculum,selection_view


def test_capacity_curriculum_balanced_disjoint_and_selection_train_only():
    a=selection_view(2997000);b=curriculum(3007000)
    assert len(a)==len(b)==24
    assert all(set(t)=={'train'} for t in a)
    assert sum(len(t['train']) for t in a)==70
    names=[t.get('family',t.get('metadata',{}).get('family')) for t in b]
    assert len(set(names))==12 and all(names.count(f)==2 for f in names)
    assert all(x['train']!=y['train'] for x in a for y in b)
