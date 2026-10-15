import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from census_grid_train_structure import census_train,transform


def test_tile_and_pixel_repeat_are_distinct_and_all_demos_required():
    x=[[1,2],[3,4]]
    tiled=transform(x,'tile2').tolist(); repeated=transform(x,'repeat2').tolist()
    assert tiled!=repeated
    assert census_train({'train':[{'input':x,'output':tiled}]})['common_exact_rules']==['tile2']
    assert census_train({'train':[{'input':x,'output':tiled},{'input':x,'output':repeated}]})['common_exact_rules']==[]


def test_majority_crop_and_rejection_of_query_fields():
    x=[[5,5,5],[5,0,5],[5,5,5]]
    assert transform(x,'crop_majority').tolist()==[[0]]
    assert transform(x,'crop_zero').tolist()==x
    with pytest.raises(ValueError):census_train({'train':[{'input':x,'output':x}],'test':[]})


def test_color_associations_must_be_consistent_across_demonstrations():
    pair={'input':[[1,2]],'output':[[3,4]]}
    assert 'identity' in census_train({'train':[pair]})['common_rules_with_color_association']
    conflicting={'input':[[1,2]],'output':[[5,4]]}
    assert 'identity' not in census_train({'train':[pair,conflicting]})['common_rules_with_color_association']


def test_object_semantics_connectivity_ties_and_nonzero_background():
    x=[[5,5,5,5,5],[5,1,2,5,3],[5,1,5,5,5],[5,5,5,5,5]]
    assert transform(x,'object_largest_mono').tolist()==[[1],[1]]
    assert transform(x,'object_largest_any').tolist()==[[1,2],[1,5]]
    assert transform(x,'object_smallest_mono').tolist()==[[2]]
    assert transform([[0,1],[1,0]],'object_largest_mono').tolist()==[[1]]
    assert transform([[7,7]],'object_largest_any').tolist()==[[7,7]]


def test_object_removal_and_retention_keep_canvas_and_other_objects_distinct():
    x=[[0,0,0,0],[0,2,2,0],[0,0,0,3]]
    assert transform(x,'erase_largest_mono').tolist()==[[0,0,0,0],[0,0,0,0],[0,0,0,3]]
    assert transform(x,'retain_largest_mono').tolist()==[[0,0,0,0],[0,2,2,0],[0,0,0,0]]
    assert transform(x,'erase_smallest_mono').tolist()==transform(x,'retain_largest_mono').tolist()


def test_containment_uses_four_connectivity_and_preserves_outside_background():
    x=[[0,0,0,0,0],[0,2,2,2,0],[0,2,0,2,0],[0,2,2,2,0],[0,0,0,0,0]]
    y=transform(x,'fill_holes_zero')
    assert y[2,2]==2 and y[0,0]==0
    x[1][2]=0
    assert transform(x,'fill_holes_zero').tolist()==x
    assert transform([[7,7]],'fill_holes_majority').tolist()==[[7,7]]
