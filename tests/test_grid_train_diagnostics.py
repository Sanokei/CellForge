import pytest
from cellforge.reference.grid_train_diagnostics import diagnose_train,pixel_comparison
def test_query_fields_rejected_before_predictor():
 def no_call(*a):raise AssertionError('must not call')
 with pytest.raises(ValueError,match='train-only'):diagnose_train({'train':[],'test':[{'output':'secret'}]},no_call)
def test_fixed_rank_and_shape_failure():
 task={'train':[{'input':[[1]],'output':[[2]]},{'input':[[3]],'output':[[4]]}]}
 r=diagnose_train(task,lambda d,x:([[[2]],[[0]]] if x==[[1]] else [[[0]],[[4]]]))
 assert not r['demonstration_consistent'] and r['failure_class']=='pixel-rule-or-routing'
 r=diagnose_train(task,lambda d,x:[[[0,0]]]);assert r['failure_class']=='observed-shape-mismatch'
def test_zero_is_not_padding_and_difference_sample_bounded():
 r=pixel_comparison([[0]],[[0,0]]);assert r['pixel_accuracy']==.5 and not r['shape_match'];assert r['difference_sample'][0]['expected'] is None
 r=pixel_comparison([[1]*10],[[0]*10]);assert len(r['difference_sample'])==8
