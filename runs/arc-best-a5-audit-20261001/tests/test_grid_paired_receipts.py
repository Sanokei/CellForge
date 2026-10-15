import copy
import pytest
from cellforge.reference.grid_paired_receipts import compare_native


def receipt(solved,manifest='same'):
    records=[dict(task_id=k,solved=v,demo_consistent=v,demo_consistent_but_wrong=False,query_successes=[v]) for k,v in solved.items()]
    row=dict(split='native-development',task_records=records,tasks=len(records),tasks_solved=sum(solved.values()),
             test_inputs=len(records),test_inputs_correct=sum(solved.values()),demo_consistent_tasks=sum(solved.values()),demo_consistent_but_wrong_tasks=0)
    return dict(policy=dict(native_manifest_sha256=manifest,graph_hash='graph'),rows=[row])


def test_gains_and_regressions_are_both_reported():
    result=compare_native(receipt({'a':True,'b':False}),receipt({'a':False,'b':True}))
    assert result['gained']==['a'] and result['lost']==['b'] and result['net_gain']==0


def test_missing_or_tampered_receipts_cannot_claim_pairing():
    a=receipt({'a':True});b=copy.deepcopy(a)
    del b['rows'][0]['task_records']
    with pytest.raises(ValueError,match='missing'):compare_native(a,b)
    b=receipt({'a':True});b['rows'][0]['tasks_solved']=0
    with pytest.raises(ValueError,match='disagreement'):compare_native(a,b)
    with pytest.raises(ValueError,match='manifests'):compare_native(a,receipt({'a':True},'other'))


def test_duplicates_and_invalid_query_outcomes_rejected():
    a=receipt({'a':True});b=copy.deepcopy(a)
    b['rows'][0]['task_records']*=2
    with pytest.raises(ValueError,match='duplicate'):compare_native(a,b)
    b=copy.deepcopy(a);b['rows'][0]['task_records'][0]['query_successes']=[1]
    with pytest.raises(ValueError,match='boolean'):compare_native(a,b)
