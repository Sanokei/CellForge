import numpy as np
import pytest
from cellforge.reference.grid_curriculum import (FAMILIES,make_grid_task,build_curriculum,
                                                validate_partition_disjoint,runtime_inputs,classify_train_families,task_io_fingerprint,generate_task)
from cellforge.reference.arc_grid_operator import validate_grid


@pytest.mark.parametrize('family',FAMILIES)
def test_reproducibility_bounds_and_independent_geometry(family):
    for difficulty in ('tiny','small'):
        record=make_grid_task(family,13,difficulty=difficulty)
        assert record==make_grid_task(family,13,difficulty=difficulty)
        task=record['task']; assert len(task['train'])==3 and len(task['test'])==1
        for pair in task['train']+task['test']:
            x,y=pair['input'],pair['output']; validate_grid(x); validate_grid(y)
            assert 3<=len(x)<=9 and 3<=len(x[0])<=9
            if family=='identity': assert y==x
            if family=='reflect_horizontal': assert y==[list(reversed(row)) for row in x]
            if family=='reflect_vertical': assert y==list(reversed(x))
            if family=='transpose': assert y==[list(row) for row in zip(*x)]
            if family=='tiling2x': assert y==[row+row for row in x]+[row+row for row in x]
            if family=='bbox_crop_bg0':
                occupied=[(r,c) for r,row in enumerate(x) for c,v in enumerate(row) if v]
                top=min(r for r,c in occupied); bottom=max(r for r,c in occupied)
                left=min(c for r,c in occupied); right=max(c for r,c in occupied)
                assert y==[row[left:right+1] for row in x[top:bottom+1]]
            if family=='local_fill':
                for r,row in enumerate(x):
                    for c,v in enumerate(row):
                        neighborhood=[x[a][b] for a,b in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if 0<=a<len(x) and 0<=b<len(row)]
                        assert y[r][c]==(v if v else max(neighborhood))


def test_color_permutation_identifiable_and_query_consistent():
    for seed in range(20):
        record=make_grid_task('color_permutation',seed)
        mapping={}
        for pair in record['task']['train']:
            for a,b in zip(np.array(pair['input']).flat,np.array(pair['output']).flat):
                if int(a) in mapping: assert mapping[int(a)]==int(b)
                mapping[int(a)]=int(b)
        assert set(mapping)==set(range(10)) and set(mapping.values())==set(range(10))
        query=record['task']['test'][0]
        assert query['output']==[[mapping[v] for v in row] for row in query['input']]


def test_fixed_partitions_disjoint_and_hash_reproducible():
    selection=build_curriculum('selection',per_family_bin=3)
    validation=build_curriculum('validation',per_family_bin=3)
    assert selection==build_curriculum('selection',per_family_bin=3)
    assert selection['tasks']==validation['tasks']==48
    assert validate_partition_disjoint(selection,validation)
    with pytest.raises(ValueError): validate_partition_disjoint(selection,dict(selection,partition='validation'))


def test_runtime_view_excludes_query_target_and_owns_no_rule():
    record=make_grid_task('transpose',19)
    train,query=runtime_inputs(record)
    assert train==record['task']['train'] and query==record['task']['test'][0]['input']
    query[0][0]=99
    assert record['task']['test'][0]['input'][0][0]!=99
    assert record['provenance']['benchmark_data_read'] is False


def test_annotations_multilabel_consistent_bijection_and_no_query_access():
    identity=[{'input':[[0,1],[2,3]],'output':[[0,1],[2,3]]}]
    labels=classify_train_families(identity)
    assert 'identity' in labels and 'color_permutation' in labels
    noninjective=[{'input':[[0,1]],'output':[[2,2]]}]
    assert 'color_permutation' not in classify_train_families(noninjective)
    inconsistent=[{'input':[[0,1]],'output':[[2,3]]},{'input':[[0,1]],'output':[[4,3]]}]
    assert 'color_permutation' not in classify_train_families(inconsistent)
    for family in FAMILIES:
        record=make_grid_task(family,51)
        assert family in record['train_family_annotations']
        original=classify_train_families(record['task']['train'])
        record['task']['test'][0]['output']=[[9]]
        assert classify_train_families(record['task']['train'])==original


def test_many_random_tasks_have_distinct_exact_io_across_splits():
    a=build_curriculum('selection',per_family_bin=15)
    b=build_curriculum('validation',per_family_bin=15)
    assert validate_partition_disjoint(a,b)
    assert len({task_io_fingerprint(i) for i in a['items']+b['items']})==480
    duplicate=dict(b,items=[dict(b['items'][0],task=a['items'][0]['task'],task_sha256=a['items'][0]['task_sha256'])]+b['items'][1:])
    with pytest.raises(ValueError): validate_partition_disjoint(a,duplicate)


def test_operator_raw_task_api_and_alternating_bins():
    for seed,difficulty in ((731000,'tiny'),(731001,'small')):
        task=generate_task('identity',seed)
        assert len(task['train'])==3 and len(task['test'])==1
        assert task['metadata']['difficulty']==difficulty
        assert 'identity' in classify_train_families(task['train'])
