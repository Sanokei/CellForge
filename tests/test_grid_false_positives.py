from cellforge.reference.grid_false_positives import score_with_consistency


def task(demos, queries):
    return {'train': [{'input': [[a]], 'output': [[b]]} for a, b in demos],
            'test': [{'input': [[a]], 'output': [[b]]} for a, b in queries]}


def test_demo_fit_can_fail_queries_and_zero_denominator_is_unavailable():
    tasks = [('fit-fails', task([(1, 1), (2, 2)], [(3, 4)])),
             ('fit-solves', task([(1, 1)], [(3, 3)]))]
    result = score_with_consistency(tasks, lambda demos, grid: [grid])
    assert result['tasks_solved'] == 1
    assert result['demo_consistent_tasks'] == 2
    assert result['demo_consistent_but_wrong_tasks'] == 1
    assert result['false_positive_given_consistency'] == .5
    result = score_with_consistency([('none', task([(1, 2)], [(3, 4)]))], lambda demos, grid: [grid])
    assert result['false_positive_given_consistency'] is None


def test_consistency_requires_one_rank_for_all_demos_but_queries_allow_either():
    def predictor(demos, grid):
        value = grid[0][0]
        return ([[value]], [[value + 1]])
    mixed = task([(1, 1), (2, 3)], [(3, 3), (4, 5)])
    result = score_with_consistency([('mixed', mixed)], predictor)
    assert result['demo_consistent_tasks'] == 0
    assert result['tasks_solved'] == 1
    assert result['test_inputs_correct'] == 2


def test_invalid_third_attempt_and_demo_failure_are_counted():
    result = score_with_consistency([('bad', task([(1, 1)], [(2, 2)]))],
                                    lambda demos, grid: [grid, grid, grid])
    assert result['invalid_outputs'] == 1
    assert result['invalid_demo_reconstructions'] == 1
    assert result['tasks_solved'] == 0


def test_task_receipts_preserve_aggregate_scores_and_predictor_calls():
    tasks=[('partial',task([(1,1)],[(2,2),(3,4)])),
           ('success',task([(1,1)],[(2,2)]))]
    calls=[]
    def predict(demos,grid):
        calls.append((demos,grid));return [grid]
    plain=score_with_consistency(tasks,predict);count=len(calls);calls.clear()
    detailed=score_with_consistency(tasks,predict,record_tasks=True)
    assert len(calls)==count==5
    for key in ('tasks_solved','test_inputs_correct','demo_consistent_tasks','demo_consistent_but_wrong_tasks'):
        assert detailed[key]==plain[key]
    assert 'task_records' not in plain
    assert detailed['task_records'][0]==dict(task_id='partial',solved=False,demo_consistent=True,
        consistent_ranks=[0],demo_consistent_but_wrong=True,query_successes=[True,False])
    assert detailed['task_records'][1]['solved']
