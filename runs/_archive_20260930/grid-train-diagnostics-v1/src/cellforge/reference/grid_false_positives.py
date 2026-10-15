# SPDX-License-Identifier: MIT-0
"""Evaluation-only exact demo-consistency and false-positive accounting.

The predictor receives supplied demonstrations and one input. Query targets and
consistency verdicts are never provided to inference or used to choose a graph.
This rechecks complete demonstrations under final task adaptation, not an online
router's proxy loss. It is deliberately additional diagnostic work.
"""
from itertools import islice
import time
from .arc_grid_operator import validate_grid


def attempts(predictor, demonstrations, grid):
    values = list(islice(predictor(demonstrations, grid), 3))
    if not 1 <= len(values) <= 2:
        raise ValueError('one or two grid attempts required')
    return [validate_grid(value) for value in values]


def score_with_consistency(tasks, predictor):
    total = solved = consistent = wrong = correct_inputs = inputs = invalid = invalid_demo = 0
    query_wall = query_cpu = audit_wall = audit_cpu = 0.
    for _identity, task in tasks:
        demos = task['train']
        if not demos or not task['test']:
            raise ValueError('nonempty demonstrations and queries required')
        for demo in demos:
            validate_grid(demo['input']); validate_grid(demo['output'])
        # Rank identities are checked across ALL demonstrations. No per-demo
        # cherry-picking of different ranks counts as a consistent candidate.
        matching_ranks = {0, 1}
        tick = time.perf_counter(); cpu = time.process_time()
        for demo in demos:
            try:
                values = attempts(predictor, demos, demo['input'])
                expected = validate_grid(demo['output'])
                matching_ranks.intersection_update(i for i, value in enumerate(values) if value == expected)
            except (ValueError, KeyError, IndexError, TypeError):
                invalid_demo += 1; matching_ranks.clear()
        audit_wall += time.perf_counter() - tick; audit_cpu += time.process_time() - cpu
        demo_consistent = bool(matching_ranks)
        consistent += int(demo_consistent)
        tick = time.perf_counter(); cpu = time.process_time(); task_solved = True
        for query in task['test']:
            validate_grid(query['input']); expected = validate_grid(query['output']); inputs += 1
            try:
                # ARC allows either rank independently for each query input.
                success = expected in attempts(predictor, demos, query['input'])
            except (ValueError, KeyError, IndexError, TypeError):
                invalid += 1; success = False
            correct_inputs += int(success); task_solved &= success
        query_wall += time.perf_counter() - tick; query_cpu += time.process_time() - cpu
        total += 1; solved += int(task_solved); wrong += int(demo_consistent and not task_solved)
    return {'protocol': 'grid-exact-demo-consistency/1', 'tasks': total, 'tasks_solved': solved,
            'test_inputs': inputs, 'test_inputs_correct': correct_inputs, 'invalid_outputs': invalid,
            'invalid_demo_reconstructions': invalid_demo, 'demo_consistent_tasks': consistent,
            'demo_consistent_but_wrong_tasks': wrong,
            'false_positive_given_consistency': wrong / consistent if consistent else None,
            'false_positive_fraction_all_tasks': wrong / total if total else None,
            'inference_wall_seconds': query_wall, 'inference_cpu_seconds': query_cpu,
            'inference_cpu_seconds_per_task': query_cpu / total if total else None,
            'tasks_per_second': total / query_wall if query_wall else None,
            'consistency_audit_wall_seconds': audit_wall, 'consistency_audit_cpu_seconds': audit_cpu,
            'consistency_scope': 'a fixed ranked output must reconstruct every complete demonstration after all demo adaptation; not the online routing proxy',
            'limitation': 'rank consistency assumes stable candidate semantics across inputs; runtime fixtures/route traces must establish this for an actual model'}
