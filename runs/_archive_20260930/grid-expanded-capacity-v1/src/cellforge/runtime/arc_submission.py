"""Offline CPU ARC two-attempt submission packaging through a compiled graph.

Only demonstrated input/output pairs and query inputs enter GridPredictor.
Task identifiers are output dictionary keys only. Query labels are ignored.
No fallback is invented: malformed graph outputs or runtime errors abort.
Competition eligibility/upload/authentication are separate owner actions.
"""
from __future__ import annotations
import copy
import sys
import time
from .grid import GridPredictor


class ARCSubmissionError(ValueError):
    pass


def validate_grid(grid):
    if not isinstance(grid,list) or not 1<=len(grid)<=30:raise ARCSubmissionError('grid height must be1..30')
    if not isinstance(grid[0],list) or not 1<=len(grid[0])<=30:raise ARCSubmissionError('grid width must be1..30')
    width=len(grid[0])
    if any(not isinstance(row,list) or len(row)!=width or any(type(color)!=int or not 0<=color<=9 for color in row) for row in grid):
        raise ARCSubmissionError('grid must be rectangular with integer colors0..9')
    return grid


def peak_rss():
    try:
        import resource
        amount=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'peak_rss_bytes':int(amount if sys.platform=='darwin' else amount*1024),
                'peak_rss_scope':'process lifetime high-water mark, including setup',
                'resource_unavailable_reason':None}
    except (ImportError,AttributeError,OSError) as error:
        return {'peak_rss_bytes':None,'peak_rss_scope':None,
                'resource_unavailable_reason':type(error).__name__+': resource.getrusage unavailable on this platform'}


def run_arc_submission(graph,tasks):
    """Return (submission, metrics), fail atomically before emitting outputs.

    Single process, one compiled GridPredictor reused; each query resets graph
    recurrent state and processes its task demonstrations anew. Caller owns a
    hard CPU/wall timeout: this function provides no speculative timeout grid.
    """
    if not isinstance(tasks,dict) or not tasks:raise ARCSubmissionError('nonempty task-id dictionary required')
    start=time.perf_counter();cpu_start=time.process_time()
    frozen_hash=graph.hash();predictor=GridPredictor(graph,backend='compiled')
    compile_seconds=time.perf_counter()-start
    submission={};query_count=duplicates=node_execs=macs=peak_elements=0;predict_seconds=0.
    for task_id,task in tasks.items():
        if not isinstance(task_id,str) or not isinstance(task,dict):raise ARCSubmissionError('string task-id and task object required')
        train=task.get('train');test=task.get('test')
        if not isinstance(train,list) or not train or not isinstance(test,list) or not test:raise ARCSubmissionError('nonempty train/test arrays required')
        demonstrations=[]
        for pair in train:
            if not isinstance(pair,dict) or 'input' not in pair or 'output' not in pair:raise ARCSubmissionError('demonstration input/output required')
            demonstrations.append({'input':validate_grid(pair['input']),'output':validate_grid(pair['output'])})
        answers=[]
        for query in test:
            if not isinstance(query,dict) or 'input' not in query:raise ARCSubmissionError('query input required')
            # Deliberately do not inspect query.output, task family or task id
            # while constructing graph inputs.
            query_input=validate_grid(query['input'])
            tick=time.perf_counter();attempts=predictor(demonstrations,query_input);predict_seconds+=time.perf_counter()-tick
            if not isinstance(attempts,list) or len(attempts) not in (1,2):raise ARCSubmissionError('graph must return one or two attempts')
            if len(attempts)!=predictor.attempts:raise ARCSubmissionError('returned attempt count differs from graph declaration')
            for grid in attempts:validate_grid(grid)
            if len(attempts)==1:attempts=[attempts[0],copy.deepcopy(attempts[0])];duplicates+=1
            answers.append({'attempt_1':attempts[0],'attempt_2':attempts[1]});query_count+=1
            stats=predictor.last_exec or {}
            node_execs+=stats.get('node_execs',0);macs+=stats.get('macs',0)
            peak_elements=max(peak_elements,stats.get('peak_tensor_elements',0))
        submission[task_id]=answers
    if graph.hash()!=frozen_hash:raise ARCSubmissionError('graph changed during submission generation')
    seconds=time.perf_counter()-start
    metrics={'protocol':'offline-arc-cpu-submission/1','graph_hash':frozen_hash,'backend':'compiled',
        'task_count':len(tasks),'query_count':query_count,'model_attempts':predictor.attempts,'duplicated_single_attempt_queries':duplicates,
        'compile_seconds':compile_seconds,'predict_seconds':predict_seconds,'wall_seconds':seconds,'cpu_seconds':time.process_time()-cpu_start,
        'tasks_per_second':len(tasks)/seconds,'queries_per_second':query_count/seconds,
        'node_execs':node_execs,'macs':macs,'peak_tensor_elements':peak_elements,
        'timeout_policy':'caller-enforced hard timeout; no graph or host fallback after runtime error',
        'competition_scope':'JSON interface only; no competition validity, private accuracy, submission or authentication proof',**peak_rss()}
    return submission,metrics
