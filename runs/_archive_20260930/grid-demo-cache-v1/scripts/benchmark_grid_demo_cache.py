# SPDX-License-Identifier: MIT-0
"""Frozen ranked-output differential and alternating local adaptation cache timing."""
import argparse,json,time,resource,hashlib
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from run_joint_layout_repair import corpus


class Counted(GridPredictor):
    def __init__(self,g,cached):
        super().__init__(g,backend='compiled',cache_demo_state=cached)
        self.calls=0;self.steps=0;self.hits=0
    def __call__(self,demos,query):
        result=super().__call__(demos,query)
        self.calls+=1;self.steps+=self.last_runtime_steps;self.hits+=self.last_demo_cache_hit
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise ValueError('fresh receipt required')
    g=Graph.load('runs/nine-slot-capacity-v1/nine-active');tasks=corpus(4637000,1)
    policy=dict(protocol='grid-demo-cache-benchmark/1',graph_hash=g.hash(),tasks=14,
        order=[False,True,True,False],seed=4637000,gradients=False,native_loaded=0,
        source_sha256=hashlib.sha256(Path('src/cellforge/runtime/grid.py').read_bytes()).hexdigest(),
        scope='same frozen predictions; single-entry demonstration state cache; no solver improvement')
    args.out.mkdir();(args.out/'policy.json').write_text(json.dumps(policy,indent=2))
    plain=Counted(g,False);cached=Counted(g,True);checks=0
    for task in tasks:
        for pair in task['train']+task['test']:
            assert plain(task['train'],pair['input'])==cached(task['train'],pair['input']);checks+=1
    rows=[];reference=None
    for enabled in policy['order']:
        predictor=Counted(g,enabled);start=time.perf_counter();cpu=time.process_time()
        score=score_with_consistency([(str(i),t) for i,t in enumerate(tasks)],predictor,record_tasks=True)
        if reference is None:reference=score['task_records']
        assert score['task_records']==reference
        rows.append(dict(cached=enabled,wall_seconds=time.perf_counter()-start,
            cpu_seconds=time.process_time()-cpu,predictor_calls=predictor.calls,
            runtime_steps=predictor.steps,cache_hits=predictor.hits,score=score))
    assert g.hash()==policy['graph_hash']
    result=dict(policy=policy,ranked_output_parity_checks=checks,rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(parity_checks=checks,rows=[{k:v for k,v in r.items() if k!='score'} for r in rows],rss=result['peak_rss_bytes'])))


if __name__=='__main__':main()
