"""Synthetic 30x30 CPU cost; no ARC data or capability score."""
import argparse,hashlib,json,time,os
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor

def main():
    p=argparse.ArgumentParser(); p.add_argument('--artifact',required=True); p.add_argument('--out',required=True); p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if not 1<=args.repeats<=10: raise ValueError('bounded repeat count')
    graph=Graph.load(args.artifact); predictor=GridPredictor(graph)
    grid=(np.arange(900).reshape(30,30)%10).tolist(); demos=[{'input':grid,'output':grid}]
    times=[]; correct=0
    for _ in range(args.repeats):
        start=time.perf_counter(); output=predictor(demos,grid); times.append(time.perf_counter()-start); correct+=output[0]==grid
    rss=None
    if os.name!='nt':
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result={'protocol':'nca-synthetic-cpu/1','hash':graph.hash(),'recipe':graph.meta['nca_recipe'],'stats':graph.stats(),
            'microcells':sum(d['kind']=='cell' for d in graph.defs.values()),'shape':[30,30],
            'canonical_calls':args.repeats,'phases_per_call':2,'seconds':times,'calls_per_second':args.repeats/sum(times),
            'identity_calls':correct,'peak_rss_bytes':rss,'rss_scope':'whole probe process Linux ru_maxrss; unavailable on Windows',
            'last_exec':predictor.last_exec,'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'benchmark_tasks':0,'gradient_steps':0,
            'limitations':'synthetic identity/CPU diagnostic only; no demonstration adaptation or ARC score'}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
