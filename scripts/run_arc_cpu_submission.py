"""Offline single-process CPU ARC JSON runner; no competition upload.

Reuse saved graph directory plus CellForge src and this script; Python3.10+
and NumPy required. The owner must attach these files to an offline Kaggle
notebook, verify current competition rules/runtime/data paths, and submit it
separately. This CLI neither authenticates nor proves a valid competition entry.
"""
import os
# Set before importing NumPy/runtime. No worker pools or model downloads.
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cellforge.ir.graph import Graph
from cellforge.runtime.arc_submission import run_arc_submission,peak_rss
import cellforge.runtime.arc_submission as adapter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',required=True);p.add_argument('--tasks',required=True)
    p.add_argument('--out',required=True);p.add_argument('--metrics',required=True);args=p.parse_args()
    out=Path(args.out);metrics_path=Path(args.metrics)
    if out.resolve()==metrics_path.resolve():raise ValueError('submission and metrics paths must differ')
    if out.exists() or metrics_path.exists():raise ValueError('preserve prior outputs; fresh output paths required')
    start=time.perf_counter();cpu_start=time.process_time()
    raw=Path(args.tasks).read_bytes()
    provenance={'input_sha256':hashlib.sha256(raw).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'adapter_sha256':hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
        'thread_environment':{name:os.environ[name] for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    try:
        graph=Graph.load(args.graph);provenance['graph_hash']=graph.hash()
        submission,metrics=run_arc_submission(graph,json.loads(raw))
    except Exception as error:
        failure={'protocol':'offline-arc-cpu-submission/1','status':'failed','error_type':type(error).__name__,'error':str(error),
                 'total_wall_seconds':time.perf_counter()-start,'total_cpu_seconds':time.process_time()-cpu_start,**provenance,**peak_rss()}
        metrics_path.parent.mkdir(parents=True,exist_ok=True);metrics_path.write_text(json.dumps(failure,indent=2)+'\n')
        raise
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(submission)+'\n')
    metrics.update(provenance);metrics.update({'status':'complete','output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'total_wall_seconds':time.perf_counter()-start,'total_cpu_seconds':time.process_time()-cpu_start})
    metrics_path.parent.mkdir(parents=True,exist_ok=True);metrics_path.write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({'tasks':metrics['task_count'],'queries':metrics['query_count'],'graph_hash':metrics['graph_hash'],'wall_seconds':metrics['total_wall_seconds']}))


if __name__=='__main__':main()
