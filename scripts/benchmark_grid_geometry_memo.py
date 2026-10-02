# SPDX-License-Identifier: MIT-0
"""Paired local cache throughput diagnostic; no new model or benchmark tuning."""
import argparse,json,hashlib,time,os
from pathlib import Path
import numpy as np
import psutil
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from grid_geometry_memo import GeometryMemoGridPredictor
from measure_grid_expanded_capacity import family

def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('fresh output required')
 g=Graph.load(a.graph);tasks=family(3257000,2);a.out.mkdir()
 policy=dict(graph_hash=g.hash(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),memo_sha256=hashlib.sha256(Path('scripts/grid_geometry_memo.py').read_bytes()).hexdigest(),tasks=24,repetitions=3,seed=3257000,manifest=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),native_loaded=0,scope='local Windows paired throughput; no server speed claim',environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
 engines={'compiled':GridPredictor(g,backend='compiled'),'memo':GeometryMemoGridPredictor(g)}
 # Untimed warmup and parity check, including semantic execution counters.
 for t in tasks:
  x=t['test'][0]['input'];left=engines['compiled'](t['train'],x);right=engines['memo'](t['train'],x)
  assert left==right and engines['compiled'].last_exec==engines['memo'].last_exec
 rows=[]
 for rep in range(3):
  for arm in (('compiled','memo') if rep%2==0 else ('memo','compiled')):
   engine=engines[arm];wall=time.perf_counter();cpu=time.process_time();outputs=[]
   for t in tasks:outputs.append(engine(t['train'],t['test'][0]['input']))
   elapsed=time.perf_counter()-wall;cpu=time.process_time()-cpu
   rows.append(dict(rep=rep,arm=arm,wall_seconds=elapsed,cpu_seconds=cpu,tasks_per_second=24/elapsed,outputs_sha256=hashlib.sha256(json.dumps(outputs).encode()).hexdigest()))
 assert len({r['outputs_sha256'] for r in rows})==1
 info=psutil.Process().memory_info();result=dict(policy=policy,rows=rows,parity_queries=24,peak_rss_bytes=getattr(info,'peak_wset',None),resource_scope='whole process contains both engines; not isolated memory comparison',memo_hits=engines['memo']._compiled.memo_hits,memo_misses=engines['memo']._compiled.memo_misses,semantic_counters_are_not_physical_work=True)
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(rows=rows,peak_rss_bytes=result['peak_rss_bytes'],memo_hits=result['memo_hits'])))
if __name__=='__main__':main()
