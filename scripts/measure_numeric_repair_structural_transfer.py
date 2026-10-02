# SPDX-License-Identifier: MIT-0
"""Structural transfer report for frozen authored numeric repairs."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency

def structural_examples(seed,count):
 rng=np.random.default_rng(seed);tasks=[]
 for i in range(count):
  pairs=[]
  for j in range(3):
   h,w=map(int,rng.integers(3,16,size=2));x=rng.choice([0,1,2,4,5,6,7,8,9],size=(h,w),p=[.1,.1,.2,.3,.06,.06,.06,.06,.06])
   # Guarantee one positive instance; evaluate every other source simultaneously.
   r=int(rng.integers(h));c=int(rng.integers(1,w-1));x[r,c]=2;x[r,c-1]=4;x[r,c+1]=4
   y=x.copy()
   for row in range(h):
    for col in range(w):
     if x[row,col]==2 and 0<col<w-1 and x[row,col-1]==4 and x[row,col+1]==4:y[row,col]=3
   pairs.append(dict(input=x.tolist(),output=y.tolist()))
  tasks.append(dict(train=pairs[:2],test=pairs[2:]))
 return tasks

def main():
 import psutil
 out=Path('runs/numeric-repair-structural-transfer-v1')
 if out.exists():raise ValueError('fresh output required')
 directory=Path('runs/numeric-weight-repair-prototype-v3');hashes=json.loads((directory/'winners-frozen.json').read_text());graphs={a:Graph.load(directory/f'{a}-round-2') for a in hashes};assert all(g.hash()==hashes[a] for a,g in graphs.items())
 tasks=structural_examples(3387000,100);out.mkdir();policy=dict(protocol='numeric-repair-structural-transfer/1',seed=3387000,tasks=100,manifest=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),hashes=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gradients=False,native_loaded=0,scope='new structural synthetic report; author knows target rule, not causal feedback experiment')
 (out/'policy.json').write_text(json.dumps(policy,indent=2));start=time.perf_counter();rows=[]
 for arm,g in graphs.items():
  rows.append(dict(arm=arm,score=score_with_consistency([(str(i),t) for i,t in enumerate(tasks)],GridPredictor(g,backend='compiled'),record_tasks=True)))
 # Reference parity on ten independently generated structural examples.
 g=graphs['maker_repair'];plain=GridPredictor(g,backend='reference');fast=GridPredictor(g,backend='compiled')
 for t in structural_examples(3397000,10):assert plain(t['train'],t['test'][0]['input'])==fast(t['train'],t['test'][0]['input'])
 result=dict(policy=policy,rows=rows,reference_parity_tasks=10,wall_seconds=time.perf_counter()-start,peak_rss_bytes=psutil.Process().memory_info().peak_wset,resource_scope='local Windows whole process, not server performance')
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(rows=[dict(arm=x['arm'],solved=x['score']['tasks_solved'],wrong=x['score']['demo_consistent_but_wrong_tasks']) for x in rows],wall_seconds=result['wall_seconds'])))
if __name__=='__main__':main()
