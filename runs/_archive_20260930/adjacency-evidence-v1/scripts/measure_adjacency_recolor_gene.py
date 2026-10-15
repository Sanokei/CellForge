# SPDX-License-Identifier: MIT-0
"""Frozen adjacency gene DEV measurement; no public evaluation or optimizer."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from cellforge.governance.ledger import Ledger

def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--private',required=True);p.add_argument('--out',required=True);p.add_argument('--family',required=True);a=p.parse_args()
 out=Path(a.out)
 if out.exists():raise ValueError('fresh output required')
 g=Graph.load(a.graph);predict=GridPredictor(g,backend='compiled')
 directory=Path(a.private)/'arcagi2/data/training'
 paths=[p for k,p in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in directory.glob('*.json')) if int(k,16)%5!=0][:200]
 if len(paths)!=200:raise ValueError('DEV200 missing')
 native=[];manifest=hashlib.sha256()
 for path in paths:
  raw=path.read_bytes();manifest.update(path.stem.encode()+b':'+hashlib.sha256(raw).digest());native.append((path.stem,json.loads(raw)))
 raw_family=Path(a.family).read_bytes();family=json.loads(raw_family)['tasks']
 if len(family)!=300:raise ValueError('expected300original tasks')
 synthetic=[(str(i),task) for i,task in enumerate(family)]
 policy={'protocol':'adjacency-frozen-evidence/1','graph_hash':g.hash(),'stats':g.stats(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'native_manifest_sha256':manifest.hexdigest(),'family_sha256':hashlib.sha256(raw_family).hexdigest(),'synthetic_tasks':300,'synthetic_partition':'first200selection,last100validation;no selection or edits','audit_evaluations':0,'final_evaluations':0,'public_evaluation_loaded':False,'model_edits':0,'gradients':False,'limitations':g.meta['limitations']}
 out.mkdir();ledger=Ledger(out/'ledger.jsonl')
 def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
 write('policy.json',policy);ledger.append('policy_frozen',policy,'operator');rows=[]
 # Separate end-to-end adapter throughput calibration at30x30, including3demos.
 grid=np.zeros((30,30),dtype=int);grid[2:10,2:10]=2;grid[20:23,20:23]=3;pair30={'input':grid.tolist(),'output':np.full((8,8),2).tolist()}
 start=time.perf_counter();cpu=time.process_time()
 for _ in range(20):predict([pair30]*3,pair30['input'])
 calibration={'tasks':20,'grid_shape':[30,30],'demos_per_task':3,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu}
 calibration['tasks_per_second']=20/calibration['wall_seconds'];write('calibration.json',calibration);ledger.append('calibration',calibration,'operator')
 for split,tasks in [('original-synthetic-selection',synthetic[:200]),('original-synthetic-validation',synthetic[200:]),('native-development',native)]:
  row=score_with_consistency(tasks,predict);row.update(split=split,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);rows.append(row);write('progress.json',rows);ledger.append('graph_scored',row,'operator')
 result={'policy':policy,'calibration':calibration,'rows':rows};ledger.append('complete',result,'operator');result['ledger_verified']=ledger.verify();write('result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
