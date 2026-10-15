# SPDX-License-Identifier: MIT-0
"""Frozen public-training-only diagnostics; no query targets enter diagnostic API."""
import argparse,hashlib,json,time,resource
from pathlib import Path
from collections import Counter
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_train_diagnostics import diagnose_train
from cellforge.governance.ledger import Ledger
EXPECTED='4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4'
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',nargs=2,action='append',required=True);p.add_argument('--private',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
 if out.exists():raise ValueError('fresh diagnostics required')
 graphs=[(name,Graph.load(path)) for name,path in a.model];directory=Path(a.private)/'arcagi2/data/training'
 paths=[p for k,p in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in directory.glob('*.json')) if int(k,16)%5!=0][:200]
 if len(paths)!=200:raise ValueError('DEV200missing')
 manifest=hashlib.sha256();tasks=[]
 for path in paths:
  raw=path.read_bytes();manifest.update(path.stem.encode()+b':'+hashlib.sha256(raw).digest());parsed=json.loads(raw)
  sanitized={'train':[{'input':d['input'],'output':d['output']} for d in parsed['train']]};tasks.append((path.stem,sanitized));del parsed
 if manifest.hexdigest()!=EXPECTED:raise ValueError('DEVfilecontentchanged')
 out.mkdir();ledger=Ledger(out/'ledger.jsonl');policy={'protocol':'train-only-grid-diagnostics/1','native_manifest':manifest.hexdigest(),'model_hashes':{n:g.hash() for n,g in graphs},'tasks':200,'evidence':'train-pairs','query_labels_used':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'public_evaluation_loaded':False,'audit_evaluations':0,'final_evaluations':0,'selection_or_graph_edits':False,'gradients':False}
 def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
 write('policy.json',policy);ledger.append('policy_frozen',policy,'operator');summaries={};start=time.perf_counter();cpu=time.process_time()
 for name,g in graphs:
  predictor=GridPredictor(g,backend='compiled');rows=[]
  for identity,task in tasks:rows.append(dict(diagnose_train(task,predictor),task_id=identity))
  write(name+'-diagnostics.json',rows);classes=Counter(r['failure_class'] for r in rows)
  summaries[name]={'classes':dict(classes),'tasks':len(rows),'exact_demonstration_consistent':sum(r['demonstration_consistent'] for r in rows),'shape_changing_demo_tasks':sum(any(len(d['input'])!=len(d['output']) or len(d['input'][0])!=len(d['output'][0]) for d in task['train']) for _,task in tasks),'diagnostic_sha256':hashlib.sha256((out/(name+'-diagnostics.json')).read_bytes()).hexdigest()};ledger.append('train_diagnostics_completed',summaries[name],'operator')
 result={'policy':policy,'summaries':summaries,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'limitations':'development diagnostics, not query/generalization score; observed failures may reflect routing rather than absent gene capability'}
 ledger.append('completed',result,'operator');result['ledger_verified']=ledger.verify();write('result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
