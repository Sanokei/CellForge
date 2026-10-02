# SPDX-License-Identifier: MIT-0
"""Bounded public-training-only graph failures for offline makers."""
import argparse,json,hashlib,time,resource
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.construction.maker_diagnostics import train_counterexamples

def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('fresh diagnostics required')
 g=Graph.load(a.graph);paths=[x for h,x in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in (a.private/'arcagi2/data/training').glob('*.json')) if int(h,16)%5!=0][:200]
 if len(paths)!=200:raise ValueError('complete fixed development partition required')
 # Only train fields leave the loader; public evaluation/audit paths excluded.
 tasks=[dict(task_id=x.stem,train=json.loads(x.read_text())['train']) for x in paths[:100]]
 policy=dict(protocol='arc-development-maker-diagnostics/1',graph_hash=g.hash(),partition='fixed hash-ordered DEV200, first100 train-only tasks',candidate_task_ids=[x.stem for x in paths[:100]],train_manifest=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),counterexample_limit=5,per_task_limit=1,gradients=False,audit_evaluations=0,final_evaluations=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 a.out.mkdir();(a.out/'policy.json').write_text(json.dumps(policy,indent=2));start=time.perf_counter();d=train_counterexamples(g,tasks,limit=5,per_task_limit=1);d['policy']=policy;d['wall_seconds']=time.perf_counter()-start;d['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 selected={x['task_id'] for x in d['counterexamples']};(a.out/'train_task_views.json').write_text(json.dumps([t for t in tasks if t['task_id'] in selected],indent=2))
 (a.out/'diagnostics.json').write_text(json.dumps(d,indent=2));print(json.dumps(dict(examples=len(d['counterexamples']),calls=d['diagnostic_predictor_calls'],wall_seconds=d['wall_seconds'],tasks=[x['task_id'] for x in d['counterexamples']])))
if __name__=='__main__':main()
