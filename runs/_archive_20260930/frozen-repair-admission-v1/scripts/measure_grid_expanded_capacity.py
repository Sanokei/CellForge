# SPDX-License-Identifier: MIT-0
"""Frozen active-slot ablation on original non-overflow synthetic examples."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.governance.ledger import Ledger
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from cellforge.reference.grid_curriculum import FAMILIES,generate_task
from build_grid_adjacency_recolor_gene import generate_adjacency_family
from build_grid_object_association_child import family as object_family


def family(seed=3037000,per_family=10):
    tasks=[generate_task(f,seed+j*10000+i) for j,f in enumerate(FAMILIES) for i in range(per_family)]
    tasks+=generate_adjacency_family(per_family,seed)['tasks']
    for t in object_family(seed+100000)['tasks'][:per_family]:t['family']='object_recolor';tasks.append(t)
    for f in (2,3):
        rng=np.random.default_rng(seed+200000+f)
        for i in range(per_family):
            pairs=[]
            for j in range(4):
                h,w=map(int,rng.integers(2,9,size=2));x=rng.integers(0,10,size=(h,w))
                pairs.append(dict(input=x.tolist(),output=np.repeat(np.repeat(x,f,axis=0),f,axis=1).tolist()))
            tasks.append(dict(family=f'repeat{f}',train=pairs[:3],test=pairs[3:]))
    return tasks


def main():
    import resource
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    inventory=json.loads(a.inventory.read_text());graphs={}
    for row in inventory['artifacts']:
        g=Graph.load(a.inventory.parent/row['name']);assert g.hash()==row['hash']
        check_profile(g,'ladder-micro/1');graphs[row['name']]=g
    left,right=graphs.values()
    assert left.params.keys()==right.params.keys() and all(np.array_equal(left.params[k],right.params[k]) for k in left.params)
    assert left.meta['state_dim']==right.meta['state_dim']==1515
    tasks=family();families=[t.get('family',t.get('metadata',{}).get('family')) for t in tasks]
    fingerprints=[hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest() for t in tasks];assert len(set(fingerprints))==120
    policy=dict(protocol='expanded-active-slot-ablation/1',graph_hashes={k:g.hash() for k,g in graphs.items()},seed=3037000,
                family_sha256=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),families=families,
                scope='fixed authored active-slot utility; no evolutionary discovery or independent frontier evaluation',
                scaling_overflow_cases=0,gradients=False,native_tasks_loaded=0,public_evaluation_loaded=False,audit_evaluations=0,final_evaluations=0)
    a.out.mkdir();ledger=Ledger(a.out/'ledger.jsonl');ledger.append('policy_frozen',policy,'operator');rows=[]
    for name,g in graphs.items():
        row=dict(score_with_consistency([(str(i),t) for i,t in enumerate(tasks)],GridPredictor(g,backend='compiled'),record_tasks=True),arm=name,stats=g.stats())
        rows.append(row);ledger.append('graph_scored',row,'operator')
    result=dict(policy=policy,rows=rows,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    ledger.append('completed',result,'operator');result['ledger_verified']=ledger.verify()
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=[{k:v for k,v in r.items() if k not in ('task_records','stats')} for r in rows],ledger_verified=result['ledger_verified'])))


if __name__=='__main__':main()
