# SPDX-License-Identifier: MIT-0
"""Frozen authored scaling evidence; expanded public training, no audit/final."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from cellforge.governance.ledger import Ledger


def generate_family(factor,seed):
    rng=np.random.default_rng(seed);tasks=[]
    for i in range(300):
        pairs=[]
        for j in range(4):
            h,w=map(int,rng.integers(1,30//factor+1,size=2))
            if i%10==0 and j==3:h=30//factor+1
            x=rng.integers(0,10,size=(h,w))
            y=np.repeat(np.repeat(x,factor,axis=0),factor,axis=1) if max(h,w)*factor<=30 else x
            pairs.append(dict(input=x.tolist(),output=y.tolist()))
        tasks.append(dict(train=pairs[:3],test=pairs[3:]))
    return tasks


def main():
    import resource
    p=argparse.ArgumentParser();p.add_argument('--graphs',required=True);p.add_argument('--private',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('fresh output required')
    graphs={f:Graph.load(Path(a.graphs)/f'repeat{f}-v1') for f in (2,3)}
    d=Path(a.private)/'arcagi2/data/training'
    paths=[p for h,p in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in d.glob('*.json')) if int(h,16)%5!=0]
    manifest=hashlib.sha256();native=[]
    for path in paths:
        raw=path.read_bytes();manifest.update(path.stem.encode()+b':'+hashlib.sha256(raw).digest());native.append((path.stem,json.loads(raw)))
    if manifest.hexdigest()!='9b82cdec59033c4fe67ebd2f8b93ea2a495b90ec9bff5a0e4a5c851cd4291af9':raise ValueError('expanded dev changed')
    families={f:generate_family(f,2947000+f) for f in graphs}
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    for tasks in families.values():
        fingerprints=[digest(t) for t in tasks]
        if len(set(fingerprints))!=300:raise ValueError('duplicate synthetic I/O')
    out.mkdir();ledger=Ledger(out/'ledger.jsonl')
    policy=dict(protocol='repeat-genes-frozen-evidence/1',graph_hashes={f:g.hash() for f,g in graphs.items()},
                native_manifest_sha256=manifest.hexdigest(),native_tasks=len(native),
                synthetic_hashes={f:digest(t) for f,t in families.items()},synthetic_partition='first200 development, last100 synthetic validation',
                scope='expanded exposed training partition; scaling examples already identified from demos, not independent final validation',
                gradients=False,audit_evaluations=0,final_evaluations=0,public_evaluation_loaded=False)
    ledger.append('policy_frozen',policy,'operator');rows=[]
    for f,g in graphs.items():
        predict=GridPredictor(g,backend='compiled');synthetic=[(str(i),t) for i,t in enumerate(families[f])]
        for split,tasks in [('synthetic-development',synthetic[:200]),('synthetic-validation',synthetic[200:]),('expanded-native-development',native)]:
            row=dict(score_with_consistency(tasks,predict,record_tasks=True),factor=f,split=split,stats=g.stats())
            rows.append(row);ledger.append('graph_scored',row,'operator')
    r=dict(policy=policy,rows=rows,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,ledger_verified=ledger.verify())
    (out/'result.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(rows=[{k:v for k,v in r.items() if k not in ('task_records','stats')} for r in rows],ledger_verified=r['ledger_verified'])))


if __name__=='__main__':main()
