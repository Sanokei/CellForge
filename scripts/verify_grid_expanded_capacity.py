# SPDX-License-Identifier: MIT-0
"""Check active-slot ablation and paired family outcomes; no task data reads."""
import argparse,json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.governance.ledger import Ledger


def verify(root,inventory):
    r=json.loads((root/'result.json').read_text());ledger=Ledger(root/'ledger.jsonl');entries=ledger.entries()
    assert ledger.verify()==4 and entries[0]['payload']==r['policy']
    assert entries[-1]['payload']=={k:v for k,v in r.items() if k!='ledger_verified'}
    assert [e['payload'] for e in entries if e['type']=='graph_scored']==r['rows']
    p=r['policy'];assert not p['gradients'] and not p['public_evaluation_loaded']
    assert p['native_tasks_loaded']==p['audit_evaluations']==p['final_evaluations']==p['scaling_overflow_cases']==0
    inv=json.loads(inventory.read_text());graphs={}
    summary={};outcomes={}
    for row in inv['artifacts']:
        name=row['name'];g=Graph.load(inventory.parent/name);graphs[name]=g
        assert g.hash()==row['hash']==p['graph_hashes'][name]
        check_profile(g,'ladder-micro/1')
        s=next(x for x in r['rows'] if x['arm']==name);assert s['stats']==g.stats()
        records=s['task_records'];assert len(records)==s['tasks']==120
        assert [x['task_id'] for x in records]==list(map(str,range(120)))
        for x in records:
            assert x['solved']==all(x['query_successes'])
            assert x['demo_consistent_but_wrong']==(x['demo_consistent'] and not x['solved'])
        assert sum(x['solved'] for x in records)==s['tasks_solved']
        assert sum(x['demo_consistent'] for x in records)==s['demo_consistent_tasks']
        assert sum(x['demo_consistent_but_wrong'] for x in records)==s['demo_consistent_but_wrong_tasks']
        family_scores={f:sum(x['solved'] for x,family in zip(records,p['families']) if family==f) for f in set(p['families'])}
        summary[name]=dict(solved=s['tasks_solved'],families=family_scores,tasks_per_second=s['tasks_per_second'],cpu_seconds_per_task=s['inference_cpu_seconds_per_task'],
                           demo_consistent=s['demo_consistent_tasks'],consistent_wrong=s['demo_consistent_but_wrong_tasks'])
        outcomes[name]={x['task_id']:x['solved'] for x in records}
    left,right=graphs.values();assert left.params.keys()==right.params.keys()
    assert all(np.array_equal(left.params[k],right.params[k]) for k in left.params)
    return dict(arms=summary,paired_gains=[i for i in outcomes['expanded'] if outcomes['expanded'][i] and not outcomes['two-active-control'][i]],
                paired_losses=[i for i in outcomes['expanded'] if not outcomes['expanded'][i] and outcomes['two-active-control'][i]],
                ledger_verified=4,peak_rss_bytes=r['peak_rss_bytes'],scope='authored active-gene ablation, not evolutionary or independent frontier evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);a=p.parse_args()
    print(json.dumps(verify(a.run,a.inventory),indent=2))
