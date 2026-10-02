# SPDX-License-Identifier: MIT-0
"""Validate selector receipts and compute paired outcomes; no task-file reads."""
import argparse,json
from pathlib import Path
from cellforge.governance.ledger import Ledger
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile


def verify(root,genes,version):
    r=json.loads((root/'result.json').read_text());ledger=Ledger(root/'ledger.jsonl')
    assert ledger.verify()==7
    entries=ledger.entries();assert entries[0]['payload']==r['policy']
    assert [e['payload'] for e in entries[1:]]==r['rows']
    summary=[];native={};profiles={}
    for arm,suffix in [('adapt',''),('control','-no-update')]:
        g=Graph.load(genes/f'scaling-selector{suffix}-v{version}')
        assert g.hash()==r['policy']['graph_hashes'][arm]
        try:profiles[arm]=check_profile(g,'ladder-micro/1')
        except ValueError as e:
            if version!=1:raise
            profiles[arm]={'failure':str(e)}
        rows=[x for x in r['rows'] if x['arm']==arm]
        assert len(rows)==3
        for row,n in zip(rows,(200,100,789)):
            records=row['task_records'];assert len(records)==row['tasks']==n
            assert len({x['task_id'] for x in records})==n
            assert row['stats']==g.stats()
            for x in records:
                assert x['solved']==all(x['query_successes'])
                assert x['demo_consistent_but_wrong']==(x['demo_consistent'] and not x['solved'])
            for key,field in [('tasks_solved','solved'),('demo_consistent_tasks','demo_consistent'),('demo_consistent_but_wrong_tasks','demo_consistent_but_wrong')]:
                assert sum(x[field] for x in records)==row[key]
            assert sum(sum(x['query_successes']) for x in records)==row['test_inputs_correct']
            summary.append({k:row[k] for k in ['arm','split','tasks_solved','demo_consistent_tasks','demo_consistent_but_wrong_tasks','tasks_per_second','inference_cpu_seconds_per_task']})
            if row['split']=='expanded-native-development':native[arm]={x['task_id']:x['solved'] for x in records}
    assert native['adapt'].keys()==native['control'].keys()
    p=r['policy'];assert not p['gradients'] and not p['public_evaluation_loaded']
    assert p['audit_evaluations']==p['final_evaluations']==0
    return dict(rows=summary,profiles=profiles,ledger_verified=7,peak_rss_bytes=r['peak_rss_bytes'],
                gained=[i for i in native['adapt'] if native['adapt'][i] and not native['control'][i]],
                lost=[i for i in native['adapt'] if native['control'][i] and not native['adapt'][i]],
                scope='receipt consistency and paired outcomes, not independent score replication')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--genes',type=Path,required=True);p.add_argument('--version',type=int,choices=[1,2],required=True)
    a=p.parse_args();print(json.dumps(verify(a.run,a.genes,a.version),indent=2))
