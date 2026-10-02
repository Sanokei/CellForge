# SPDX-License-Identifier: MIT-0
"""Verify identical search trajectory and outcomes across cache-only replay."""
import argparse,json
from pathlib import Path
from cellforge.governance.ledger import Ledger


def compare(plain,cached):
    results=[json.loads((root/'result.json').read_text()) for root in (plain,cached)]
    records=[]
    for root,result in zip((plain,cached),results):
        ledger=Ledger(root/'ledger.jsonl');assert ledger.verify()==36
        assert ledger.find('completed')[0]['payload']=={k:v for k,v in result.items() if k!='ledger_verified'}
        records.append([x['payload'] for x in ledger.entries() if x['type']=='physical_selection_evaluation'])
    ignored={'source_sha256','selection_cache','demo_state_cache'}
    assert [{k:v for k,v in r['policy'].items() if k not in ignored} for r in results][0]=={k:v for k,v in results[1]['policy'].items() if k not in ignored}
    assert not results[0]['policy'].get('demo_state_cache',False)
    assert results[1]['policy']['demo_state_cache']
    assert len(records[0])==len(records[1])==18
    for a,b in zip(*records):
        for key in ('arm','kind','artifact_hash','canonical_genotype','genome'):
            assert a[key]==b[key]
        for key in ('solves','solved','tasks','failed','errors','predictor_calls'):
            assert a['fitness'][key]==b['fitness'][key]
    assert results[0]['winners']==results[1]['winners']
    for a,b in zip(results[0]['rows'],results[1]['rows']):
        for key in ('arm','artifact_hash','task_records','tasks_solved','demo_consistent_but_wrong_tasks'):
            assert a[key]==b[key]
    arms={}
    for arm in results[0]['costs']:
        a,b=[r['costs'][arm] for r in results]
        assert a['physical_evaluations']==b['physical_evaluations']==6
        assert a['predictor_calls']==b['predictor_calls']==234
        assert b['runtime_steps']==468 and b['demo_state_cache_hits']==150
        arms[arm]=dict(selection_wall_speedup=a['evaluation_wall_seconds']/b['evaluation_wall_seconds'],
            cached_evaluations_per_second=6/b['evaluation_wall_seconds'])
    return dict(identical_trajectories=True,physical_evaluations=18,arms=arms,
        aggregate_selection_wall_speedup=sum(x['evaluation_wall_seconds'] for x in results[0]['costs'].values())/sum(x['evaluation_wall_seconds'] for x in results[1]['costs'].values()),
        scope='same-seed execution replay, not independent efficacy replication')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plain',type=Path,required=True);p.add_argument('--cached',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh comparison receipt required')
    r=compare(a.plain,a.cached);a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
