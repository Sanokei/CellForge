# SPDX-License-Identifier: MIT-0
"""Check frozen scaling evidence without accessing native task data."""
import argparse, hashlib, json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger


def verify(root, genes):
    result=json.loads((root/'result.json').read_text())
    ledger=Ledger(root/'ledger.jsonl')
    assert ledger.verify()==7
    entries=ledger.entries()
    assert entries[0]['payload']==result['policy']
    assert [e['payload'] for e in entries[1:]]==result['rows']
    p=result['policy']
    assert p['native_tasks']==789
    assert p['native_manifest_sha256']=='9b82cdec59033c4fe67ebd2f8b93ea2a495b90ec9bff5a0e4a5c851cd4291af9'
    assert not p['gradients'] and not p['public_evaluation_loaded']
    assert p['audit_evaluations']==p['final_evaluations']==0
    assert len(result['rows'])==6
    summary=[];native_sets=[]
    for factor in (2,3):
        g=Graph.load(genes/f'repeat{factor}-v1')
        assert g.hash()==p['graph_hashes'][str(factor)]
        rows=[r for r in result['rows'] if r['factor']==factor]
        assert [r['split'] for r in rows]==['synthetic-development','synthetic-validation','expanded-native-development']
        for r,expected in zip(rows,(200,100,789)):
            records=r['task_records'];ids=[x['task_id'] for x in records]
            assert len(records)==len(set(ids))==r['tasks']==expected
            assert r['stats']==g.stats()
            for x in records:
                assert x['solved']==all(x['query_successes'])
                assert x['demo_consistent_but_wrong']==(x['demo_consistent'] and not x['solved'])
            assert sum(x['solved'] for x in records)==r['tasks_solved']
            assert sum(x['demo_consistent'] for x in records)==r['demo_consistent_tasks']
            assert sum(x['demo_consistent_but_wrong'] for x in records)==r['demo_consistent_but_wrong_tasks']
            assert sum(len(x['query_successes']) for x in records)==r['test_inputs']
            assert sum(sum(x['query_successes']) for x in records)==r['test_inputs_correct']
            if r['split']=='expanded-native-development':
                assert all(int(hashlib.sha256(i.encode()).hexdigest(),16)%5 for i in ids)
                native_sets.append(ids)
                summary.append(dict(factor=factor,solved_ids=[x['task_id'] for x in records if x['solved']],
                                    tasks_per_second=r['tasks_per_second'],cpu_seconds_per_task=r['inference_cpu_seconds_per_task'],
                                    params=g.stats()['unique_param_storage']))
    assert native_sets[0]==native_sets[1]
    union=set().union(*(set(r['solved_ids']) for r in summary))
    return dict(ledger_verified=7,rows_verified=6,native_union=len(union),genes=summary,
                peak_rss_bytes=result['peak_rss_bytes'],verification_scope='receipt integrity and frozen artifacts; not independent score replication')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--genes',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(verify(a.run,a.genes),indent=2))
