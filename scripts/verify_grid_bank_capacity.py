# SPDX-License-Identifier: MIT-0
"""Verify finite-bank diagnostic receipts; query-oracle bounds stay explicit."""
import argparse,json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from run_grid_population_matched import canonical_genotype
from build_grid_two_slot_genome import instantiate_genome


def verify(root,base):
    r=json.loads((root/'result.json').read_text());ledger=Ledger(root/'ledger.jsonl');entries=ledger.entries()
    assert ledger.verify()==101
    assert entries[0]['payload']==r['policy']
    assert entries[-1]['payload']=={k:v for k,v in r.items() if k!='ledger_verified'}
    p=r['policy'];assert base.hash()==p['base_hash']
    assert not p['gradients'] and not p['public_evaluation_loaded']
    assert p['native_tasks_loaded']==p['audit_evaluations']==p['final_evaluations']==0
    assert len(r['selection'])==len(r['records'])==49
    assert [e['payload'] for e in entries if e['type']=='physical_selection_evaluation']==r['selection']
    assert [e['payload'] for e in entries if e['type']=='diagnostic_report_scored']==r['records']
    eligible=[x for x in r['selection'] if not x['fitness']['failed']]
    winner=min(eligible,key=lambda x:(-x['fitness']['solved'],x['genotype']))
    assert r['winner']==winner
    assert Graph.load(root/'selected-winner').hash()==winner['graph_hash']
    union=set();family_union={f:set() for f in r['families']};checks=[]
    for selection,row in zip(r['selection'],r['records']):
        i=row['index'];assert i==selection['index']
        g=instantiate_genome(base,row['alleles'])
        assert g.hash()==row['graph_hash']==selection['graph_hash']==p['graph_hashes'][i]
        assert canonical_genotype(base,g)==selection['genotype']
        fitness=selection['fitness'];assert fitness['solved']==sum(fitness['solves'])
        assert fitness['predictor_calls']==70
        s=row['score'];records=s['task_records']
        assert len(records)==s['tasks']==24 and len({x['task_id'] for x in records})==24
        assert sum(x['solved'] for x in records)==s['tasks_solved']
        assert sum(x['demo_consistent'] for x in records)==s['demo_consistent_tasks']
        assert sum(x['demo_consistent_but_wrong'] for x in records)==s['demo_consistent_but_wrong_tasks']
        for j,x in enumerate(records):
            assert x['solved']==all(x['query_successes'])
            assert x['demo_consistent_but_wrong']==(x['demo_consistent'] and not x['solved'])
            if x['solved']:union.add(x['task_id']);family_union[r['families'][j]].add(x['task_id'])
        checks.append(dict(alleles=row['alleles'],selection=fitness['solved'],queries=s['tasks_solved'],
                          cpu_seconds_per_task=s['inference_cpu_seconds_per_task']))
    return dict(ledger_verified=101,physical_selection_evaluations=49,predictor_calls=3430,
                selected=checks[winner['index']],posthoc_best_single_queries=max(x['queries'] for x in checks),
                posthoc_union_tasks=len(union),posthoc_union_by_family={f:len(x) for f,x in family_union.items()},
                peak_rss_bytes=r['peak_rss_bytes'],all_layouts=checks,
                scope='fresh original synthetic development diagnosis; posthoc best/union are oracle bounds, not model scores or independent frontier evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    print(json.dumps(verify(a.run,Graph.load(a.base)),indent=2))
