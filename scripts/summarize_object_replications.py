# SPDX-License-Identifier: MIT-0
"""Validate all predeclared search-seed receipts, preserving negative outcomes."""
import argparse,json,statistics
from pathlib import Path
from cellforge.governance.ledger import Ledger
from cellforge.ir.graph import Graph

SEEDS=(772041,772051,772061)
ARMS=('crossover','mutation_only','random_search')


def summarize(root,*,prefix='grid-object-replication',expected_age=3):
    runs=[];commitments=None
    for seed in SEEDS:
        p=Path(root)/f'{prefix}-{seed}'
        r=json.loads((p/'result.json').read_text())
        policy=r['policy']
        if policy['seed_base']!=seed or policy['native_query_scoring']:raise ValueError('replication policy differs')
        if policy['max_age']!=expected_age:raise ValueError('population age differs')
        current=tuple(policy[k] for k in ('selection_manifest','validation_full_io_manifest','shared_base_hash'))
        if commitments is None:commitments=current
        if current!=commitments:raise ValueError('replication corpus/bank differs')
        if not r['equal_physical_calls'] or r['physical_selection_evaluations']!=18:raise ValueError('unmatched physical budget')
        ledger_count=Ledger(p/'ledger.jsonl').verify()
        rows={}
        for arm in ARMS:
            cost=r['costs'][arm]
            if cost['physical_evaluations']!=6 or cost['predictor_calls']!=348:raise ValueError('arm budget differs')
            matching=[x for x in r['rows'] if x['arm']==arm]
            if len(matching)!=1 or matching[0]['split']!='fresh-synthetic-validation':raise ValueError('unexpected report scope')
            row=matching[0]
            graph=Graph.load(p/(arm+'-winner'))
            if graph.hash()!=r['winners'][arm]:raise ValueError('winner artifact changed')
            if row['tasks']!=20 or sum(x['solved'] for x in row['task_records'])!=row['tasks_solved']:raise ValueError('score disagreement')
            rows[arm]=dict(solved=row['tasks_solved'],cpu_seconds=cost['evaluation_cpu_seconds'],
                           evaluations_per_second=6/cost['evaluation_wall_seconds'],
                           failures=cost['failed_evaluations'],winner_hash=graph.hash(),
                           alleles=graph.meta['two_slot_genome']['alleles'],report_reused=row['report_reused'])
        runs.append(dict(seed=seed,arms=rows,ledger_verified=ledger_count,peak_rss_bytes=r['peak_rss_bytes']))
    aggregate={}
    for arm in ARMS:
        values=[r['arms'][arm] for r in runs]
        aggregate[arm]=dict(mean_solved=statistics.mean(x['solved'] for x in values),
                            min_solved=min(x['solved'] for x in values),max_solved=max(x['solved'] for x in values),
                            selection_cpu_seconds=sum(x['cpu_seconds'] for x in values),
                            physical_evaluations=18,predictor_calls=1044,
                            evaluation_failures=sum(x['failures'] for x in values))
    return dict(protocol='object-bank-search-seed-summary/1',max_age=expected_age,runs=runs,aggregate=aggregate,
                scope='three random-search-seed replications on the SAME exposed20 synthetic tasks; not60 unique tasks',
                benchmark_query_scoring=False,final_evaluations=0,gradients=False,
                limitations='small fixed-curriculum study; equal evaluations, unequal CPU; not native or independent transfer')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--prefix',choices=('grid-object-replication','grid-object-age8'),default='grid-object-replication')
    p.add_argument('--max-age',type=int,choices=(3,8),default=3);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('fresh summary required')
    r=summarize(a.root,prefix=a.prefix,expected_age=a.max_age);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['aggregate']))


if __name__=='__main__':main()
