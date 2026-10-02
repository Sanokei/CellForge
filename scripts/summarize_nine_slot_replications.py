# SPDX-License-Identifier: MIT-0
"""All predeclared cached nine-slot replications; no native task access."""
import argparse,json,hashlib
from pathlib import Path
from cellforge.ir.graph import Graph
from verify_scaling_population_receipt import verify
from run_grid_population_matched import frozen_expanded_corpora,digest


def summarize(plan_path,base_path):
    plan=json.loads(plan_path.read_text());base=Graph.load(base_path)
    archive=Path(plan['source_archive'])
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==plan['source_sha256']
    manifest=json.loads(archive.with_suffix('.manifest.json').read_text())
    source_hash=next(x['sha256'] for x in manifest['files'] if x['path']=='scripts/run_grid_population_matched.py')
    runs=[];seen=set()
    assert len(plan['runs'])==3
    for spec in plan['runs']:
        root=plan_path.parent/spec['run'];r=json.loads((root/'result.json').read_text());p=r['policy']
        assert p['source_sha256']==source_hash and p['demo_state_cache']
        assert p['rank_consistent_selection'] and p['marker_family'] and p['numeric_family']
        assert p['selection_seed_base']==spec['selection_seed'] and p['validation_seed_base']==spec['report_seed']
        _,report,_,commitments=frozen_expanded_corpora(plan['selection_per_family'],
            spec['selection_seed'],spec['report_seed'],plan['report_per_family'],True,True)
        assert all(spec[key]==value and p[key]==value for key,value in commitments.items())
        fingerprints={digest(t) for t in report};assert len(fingerprints)==70 and not seen&fingerprints;seen|=fingerprints
        checked=verify(root,base,spec['search_seed'],plan['proposals_per_arm'],True,plan['report_per_family'])
        assert checked['ledger_verified']==108
        for arm,cost in r['costs'].items():
            assert cost['predictor_calls']==plan['predictor_calls_per_arm_per_run']
            assert cost['physical_evaluations']==plan['physical_evaluations_per_arm_per_run']
            assert cost['runtime_steps']==2808 and cost['demo_state_cache_hits']==900
            row=next(x for x in r['rows'] if x['arm']==arm)
            checked['arms'][arm]['wrong_consistent']=row['demo_consistent_but_wrong_tasks']
            checked['arms'][arm]['predictor_calls']=cost['predictor_calls']
        runs.append(dict(spec=spec,**checked))
    aggregate={arm:dict(solved=sum(r['arms'][arm]['solved'] for r in runs),tasks=210,
        per_run=[r['arms'][arm]['solved'] for r in runs],
        wrong_consistent=sum(r['arms'][arm]['wrong_consistent'] for r in runs),
        physical_evaluations=54,predictor_calls=4212,
        failed_evaluations=sum(r['arms'][arm]['failed_evaluations'] for r in runs))
        for arm in ('crossover','mutation_only','random_search')}
    return dict(runs=runs,aggregate=aggregate,unique_report_tasks=len(seen),
        scope='three predeclared authored synthetic replications; no independent ARC/frontier claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh summary required')
    r=summarize(a.plan,a.base);a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r['aggregate']))
