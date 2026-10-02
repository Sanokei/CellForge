# SPDX-License-Identifier: MIT-0
"""All predeclared search/corpus replications, checked before aggregation."""
import argparse,hashlib,json
from pathlib import Path
from cellforge.ir.graph import Graph
from verify_scaling_population_receipt import verify
from run_grid_population_matched import frozen_expanded_corpora,digest


def summarize(root,plan_path,base_path):
    plan=json.loads(plan_path.read_text());base=Graph.load(base_path);runs=[];seen=set()
    manifest=json.loads((plan_path.parent/'grid-expanded-replications-v1.tar.manifest.json').read_text())
    assert manifest['archive_sha256']==plan['source_sha256']
    assert hashlib.sha256((plan_path.parent/'grid-expanded-replications-v1.tar.gz').read_bytes()).hexdigest()==plan['source_sha256']
    source_hash=next(x['sha256'] for x in manifest['files'] if x['path']=='scripts/run_grid_population_matched.py')
    for spec in plan['runs']:
        seed=spec['search_seed'];path=root/f'grid-expanded-replication-{seed}'
        result=json.loads((path/'result.json').read_text());p=result['policy']
        assert p['expanded_bank'] and p['selection_seed_base']==spec['selection_seed']
        assert p['source_sha256']==source_hash and p['shared_base_hash']==base.hash()
        assert p['validation_seed_base']==spec['validation_seed']
        selection,report,_,commitments=frozen_expanded_corpora(plan['selection_per_family'],spec['selection_seed'],spec['validation_seed'],plan['report_per_family'])
        assert all(p[k]==v for k,v in commitments.items())
        # Reconstruct original examples to check cross-run query/report isolation.
        fingerprints={digest({'train':t['train'],'test':t['test']}) for t in report}
        assert len(fingerprints)==120 and not seen&fingerprints;seen.update(fingerprints)
        check=verify(path,base,seed,plan['proposals_per_arm'],True,plan['report_per_family'])
        assert check['ledger_verified']==108
        runs.append(dict(spec=spec,**check))
    assert len(runs)==3
    arms=('crossover','mutation_only','random_search')
    aggregate={arm:dict(solved=sum(r['arms'][arm]['solved'] for r in runs),tasks=360,
                        per_run=[r['arms'][arm]['solved'] for r in runs],physical_evaluations=54,predictor_calls=3780,
                        selection_cpu_seconds=sum(r['arms'][arm]['selection_cpu_seconds'] for r in runs),
                        failures=sum(r['arms'][arm]['failed_evaluations'] for r in runs)) for arm in arms}
    return dict(runs=runs,aggregate=aggregate,unique_report_tasks=len(seen),
                scope='three predeclared synthetic search/corpus replications; verified receipts, not independent ARC/frontier evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh summary required')
    r=summarize(a.root,a.plan,a.base);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['aggregate']))
