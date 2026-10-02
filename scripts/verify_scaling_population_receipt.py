# SPDX-License-Identifier: MIT-0
"""Check a frozen matched scaling pilot, without loading native task files."""
import argparse,json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from run_grid_population_matched import canonical_genotype,proposal_seeds


def verify(root,base,seed=772081,proposals=4,independent=False,report_per_family=2):
    r=json.loads((root/'result.json').read_text());p=r['policy']
    ledger=Ledger(root/'ledger.jsonl');count=ledger.verify();entries=ledger.entries()
    assert entries[0]['payload']==p and entries[-1]['type']=='completed'
    assert entries[-1]['payload']=={k:v for k,v in r.items() if k!='ledger_verified'}
    assert p['scaling_bank'] and p['object_bank'] and not p['native_query_scoring']
    if p.get('expanded_bank'):
        assert p['native_tasks_loaded']==p['scaling_overflow_cases']==0
        assert p['native_content_manifest'] is None
    assert p['seed_base']==seed and p['max_age']==8
    assert p['proposals_per_arm']==proposals
    if independent:assert p['random_streams'].startswith('sha256(')
    marker=p.get('marker_family',False)
    numeric=p.get('numeric_family',False)
    if marker or numeric:
        from run_grid_population_matched import frozen_expanded_corpora
        _,_,_,commitments=frozen_expanded_corpora(p['per_family'],p['selection_seed_base'],
            p['validation_seed_base'],report_per_family,include_marker=marker,include_numeric=numeric)
        assert all(p[key]==value for key,value in commitments.items())
    else:
        assert p['selection_tasks']==24 and p['selection_demo_pairs']==70
    assert p['audit_evaluations']==p['final_evaluations']==p['gradient_steps']==p['gradient_estimators']==0
    assert base.hash()==p['shared_base_hash']
    evaluations=proposals+2;calls=evaluations*p['selection_demo_pairs']
    assert r['physical_selection_evaluations']==3*evaluations and r['equal_physical_calls']
    physical=[e['payload'] for e in entries if e['type']=='physical_selection_evaluation']
    assert len(physical)==3*evaluations
    summary={}
    for arm in ('crossover','mutation_only','random_search'):
        cost=r['costs'][arm];records=[x for x in physical if x['arm']==arm]
        assert len(records)==cost['physical_evaluations']==evaluations
        assert cost['predictor_calls']==sum(x['fitness']['predictor_calls'] for x in records)==calls
        for x in records:assert x['fitness']['solved']==sum(x['fitness']['solves'])
        for i,a in enumerate(r['attempts'][arm],1):
            if independent:
                assert a['random_seeds']==proposal_seeds(seed,i-1,True)
                assert a['sampling']['seed']==a['random_seeds']['parent']
            g=Graph.load(root/f'{arm}-attempt-{i}')
            assert g.hash()==a['artifact_hash'] and canonical_genotype(base,g)==a['canonical_genotype']
        g=Graph.load(root/f'{arm}-winner')
        assert g.hash()==r['winners'][arm]
        rows=[x for x in r['rows'] if x['arm']==arm];assert len(rows)==1
        family_count=12+int(marker)+int(numeric)
        row=rows[0];assert row['artifact_hash']==g.hash() and row['tasks']==family_count*report_per_family
        assert row['canonical_genotype']==canonical_genotype(base,g)
        assert sum(x['solved'] for x in row['task_records'])==row['tasks_solved']
        families=list(row['family_task_counts']);assert len(families)==family_count
        assert set(row['family_task_counts'].values())=={report_per_family}
        family_scores={f:sum(x['solved'] for x in row['task_records'][report_per_family*i:report_per_family*(i+1)]) for i,f in enumerate(families)}
        summary[arm]=dict(solved=row['tasks_solved'],families=family_scores,alleles=g.meta['expanded_genome' if p.get('expanded_bank') else 'two_slot_genome']['alleles'],
                          evaluations_per_second=evaluations/cost['evaluation_wall_seconds'],selection_cpu_seconds=cost['evaluation_cpu_seconds'],
                          failed_evaluations=cost['failed_evaluations'],report_reused=row['report_reused'])
    return dict(arms=summary,ledger_verified=count,peak_rss_bytes=r['peak_rss_bytes'],
                scope='one tiny matched-call pilot; receipt checks not independent replication',expanded_bank=p.get('expanded_bank',False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--report-per-family',type=int,default=2);p.add_argument('--seed',type=int,default=772081);p.add_argument('--proposals',type=int,default=4);p.add_argument('--independent-streams',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.run,Graph.load(a.base),a.seed,a.proposals,a.independent_streams,a.report_per_family),indent=2))
