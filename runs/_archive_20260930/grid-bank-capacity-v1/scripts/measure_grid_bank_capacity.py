# SPDX-License-Identifier: MIT-0
"""Offline finite-genome coverage diagnosis; no native data or host inference."""
import argparse,itertools,json,time
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from cellforge.reference.grid_curriculum import FAMILIES,generate_task
from build_grid_adjacency_recolor_gene import generate_adjacency_family
from build_grid_object_association_child import family as object_family
from measure_grid_repeat_genes import generate_family
from build_grid_two_slot_genome import instantiate_genome
from run_grid_population_matched import canonical_genotype,evaluate_selection,digest


def curriculum(seed):
    groups=[generate_task(f,seed+j*10000+i) for j,f in enumerate(FAMILIES) for i in range(2)]
    groups+=generate_adjacency_family(2,seed)['tasks']
    for t in object_family(seed+100000)['tasks'][:2]:t['family']='object_recolor';groups.append(t)
    for f in (2,3):
        for t in generate_family(f,seed+200000+f)[:2]:t['family']=f'repeat{f}';groups.append(t)
    return groups


def selection_view(seed):
    tasks=curriculum(seed)
    # Discard generated queries and annotations before returning to selector.
    return [{'train':t['train']} for t in tasks]


def main():
    import resource
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    base=Graph.load(a.base);bank=base.meta['two_slot_genome']['available_genes'];assert len(bank)==7
    graphs=[instantiate_genome(base,g) for g in itertools.product(bank,repeat=2)]
    selection=selection_view(2997000)
    policy=dict(protocol='seven-gene-capacity-diagnostic/1',base_hash=base.hash(),genomes=49,
                selection_seed=2997000,report_seed=3007000,selection_manifest=digest(selection),selection_tasks=len(selection),
                selection_demo_pairs=sum(len(t['train']) for t in selection),graph_hashes=[g.hash() for g in graphs],
                winner_rule='maximum eligible full-demo solves, then canonical genotype hash',
                scope='enumerating finite offline genome space for diagnosis; not inference-time program search or evolutionary discovery',
                gradients=False,native_tasks_loaded=0,public_evaluation_loaded=False,audit_evaluations=0,final_evaluations=0)
    a.out.mkdir();ledger=Ledger(a.out/'ledger.jsonl');ledger.append('policy_frozen',policy,'operator')
    selections=[]
    for i,g in enumerate(graphs):
        row=dict(index=i,graph_hash=g.hash(),genotype=canonical_genotype(base,g),alleles=g.meta['two_slot_genome']['alleles'],fitness=evaluate_selection(g,selection))
        selections.append(row);ledger.append('physical_selection_evaluation',row,'operator')
        (a.out/'selection-progress.json').write_text(json.dumps(selections,indent=2)+'\n')
    eligible=[x for x in selections if not x['fitness']['failed']]
    if not eligible:raise ValueError('no eligible genotype')
    winner=min(eligible,key=lambda x:(-x['fitness']['solved'],x['genotype']))
    ledger.append('winner_frozen',winner,'operator');graphs[winner['index']].save(a.out/'selected-winner')
    # Generate report queries only after selection and selected hash are frozen.
    report=curriculum(3007000);records=[]
    assert not set(digest(t['train']) for t in report)&set(digest(t['train']) for t in selection)
    families=[t.get('family',t.get('metadata',{}).get('family')) for t in report]
    for i,g in enumerate(graphs):
        score=score_with_consistency([(str(j),t) for j,t in enumerate(report)],GridPredictor(g,backend='compiled'),record_tasks=True)
        row=dict(index=i,graph_hash=g.hash(),alleles=g.meta['two_slot_genome']['alleles'],score=score)
        records.append(row);ledger.append('diagnostic_report_scored',row,'operator')
        (a.out/'report-progress.json').write_text(json.dumps(records,indent=2)+'\n')
    result=dict(policy=policy,selection=selections,winner=winner,report_manifest=digest(report),families=families,records=records,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                limitation='all49 query results exposed by diagnosis; report maxima/unions are posthoc bounds, never selected deployable model scores')
    ledger.append('completed',result,'operator');result['ledger_verified']=ledger.verify()
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(selected_alleles=winner['alleles'],selection_score=winner['fitness']['solved'],
                         selected_query_score=records[winner['index']]['score']['tasks_solved'],
                         posthoc_max_query_score=max(r['score']['tasks_solved'] for r in records),ledger_verified=result['ledger_verified'])))


if __name__=='__main__':main()
