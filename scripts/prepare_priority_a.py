# SPDX-License-Identifier: MIT-0
"""Freeze Priority A corpora and actual bred starts before maker proposals."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.construction.maker_diagnostics import train_counterexamples
from expanded_grid_genetics import crossover
from run_grid_population_matched import frozen_expanded_corpora
from build_grid_periodic_marker_gene import examples

ROOT=Path('runs/priority-a-v2')
SEEDS=(775101,775111,775121)


def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def corpus(seed,per_family):
    original,_,_,_=frozen_expanded_corpora(per_family,seed,seed+50000000,per_family,True,True)
    new=[]
    for variant in range(4):
        for t in examples(seed+variant*10000,per_family):
            t=copy.deepcopy(t)
            for d in t['train']+t['test']:
                for k in ('input','output'):
                    a=np.asarray(d[k])
                    if variant==1:a=a[::-1,:]
                    elif variant==2:a=a.T
                    elif variant==3:
                        b=a.copy();b[a==3]=7;b[a==7]=3;a=b
                    d[k]=a.tolist()
            t['family']='periodic_completion_'+str(variant);new.append(t)
    # Old-family queries stay excluded from selection. New-family report is
    # retained separately; maker evidence contains only supplied train pairs.
    return original,new


def main():
    if ROOT.exists(): raise ValueError('fresh run required')
    inventory=Path('runs/nine-slot-population-inventory-v1')
    left=Graph.load(inventory/'parent-crop-fill');right=Graph.load(inventory/'parent-object-adjacency-scaling')
    specs=[];prepared=[];all_train=set();all_report=set()
    for seed in SEEDS:
        groups={};commitments={}
        for label,offset,count in [('selection',100000,2),('fresh',200000,2),('archive',300000,2),('report',400000,4)]:
            old,new=corpus(seed+offset,count)
            if label!='selection':
                # Fresh/archive have separate query targets for admission;
                # report queries are excluded from all maker evidence.
                _,old_report,_,_=frozen_expanded_corpora(count,seed+offset+50000000,seed+offset,count,True,True)
                groups[label]=old_report+new
                fingerprints={digest(dict(train=t['train'],test=t['test'])) for t in groups[label]}
                assert len(fingerprints)==len(groups[label]) and not fingerprints&all_report
                all_report |= fingerprints
            else:
                groups[label]=[dict(train=t['train'],task_id=f'{label}-{i}',family=t.get('family')) for i,t in enumerate(old+new)]
                fingerprints={digest(t['train']) for t in groups[label]}
                assert len(fingerprints)==len(groups[label]) and not fingerprints&all_train
                all_train |= fingerprints
            commitments[label]=digest(groups[label])
        child=crossover(left,right,seed)
        # Prioritize newly missing family in maker evidence; no report labels.
        diagnostics=train_counterexamples(child,groups['selection'][-8:],limit=3,per_task_limit=1)
        assert diagnostics['counterexamples']
        specs.append(dict(seed=seed,starting_hash=child.hash(),commitments=commitments,
            selection_tasks=36,fresh_tasks=36,archive_tasks=36,report_tasks=72))
        prepared.append((seed,child,groups,diagnostics))
    # Check train/demo duplication between report and any selection evidence.
    assert not {digest(t['train']) for _,_,g,_ in prepared for t in g['report']} & all_train
    ROOT.mkdir();left.save(ROOT/'left');right.save(ROOT/'right')
    policy=dict(protocol='priority-a-new-cells-three-arm/1',runs=specs,
        arms=['breeding_only','random_repair','maker_repair'],charged_proposals_per_arm_per_seed=16,
        starting_parents=[left.hash(),right.hash()],gradients=False,api_calls=False,
        transaction_vocabulary=['set_param','put_def','import_def','import_slot_gene','set_slot_gene'],
        vocabulary_rule='same bounds and cell/wiring grammar in both repair arms; each batch explicitly frozen before scoring',
        acceptance='strict fresh niche gain; no earlier-family or DEV200 per-task losses; no FP increase; typed fixtures/caps',
        native_anchor='200 public-training tasks ordered by SHA256(bare taskID), exclude modulo-five zero; fixed manifest4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4',
        report_rule='freeze winners before report; report once; never use report results to author repairs',
        cap=dict(parameters=1024,rss_bytes=4*1024**3,address_space_bytes=3*1024**3),
        execution='sequential Hetzner2CPU4GB, nice15, numerical threads1',
        authoring_cost='recorded edit count; session token cost not measurable, never label zero',
        decision='maker must beat random in all3seed report outcomes; report paired differences and interval over3seeds with small-sample caveat',
        exposure='all source families authored/exposed; periodic family motivated by cbded52d training diagnostics; no private ARC/generalization claim',
        freeze=['cache','memo','replay','receipts-tooling','paper reconciliation before result'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='prepared before proposal; no scored arm or archive admission yet')
    (ROOT/'PLAN.json').write_text(json.dumps(policy,indent=2))
    for seed,child,groups,diagnostics in prepared:
        home=ROOT/str(seed);home.mkdir();child.save(home/'starting')
        for label,data in groups.items():(home/(label+'.json')).write_text(json.dumps(data))
        (home/'diagnostics-01.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps(dict(seeds=SEEDS,selection_tasks_per_seed=36,report_tasks_per_seed=72,
        charged_proposals_per_arm_per_seed=16,status='prepared; not scored')))


if __name__=='__main__': main()
