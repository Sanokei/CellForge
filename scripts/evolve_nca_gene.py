"""Bounded discrete mu/lambda graph evolution on private fixed DEV train pairs.

This operator must receive train-pairs-only items. It never reads test answers.
Root launches isolated snapshots; this script starts no remote jobs.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from cellforge.construction.nca_gene import build_nca_gene,mutate_nca_gene,crossover_nca_gene
from cellforge.runtime.grid import GridPredictor


def training_pixel_fitness(graph,items):
    predictor=GridPredictor(graph); correct=0; total=0; failures=0
    for item in items:
        for pair in item['train']:
            expected=pair['output']; size=sum(len(row) for row in expected); total+=size
            try:
                predicted=predictor([pair],pair['input'])[0]
                if len(predicted)==len(expected) and all(len(a)==len(b) for a,b in zip(predicted,expected)):
                    correct+=sum(a==b for ra,rb in zip(predicted,expected) for a,b in zip(ra,rb))
            except Exception:
                failures+=1
    return {'train_pair_pixel_accuracy':correct/total if total else 0.,'correct_pixels':correct,'pixels':total,'failed_pairs':failures,'eligible':bool(total) and not failures}


def validate_items(items):
    if not isinstance(items,list) or not 1<=len(items)<=5: raise ValueError('one to five fixed development tasks required')
    for item in items:
        if set(item)-{'task_hash','family','train'} or not item.get('train'):
            raise ValueError('train-pairs-only private manifest; no test fields permitted')
        task_hash=item.get('task_hash','')
        if len(task_hash)!=64 or any(c not in '0123456789abcdef' for c in task_hash) or int(task_hash,16)%5==0:
            raise ValueError('only SHA256 modulo5 nonzero development IDs permitted')
        for pair in item['train']:
            if set(pair)!={'input','output'}: raise ValueError('only demonstration input/output permitted')
    if len({x['task_hash'] for x in items})!=len(items): raise ValueError('unique private task hashes required')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--items',required=True); p.add_argument('--private-out',required=True); p.add_argument('--public',required=True); p.add_argument('--seed',type=int,default=51001)
    p.add_argument('--proposals-per-arm',type=int,default=300); p.add_argument('--seconds',type=float,default=120.)
    args=p.parse_args(); root=Path(args.private_out); public=Path(args.public)
    if root.exists() or public.exists(): raise ValueError('fresh output paths required')
    if not 1<=args.proposals_per_arm<=300 or not 0<args.seconds<=120: raise ValueError('bounded300proposals/120seconds required')
    raw=Path(args.items).read_bytes(); items=json.loads(raw); validate_items(items)
    root.mkdir(parents=True,mode=0o700); start=time.perf_counter(); cache={}; archive={}; ledger=[]; evaluations=0
    founders=[build_nca_gene(2,s) for s in ('identity','neighbor_majority')]
    policy={'protocol':'nca-discrete-pilot/2','items_sha256':hashlib.sha256(raw).hexdigest(),'tasks':len(items),'selection':'DEV demonstration-pair pixel accuracy only; test fields forbidden','mu':4,'lambda':8,'arms':['evolution','random_founder'],'proposals_per_arm_cap':args.proposals_per_arm,'paired_wall_budget_seconds':args.seconds,'seed':args.seed,'gradient_steps':0,'test_evaluations':0,'operator_schedule':['discrete_fixed_slot_mutation','typed_row_import'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'founder_diversity':'two distinct authored seeds duplicated into4population slots'}
    (root/'policy.json').write_text(json.dumps(policy,indent=2))
    def score(graph):
        nonlocal evaluations
        key=graph.hash()
        if key not in cache: cache[key]=training_pixel_fitness(graph,items); evaluations+=1
        archive[key]=graph
        return cache[key]
    def rank(graph):
        s=score(graph); return (not s['eligible'],-s['train_pair_pixel_accuracy'],graph.hash())
    for f in founders: score(f)
    populations={arm:list(founders)*2 for arm in policy['arms']}; curves={arm:[] for arm in populations}
    proposals=0; generation=0; proposal_archive=[]
    def freeze_generation():
        for arm in populations:
            winner=min(populations[arm],key=rank)
            curves[arm].append({'generation':generation,'hash':winner.hash(),'fitness':score(winner)})
    freeze_generation()
    while proposals<args.proposals_per_arm and time.perf_counter()-start<args.seconds:
        generation+=1; children={arm:[] for arm in populations}
        for index in range(min(8,args.proposals_per_arm-proposals)):
            if time.perf_counter()-start>=args.seconds: break
            seed=args.seed+proposals
            for arm in populations:
                parent=populations[arm][index%4] if arm=='evolution' else founders[index%2]
                donor=populations[arm][(index+1)%4] if arm=='evolution' else founders[1-index%2]
                try:
                    child=mutate_nca_gene(parent,seed) if index%2==0 else crossover_nca_gene(parent,donor,seed%12,f'g{generation}_s{seed}_')
                    score(child); children[arm].append(child); proposal_archive.append(child)
                    entry={'arm':arm,'generation':generation,'seed':seed,'operator':policy['operator_schedule'][index%2],'parent_hash':parent.hash(),'donor_hash':donor.hash(),'child_hash':child.hash(),'fitness':score(child),'lineage':child.meta.get('nca_mutation',child.meta.get('nca_crossover'))}
                except Exception as exc:
                    entry={'arm':arm,'generation':generation,'seed':seed,'parent_hash':parent.hash(),'donor_hash':donor.hash(),'failure_type':type(exc).__name__}
                ledger.append(entry)
                with (root/'proposals.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
            proposals+=1
        for arm in populations: populations[arm]=sorted(populations[arm]+children[arm],key=rank)[:4]
        freeze_generation()
        (root/'selection-progress.json').write_text(json.dumps({'curves':curves,'proposals_per_arm':proposals,'elapsed_seconds':time.perf_counter()-start}))
    for key,graph in archive.items(): graph.save(root/'archive'/key)
    (root/'winners-frozen.json').write_text(json.dumps({arm:curves[arm][-1]['hash'] for arm in populations}))
    elapsed=time.perf_counter()-start
    result={'policy':policy,'curves':curves,'proposals':ledger,'proposals_per_arm':proposals,'unique_evaluations':evaluations,'elapsed_seconds':elapsed,'unique_evaluations_per_second':evaluations/elapsed,'logical_proposals_per_second':2*proposals/elapsed,'cache_scope':'shared exact graph hash; identical first-generation proposals may reuse fitness; logical proposal counts remain paired','aggregate_only':True,'limitations':'at most five DEV tasks; no test answers/evaluation, no competition score or demonstrated generalization; paired wallbudget includesfounders, archivalserialization mayexceedbudget; independent test scorer required afterwinnerfreeze'}
    (root/'receipt.json').write_text(json.dumps(result,indent=2)); public.parent.mkdir(parents=True,exist_ok=True); public.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

