# SPDX-License-Identifier: MIT-0
"""Freeze recurrence-aware literal-cell proposals and matched random grammar."""
import json,hashlib
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.construction.transactions import apply_ops
from cellforge.construction.maker_diagnostics import train_counterexamples
from priority_a_breeding import breed_champion
from build_grid_supported_periodic_gene import build_joint_supported_periodic_gene,supported_periodic_import_ops

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def main():
    plan=Path('runs/priority-a-v2/PLAN.json');previous=Path('runs/priority-a-batch04-v1')
    prior=json.loads((previous/'result.json').read_text())
    if prior['proposals_per_arm_per_seed']!=4:raise ValueError('complete fourth round required')
    out=plan.parent/'batch-05';out.mkdir(exist_ok=False)
    mate_path=plan.parent/'left';mate=Graph.load(mate_path)
    vocabulary=dict(periods=list(range(1,11)),axes=[0,1,2],minimum_support=[1,2,3,4],
        count_weights=[-1.,0.,1.,2.],selector_weights=[-1.,0.,1.,2.],selector_biases=[-1.5,-.5,.5,1.5],
        slot_edits=9,imported_gene_always_active=True)
    batch=dict(protocol='priority-a-repair-batch/1',proposal=5,runs=[],vocabulary=vocabulary,
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        information_channel='bounded exposed DEV200 train-pair failures, retraced on each actual bred graph; no report data',
        limitation='maker fresh set saturated; existing admission intentionally unchanged')
    for seed in (775101,775111,775121):
        rng=np.random.default_rng(seed+5000000);row=dict(seed=seed,repairs={})
        for arm in ('breeding_only','random_repair','maker_repair'):
            champion=Graph.load(previous/f'{seed}-{arm}-selected');bred=breed_champion(champion,mate,seed+5000000)
            bred.save(out/f'{seed}-{arm}-bred')
            source=Path('runs/priority-a-native-diag-v1')/(f'{seed}-{arm}.json')
            evidence=json.loads(source.read_text())
            if evidence['graph_hash']!=champion.hash():raise ValueError('exact native diagnostic parent required')
            tasks=[dict(task_id=e['task_id'],family='public-training',train=[dict(input=e['input'],output=e['expected'])]) for e in evidence['counterexamples']]
            diag=train_counterexamples(bred,tasks,limit=5,per_task_limit=1)
            diag['native_evidence_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
            path=out/f'{seed}-{arm}-diagnostics.json';path.write_text(json.dumps(diag,indent=2))
            repair=dict(champion_hash=champion.hash(),mate_path=mate_path.as_posix(),mate_hash=mate.hash(),
                breeding_seed=seed+5000000,bred_hash=bred.hash(),diagnostics_path=path.as_posix(),
                diagnostics_sha256=digest(diag),ops=[],actor=arm,
                model='offline-session-maker' if arm=='maker_repair' else 'none',rationale='Charged unmodified crossover control.')
            if arm!='breeding_only':
                recipe=dict(period=3,axis=2,minimum=2,weights=np.eye(2).tolist(),selector_weight=1.,selector_bias=-.5)
                if arm=='random_repair':
                    recipe=dict(period=int(rng.choice(vocabulary['periods'])),axis=int(rng.choice(vocabulary['axes'])),
                        minimum=int(rng.choice(vocabulary['minimum_support'])),weights=rng.choice(vocabulary['count_weights'],size=(2,2)).tolist(),
                        selector_weight=float(rng.choice(vocabulary['selector_weights'])),selector_bias=float(rng.choice(vocabulary['selector_biases'])))
                donor=build_joint_supported_periodic_gene(**recipe);donor_path=out/f'{seed}-{arm}-donor';donor.save(donor_path)
                slots=champion.meta['expanded_genome']['slots'];slot=9 if arm=='maker_repair' else int(rng.choice(slots))
                ops=supported_periodic_import_ops(bred,donor,slot,'a05_')
                gene='a05_joint_supported_periodic.gene'
                choices=list(champion.meta['expanded_genome']['alleles']) if arm=='maker_repair' else rng.choice(bred.meta['expanded_genome']['available_genes'],size=len(slots)).tolist()
                choices[slots.index(slot)]=gene
                ops += [dict(op='set_slot_gene',slot=s,gene=g) for s,g in zip(slots,choices)]
                candidate=apply_ops(bred,ops,resolver=lambda _:donor)[0];candidate.save(out/f'{seed}-{arm}-frozen-candidate')
                repair.update(ops=ops,donor_path=donor_path.as_posix(),donor_hash=donor.hash(),materialized_hash=candidate.hash(),recipe=recipe,
                    rationale='Literal repeated-line support and selector weights, with global sparse-template fallback; maker preserves niche coverage, random uses identical bounded grammar and an active imported gene.')
            row['repairs'][arm]=repair
        batch['runs'].append(row)
    (out/'PROPOSALS.json').write_text(json.dumps(batch,indent=2));print(json.dumps(dict(proposal=5,report_loaded=0)))

if __name__=='__main__':main()
