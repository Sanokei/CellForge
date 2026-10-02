# SPDX-License-Identifier: MIT-0
"""Freeze train-diagnostic-driven contextual weight mutation and controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.construction.transactions import apply_ops
from cellforge.construction.maker_diagnostics import train_counterexamples
from priority_a_breeding import breed_champion
from demonstration_mode_blueprint import mode_blueprint_ops

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=(3,4),default=3)
    args=parser.parse_args();step=args.step
    plan=Path('runs/priority-a-v2/PLAN.json');previous=Path(f'runs/priority-a-batch{step-1:02d}-v1')
    prior=json.loads((previous/'result.json').read_text())
    if prior['proposals_per_arm_per_seed']!=step-1:raise ValueError('complete prior round required')
    out=plan.parent/f'batch-{step:02d}';out.mkdir(exist_ok=False)
    batch=dict(protocol='priority-a-repair-batch/1',proposal=step,runs=[],
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        vocabulary=dict(context_weights=[-1.,0.,1.,2.],context_biases=[-1.5,-.5,.5,1.5],
            wiring='demonstrated shape change versus accumulated identity reconstruction error',
            slot_edits=9,gradient_use=False,report_access=False))
    mate_path=plan.parent/('left' if step%2 else 'right');mate=Graph.load(mate_path)
    for seed in (775101,775111,775121):
        row=dict(seed=seed,repairs={});rng=np.random.default_rng(seed+step*1000000)
        selection=json.loads((plan.parent/str(seed)/'selection.json').read_text())
        for arm in ('breeding_only','random_repair','maker_repair'):
            champion=Graph.load(previous/f'{seed}-{arm}-selected')
            bred=breed_champion(champion,mate,seed+step*1000000)
            bred.save(out/f'{seed}-{arm}-bred')
            diag=train_counterexamples(bred,selection,limit=5,per_task_limit=1)
            dp=out/f'{seed}-{arm}-diagnostics.json';dp.write_text(json.dumps(diag,indent=2))
            repair=dict(champion_hash=champion.hash(),mate_path=mate_path.as_posix(),mate_hash=mate.hash(),
                breeding_seed=seed+step*1000000,bred_hash=bred.hash(),diagnostics_path=dp.as_posix(),
                diagnostics_sha256=digest(diag),ops=[],actor=arm,model='offline-session-maker' if arm=='maker_repair' else 'none',
                rationale='Charged crossover control; repair arms share contextual cell grammar and nine slot assignments.')
            if arm!='breeding_only':
                genes=bred.meta['expanded_genome']['available_genes'];compound=next((g for g in genes if g.endswith('dense_crop_fill.gene')),None)
                ops=[];resolver=None
                if compound is None:
                    donor_path=plan.parent/'batch-02'/f'{seed}-maker_repair-donor';donor=Graph.load(donor_path)
                    repair.update(donor_path=donor_path.as_posix(),donor_hash=donor.hash())
                    resolver=lambda ref:donor if ref==donor.hash() else None
                    ops.append(dict(op='import_slot_gene',**{'from':donor.hash()},name='dense_crop_fill.gene',prefix=f'a{step:02d}_',slot=9))
                    compound=f'a{step:02d}_dense_crop_fill.gene'
                intermediate=apply_ops(bred,ops,resolver=resolver)[0] if ops else bred
                weight,bias=((2.,-.5) if step==4 else (1.,-.5)) if arm=='maker_repair' else (float(rng.choice(batch['vocabulary']['context_weights'])),float(rng.choice(batch['vocabulary']['context_biases'])))
                if 'context_mode_gate' in intermediate.defs:
                    ops += [dict(op='set_param',name='context_mode.W',value=[[weight]]),dict(op='set_param',name='context_mode.b',value=[bias])]
                else:ops+=mode_blueprint_ops(intermediate,compound,weight,bias)
                slots=champion.meta['expanded_genome']['slots'];choices=list(champion.meta['expanded_genome']['alleles'])
                if arm=='maker_repair':
                    if step==3:choices[slots.index(10 if seed==775121 else 9)]=compound
                    else:
                        periodic=next(g for g in intermediate.meta['expanded_genome']['available_genes'] if g.endswith('periodic.gene'))
                        choices=[compound,periodic,'joint_tile_repeat.gene','scaling_bank2_repeat.gene',
                            'scaling_bank3_repeat.gene','object_bank_largest4.gene','bank_adjacency_recolor.gene',
                            'joint_mirror_repeat.gene','joint_numeric_adjacency_recolor.gene']
                else:choices=rng.choice(intermediate.meta['expanded_genome']['available_genes'],size=len(slots)).tolist()
                ops += [dict(op='set_slot_gene',slot=s,gene=g) for s,g in zip(slots,choices)]
                candidate=apply_ops(bred,ops,resolver=resolver)[0]
                candidate.save(out/f'{seed}-{arm}-frozen-candidate')
                repair.update(ops=ops,materialized_hash=candidate.hash(),context_weight=weight,context_bias=bias,
                    rationale=('Literal context gate weights and blueprint use train-pair shape changes to preserve thin crop while enabling fill.' if step==3 else
                        'Literal contextual gate weights plus nine specialist niches preserve separate tiling and enlargement genes demonstrated by train-only primitive diagnostics; both transformations double dimensions, so retain reconstruction routing.')+' Random control samples the identical gate and slot vocabulary.')
            row['repairs'][arm]=repair
        batch['runs'].append(row)
    (out/'PROPOSALS.json').write_text(json.dumps(batch,indent=2))
    print(json.dumps(dict(proposal=step,seeds=3,arms=3,report_loaded=0)))

if __name__=='__main__':main()
