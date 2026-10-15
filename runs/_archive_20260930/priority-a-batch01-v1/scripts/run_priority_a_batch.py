# SPDX-License-Identifier: MIT-0
"""Execute a frozen Priority A repair batch with full development admission."""
import argparse,hashlib,json,time,resource
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter
from cellforge.construction.maker_operator import MakerOperator
from cellforge.reference.grid_false_positives import score_with_consistency


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def section(score,manifest,source):
    return dict(partition='development',manifest=manifest,source=source,
        outcomes={r['task_id']:r['solved'] for r in score['task_records']},
        false_positive_given_consistency=score['false_positive_given_consistency'],
        wrong_consistent=score['demo_consistent_but_wrong_tasks'],
        cpu_seconds_per_task=score['inference_cpu_seconds_per_task'],tasks_per_second=score['tasks_per_second'])


def boundary_fixtures(g):
    fixtures=[]
    for size in (1,30):
        x,_,h,w=pack_grid((np.indices((size,size)).sum(0)%10).tolist())
        for phase in (0.,1.):
            ins=dict(input=x,input_height=h,input_width=w,output=x if phase==0 else np.zeros_like(x),
                output_height=h if phase==0 else np.zeros_like(h),output_width=w if phase==0 else np.zeros_like(w),
                phase=np.array([[phase]]),state=np.zeros((1,g.meta['state_dim'])))
            shapes=dict(state=[1,g.meta['state_dim']],color_logits=[1,2,30,30,10],height_logits=[1,2,30],width_logits=[1,2,30])
            fixtures.append(dict(entry=g.entries['grid_step'],inputs={k:v.tolist() for k,v in ins.items()},output_shapes=shapes))
    return fixtures


def caps(g,fixtures):
    start=time.perf_counter();plain=Interpreter(g);fast=CompiledInterpreter(g)
    for f in fixtures:
        values=[e.run_bundle(f['entry'],{k:np.asarray(v) for k,v in f['inputs'].items()}) for e in (plain,fast)]
        for k,shape in f['output_shapes'].items():
            assert values[0][k].shape==tuple(shape) and np.isfinite(values[0][k]).all()
            np.testing.assert_array_equal(values[0][k],values[1][k])
    wall=time.perf_counter()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    return dict(fixtures_passed=True,caps_passed=g.stats()['unique_param_storage']<=1024 and rss<=4*1024**3 and wall<=60,
        fixture_wall_seconds=wall,peak_rss_bytes=rss,parameter_count=g.stats()['unique_param_storage'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh batch output required')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    batch=json.loads(a.batch.read_text());assert batch['proposal']==1
    plan=json.loads(a.plan.read_text());assert batch['plan_sha256']==hashlib.sha256(a.plan.read_bytes()).hexdigest()
    paths=[x for h,x in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in a.training.glob('*.json')) if int(h,16)%5!=0][:200]
    assert len(paths)==200;content=hashlib.sha256()
    for x in paths:content.update(x.stem.encode()+b':'+hashlib.sha256(x.read_bytes()).digest())
    assert content.hexdigest()=='4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4'
    anchor=[(x.stem,json.loads(x.read_text())) for x in paths]
    a.out.mkdir();rows=[];start=time.perf_counter()
    for spec in batch['runs']:
        seed=spec['seed'];home=a.plan.parent/str(seed);parent=Graph.load(home/'starting');assert parent.hash()==spec['parent_hash']
        diag=json.loads((home/'diagnostics-01.json').read_text());assert digest(diag)==spec['diagnostics_sha256']
        prior=json.loads((a.baseline/(str(seed)+'.json')).read_text());assert prior['hash']==parent.hash()
        corpus={label:json.loads((home/(label+'.json')).read_text()) for label in ('fresh','archive')}
        planned=next(x for x in plan['runs'] if x['seed']==seed)
        assert all(digest(corpus[k])==planned['commitments'][k] for k in corpus)
        manifests={**{k:digest(v) for k,v in corpus.items()},'anchor':content.hexdigest()}
        parent_metrics={k:section(prior['scores'][k],manifests[k],'public-training-real' if k=='anchor' else 'authored-generator') for k in ('fresh','archive','anchor')}
        fixture=boundary_fixtures(parent)
        for arm in ('breeding_only','random_repair','maker_repair'):
            repair=spec['repairs'][arm]
            donor=Graph.load(repair['donor_path']) if repair.get('donor_path') else None
            if donor is not None:assert donor.hash()==repair['donor_hash']
            def resolver(ref):
                if donor is None or donor.hash()!=ref:raise ValueError('unknown exact donor')
                return donor
            def evaluate(g):
                metrics=caps(g,fixture)
                for label in ('fresh','archive','anchor'):
                    tasks=anchor if label=='anchor' else [(str(i),t) for i,t in enumerate(corpus[label])]
                    score=score_with_consistency(tasks,GridPredictor(g,backend='compiled'),record_tasks=True)
                    metrics[label]=section(score,manifests[label],'public-training-real' if label=='anchor' else 'authored-generator')
                return metrics
            op=MakerOperator(a.out/f'{seed}-{arm}.jsonl',budget=16)
            g,receipt=op.submit(parent,diagnostics=diag,ops=repair['ops'],
                provenance=dict(actor=repair['actor'],model=repair['model'],method='manual-llm-maker' if arm=='maker_repair' else 'non-llm-mutation-control',rationale=repair['rationale']),
                fixtures=fixture,resolver=resolver,evaluate=evaluate,evaluate_unchanged=True,admission_parent=parent_metrics)
            chosen=g if receipt['status']=='accepted' else parent
            chosen.save(a.out/f'{seed}-{arm}-selected')
            if g is not None:g.save(a.out/f'{seed}-{arm}-candidate')
            row=dict(seed=seed,arm=arm,receipt=receipt,selected_hash=chosen.hash());rows.append(row)
            (a.out/f'{seed}-{arm}-result.json').write_text(json.dumps(row,indent=2))
            print(json.dumps(dict(seed=seed,arm=arm,status=receipt['status'],error=receipt.get('error'),
                scores={k:sum(v['outcomes'].values()) for k,v in receipt.get('development_metrics',{}).items() if k in ('fresh','archive','anchor')})),flush=True)
    r=dict(rows=rows,wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        report_loaded=0,proposals_per_arm_per_seed=1,remaining_proposals_per_arm_per_seed=15,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'result.json').write_text(json.dumps(r,indent=2))


if __name__=='__main__':main()
