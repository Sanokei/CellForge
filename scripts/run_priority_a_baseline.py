# SPDX-License-Identifier: MIT-0
"""Priority A baseline/admission populations; never loads report corpus."""
import argparse,hashlib,json,time,resource
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh baseline output required')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    plan=json.loads(a.plan.read_text());paths=[x for h,x in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in a.training.glob('*.json')) if int(h,16)%5!=0][:200]
    assert len(paths)==200
    content=hashlib.sha256()
    for x in paths:content.update(x.stem.encode()+b':'+hashlib.sha256(x.read_bytes()).digest())
    assert content.hexdigest()=='4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4'
    anchor=[(x.stem,json.loads(x.read_text())) for x in paths]
    a.out.mkdir();start=time.perf_counter();rows=[]
    for spec in plan['runs']:
        home=a.plan.parent/str(spec['seed']);g=Graph.load(home/'starting');assert g.hash()==spec['starting_hash']
        scores={}
        for label in ('fresh','archive','anchor'):
            if label=='anchor':tasks=anchor
            else:
                corpus=json.loads((home/(label+'.json')).read_text());assert digest(corpus)==spec['commitments'][label]
                tasks=[(str(i),t) for i,t in enumerate(corpus)]
            scores[label]=score_with_consistency(tasks,GridPredictor(g,backend='compiled'),record_tasks=True)
            print(json.dumps(dict(seed=spec['seed'],group=label,solved=scores[label]['tasks_solved'],fp=scores[label]['demo_consistent_but_wrong_tasks'])),flush=True)
        row=dict(seed=spec['seed'],hash=g.hash(),scores=scores);rows.append(row)
        (a.out/(str(spec['seed'])+'.json')).write_text(json.dumps(row,indent=2))
    r=dict(plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),rows=rows,
        wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        report_loaded=0,proposal_evaluations=0,native_content_manifest=content.hexdigest(),
        scope='common preproposal baselines; separate from matched16proposal budgets')
    (a.out/'result.json').write_text(json.dumps(r,indent=2))


if __name__=='__main__':main()
