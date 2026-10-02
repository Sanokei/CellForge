# SPDX-License-Identifier: MIT-0
"""Paired finite-age ablation from all predeclared receipts; no model scoring."""
import argparse,json
from pathlib import Path
from summarize_object_replications import summarize,SEEDS,ARMS


def compare(root):
    young=summarize(root,expected_age=3)
    old=summarize(root,prefix='grid-object-age8',expected_age=8)
    paired=[]
    for seed,a,b in zip(SEEDS,young['runs'],old['runs']):
        p=Path(root)
        x=json.loads((p/f'grid-object-replication-{seed}'/'result.json').read_text())['policy']
        y=json.loads((p/f'grid-object-age8-{seed}'/'result.json').read_text())['policy']
        differing={k for k in set(x)|set(y) if x.get(k)!=y.get(k)}
        if differing-{'max_age','source_sha256'}:raise ValueError('unmatched non-age policy: '+str(differing))
        paired.append(dict(seed=seed,policy_differences=sorted(differing),
                           arms={arm:dict(age3_solved=a['arms'][arm]['solved'],
                                          age8_solved=b['arms'][arm]['solved'],
                                          solved_delta=b['arms'][arm]['solved']-a['arms'][arm]['solved'],
                                          age3_cpu=a['arms'][arm]['cpu_seconds'],age8_cpu=b['arms'][arm]['cpu_seconds']) for arm in ARMS}))
    return dict(protocol='paired-finite-lifetime-ablation/1',age3=young,age8=old,paired=paired,
                scope='same20 exposed synthetic tasks, same three search seeds; no native query score',
                limitation='age8 exceeds four-proposal horizon; finite-age setting isolates founder persistence here, not lifetime ecosystem performance',
                gradients=False,final_evaluations=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('fresh paired output required')
    r=compare(a.root);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'age3':r['age3']['aggregate'],'age8':r['age8']['aggregate'],'paired':r['paired']}))


if __name__=='__main__':main()
