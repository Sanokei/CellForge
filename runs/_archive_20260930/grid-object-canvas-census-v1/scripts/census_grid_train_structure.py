# SPDX-License-Identifier: MIT-0
"""Training-only gene-coverage census; never a model or inference-time solver.

Checks predeclared simple structural rules against demonstrations only. Query
fields are discarded at loading; no query score or routing decision is produced.
"""
import argparse, hashlib, json
from pathlib import Path
from collections import Counter
import numpy as np
from cellforge.governance.ledger import Ledger

EXPECTED='4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4'
RULES=('tile2','tile3','repeat2','repeat3','crop_zero','crop_majority','identity',
       'object_largest_mono','object_smallest_mono','object_largest_any','object_smallest_any',
       'erase_largest_mono','erase_smallest_mono','retain_largest_mono','retain_smallest_mono')


def object_crop(x,rule):
    """Independent diagnostic BFS, majority background, row-major tie break."""
    bg=int(np.argmax(np.bincount(x.ravel(),minlength=10)))
    h,w=x.shape;seen=set();objects=[]
    mono=rule.endswith('_mono')
    for r in range(h):
        for c in range(w):
            if (r,c) in seen or x[r,c]==bg:continue
            stack=[(r,c)];seen.add((r,c));component=[];color=x[r,c]
            while stack:
                rr,cc=stack.pop();component.append((rr,cc))
                for nr,nc in ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
                    if 0<=nr<h and 0<=nc<w and (nr,nc) not in seen and x[nr,nc]!=bg and (not mono or x[nr,nc]==color):
                        seen.add((nr,nc));stack.append((nr,nc))
            objects.append(component)
    if not objects:return x
    # Stable sort keeps first row-major discovery for equal size.
    chosen=sorted(objects,key=len,reverse='_largest_' in rule)[0]
    if rule.startswith('erase_'):
        output=x.copy()
        for r,c in chosen:output[r,c]=bg
        return output
    rr,cc=zip(*chosen);lo=(min(rr),min(cc));hi=(max(rr)+1,max(cc)+1)
    filtered=np.full(x.shape,bg,dtype=int)
    for r,c in chosen:filtered[r,c]=x[r,c]
    if rule.startswith('retain_'):return filtered
    return filtered[lo[0]:hi[0],lo[1]:hi[1]]


def transform(x,rule):
    x=np.array(x,dtype=int)
    if rule.startswith(('object_','erase_','retain_')):return object_crop(x,rule)
    if rule=='identity':return x
    if rule.startswith('tile'):return np.tile(x,(int(rule[-1]),int(rule[-1])))
    if rule.startswith('repeat'):
        k=int(rule[-1]);return np.repeat(np.repeat(x,k,axis=0),k,axis=1)
    bg=0 if rule=='crop_zero' else int(np.argmax(np.bincount(x.ravel(),minlength=10)))
    points=np.argwhere(x!=bg)
    if not len(points):return x
    lo=points.min(axis=0);hi=points.max(axis=0)+1
    return x[lo[0]:hi[0],lo[1]:hi[1]]


def census_train(task):
    if set(task)!={'train'}:raise ValueError('strict train-only view required')
    if not task['train']:raise ValueError('demonstrations required')
    matches=[];mapped=[];shapes=[]
    for rule in RULES:
        if all(np.array_equal(transform(d['input'],rule),np.array(d['output'])) for d in task['train']):matches.append(rule)
        associations={};consistent=True
        for d in task['train']:
            x=transform(d['input'],rule);y=np.array(d['output'])
            if x.shape!=y.shape:consistent=False;break
            for source,target in zip(x.ravel(),y.ravel()):
                values=associations.setdefault(int(source),set());values.add(int(target))
                if len(values)>1:consistent=False
        if consistent:mapped.append(rule)
    for d in task['train']:
        x,y=np.array(d['input']),np.array(d['output'])
        shapes.append(dict(input_shape=list(x.shape),output_shape=list(y.shape),
                           input_colors=int(len(np.unique(x))),output_colors=int(len(np.unique(y)))))
    return dict(common_exact_rules=matches,common_rules_with_color_association=mapped,
                shape_changes=any(d['input_shape']!=d['output_shape'] for d in shapes),demonstrations=shapes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('fresh output required')
    directory=Path(a.private)/'arcagi2/data/training'
    paths=[p for h,p in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in directory.glob('*.json')) if int(h,16)%5!=0][:200]
    if len(paths)!=200:raise ValueError('DEV200 missing')
    manifest=hashlib.sha256();tasks=[]
    for path in paths:
        raw=path.read_bytes();manifest.update(path.stem.encode()+b':'+hashlib.sha256(raw).digest())
        data=json.loads(raw);tasks.append((path.stem,{'train':[{'input':d['input'],'output':d['output']} for d in data['train']]}))
    if manifest.hexdigest()!=EXPECTED:raise ValueError('DEV content mismatch')
    out.mkdir();ledger=Ledger(out/'ledger.jsonl')
    policy=dict(protocol='training-structure-census/1',rules=RULES,native_manifest=EXPECTED,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='diagnostic hypotheses only; not CellForge graph coverage or query score',
                query_fields_discarded=True,public_evaluation_loaded=False,audit_evaluations=0,final_evaluations=0,gradients=False)
    ledger.append('policy_frozen',policy,'operator')
    rows=[dict(task_id=tid,**census_train(task)) for tid,task in tasks]
    counts=Counter(rule for row in rows for rule in row['common_exact_rules'])
    mapped=Counter(rule for row in rows for rule in row['common_rules_with_color_association'])
    result=dict(policy=policy,tasks=len(rows),shape_changes=sum(r['shape_changes'] for r in rows),
                common_exact_rule_counts={r:counts[r] for r in RULES},
                common_color_association_rule_counts={r:mapped[r] for r in RULES},
                unclassified=sum(not r['common_exact_rules'] for r in rows),rows=rows)
    ledger.append('census_complete',{k:v for k,v in result.items() if k!='rows'},'operator')
    result['ledger_verified']=ledger.verify()
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
