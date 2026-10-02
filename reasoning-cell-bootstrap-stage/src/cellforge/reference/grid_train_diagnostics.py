# SPDX-License-Identifier: MIT-0
"""Training-pair diagnostics only; rejects query/report fields at its API."""
from .grid_false_positives import attempts
from .arc_grid_operator import validate_grid
from .grid_curriculum import classify_train_families

def pixel_comparison(expected,actual):
 expected=validate_grid(expected);actual=validate_grid(actual)
 eh,ew=len(expected),len(expected[0]);ah,aw=len(actual),len(actual[0])
 h,w=max(eh,ah),max(ew,aw);match=0;differences=[]
 for y in range(h):
  for x in range(w):
   e=expected[y][x] if y<eh and x<ew else None;a=actual[y][x] if y<ah and x<aw else None
   if e is not None and a is not None and e==a:match+=1
   elif e!=a and len(differences)<8:differences.append({'row':y,'col':x,'expected':e,'actual':a})
 return {'expected_shape':[eh,ew],'predicted_shape':[ah,aw],'shape_match':(eh,ew)==(ah,aw),'pixel_accuracy':match/(h*w),'difference_sample':differences}

def diagnose_train(task,predictor):
 if not isinstance(task,dict) or set(task)!={'train'}:raise ValueError('train-only task required; query/report fields prohibited')
 demos=task['train']
 if not isinstance(demos,list) or not demos:raise ValueError('nonempty train pairs required')
 for d in demos:
  if set(d)!={'input','output'}:raise ValueError('sanitized demonstration fields required')
  validate_grid(d['input']);validate_grid(d['output'])
 ranks={0,1};rows=[];invalid=0
 for i,d in enumerate(demos):
  try:
   outputs=attempts(predictor,demos,d['input']);matched=[j for j,g in enumerate(outputs) if g==validate_grid(d['output'])];ranks.intersection_update(matched)
   rows.append({'demonstration':i,'exact_ranks':matched,'attempts':[pixel_comparison(d['output'],[list(row) for row in g]) for g in outputs]})
  except (ValueError,KeyError,IndexError,TypeError) as e:
   invalid+=1;ranks.clear();rows.append({'demonstration':i,'error_type':type(e).__name__})
 consistent=bool(ranks);shape_possible=all(any(x['shape_match'] for x in row.get('attempts',[])) for row in rows)
 return {'evidence':'train-pairs','partition':'development','consistent_ranks':sorted(ranks),'demonstration_consistent':consistent,'invalid_demonstrations':invalid,'failure_class':'none' if consistent else ('observed-shape-mismatch' if not shape_possible else 'pixel-rule-or-routing'),'train_family_tags':list(classify_train_families(demos)),'demonstrations':rows,'limitations':'observed output failures do not distinguish missing capability from bad routing; per-demo best-rank pixels do not establish a common rule or unseen query generalization'}
