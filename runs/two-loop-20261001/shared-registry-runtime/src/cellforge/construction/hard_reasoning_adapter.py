"""Typed multistep signed temporal Horn reasoning; isolated development adapter.

All candidate rule firings and derived evidence execute literal graph cells.
Reference semantics are a separate label oracle, never candidate execution.
No training, reporting evaluation, or learned capability claim.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import numpy as np
try:
 from cellforge.ir.graph import Graph
 from cellforge.runtime.interp import Interpreter
 from cellforge.runtime import deduction as historical
except ModuleNotFoundError as error:
 if error.name not in ('cellforge','cellforge.ir','cellforge.runtime'):raise
 sys.path.insert(0,str(Path(r'C:\Users\wkeif\Documents\GitHub\CellForge\src')))
 from cellforge.ir.graph import Graph
 from cellforge.runtime.interp import Interpreter
 from cellforge.runtime import deduction as historical
SOURCE=Path(historical.__file__).resolve()
BUILDER_SOURCE=SOURCE.parents[3]/'scripts'/'build_deduction_specialist.py'
INTERFACE='signed-temporal-horn-cell/1'
SHAPES={'rule.W1':(1,6),'rule.b1':(1,),'any.W1':(1,1),'any.b1':(1,)}
FEATURE_ORDER=['positive_matches','positive_required','negative_matches','negative_required','time_ready','time_required']
POSITION_ORDER=tuple((key,index) for key,shape in SHAPES.items() for index in np.ndindex(shape))
CONTRACT={'interface':INTERFACE,'positions':9,'shapes':{k:list(v) for k,v in SHAPES.items()},
 'weight_bounds':[-9,9],'feature_order':FEATURE_ORDER,'activation':'step',
 'input':'{facts:[{atom,polarity}],rules:[{if:[{atom,polarity}],then:{atom,polarity},not_before:int}],query:atom,horizon:int}',
 'output':'{answer:{status,positive_round,negative_round},closure:[{atom,polarity}],trace:[round rule features/preactivation/fired]}',
 'bounds':{'atoms':32,'facts':64,'rules':64,'antecedents':4,'horizon':8},
 'semantics':'Synchronous monotone signed evidence. Explicit negatives require evidence; absent is unknown. Contradictions do not imply arbitrary atoms. Rule fires only when all signed premises supported and current round >= not_before. Earliest query evidence round is part of the scored answer.'}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _params(cell):
 if not isinstance(cell,dict):raise ValueError('cell/params dictionary required')
 if 'params' in cell:
  if cell.get('interface')!=INTERFACE:raise ValueError('signed temporal interface mismatch')
  return cell['params']
 return cell
def validate(cell):
 p=_params(cell)
 if not isinstance(p,dict) or set(p)!=set(SHAPES):raise ValueError('complete literal params required')
 for k,shape in SHAPES.items():
  try:
   raw=np.asarray(p[k],dtype=object)
   if any(type(v) not in (int,float) for v in raw.flat):raise ValueError('JSON numeric literals only')
   value=np.asarray(p[k],float)
  except (ValueError,TypeError) as error:raise ValueError('numeric literal tensor required') from error
  if value.shape!=shape or not np.isfinite(value).all() or np.any(np.abs(value)>9):raise ValueError('finite declared shapes and [-9,9] bounds required')
 return len(POSITION_ORDER)
def identity(cell):
 p=_params(cell);return digest({'interface':INTERFACE,'params':{k:np.asarray(p[k],float).tolist() for k in SHAPES}})
def _atom(atom):
 if not isinstance(atom,str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,63}',atom):raise ValueError('ASCII atom identifiers only')
 return atom
def _literal(literal):
 if not isinstance(literal,dict) or set(literal)!={'atom','polarity'}:raise ValueError('explicit atom/polarity literal required')
 name=_atom(literal['atom']);polarity=literal['polarity']
 if type(polarity) is not int or polarity not in (-1,1):raise ValueError('polarity must be integer -1 or 1')
 return (name,polarity)
def check_request(q):
 if not isinstance(q,dict) or set(q)!={'facts','rules','query','horizon'}:raise ValueError('typed signed reasoning input required')
 horizon=q['horizon']
 if type(horizon) is not int or not 1<=horizon<=8:raise ValueError('horizon 1..8')
 if not isinstance(q['facts'],list) or len(q['facts'])>64 or not isinstance(q['rules'],list) or len(q['rules'])>64:raise ValueError('facts/rules <=64')
 facts={_literal(x) for x in q['facts']};names={_atom(q['query'])}|{x[0] for x in facts};rules=[]
 for rule in q['rules']:
  if not isinstance(rule,dict) or set(rule)!={'if','then','not_before'} or not isinstance(rule['if'],list) or len(rule['if'])>4:raise ValueError('bounded signed rule schema')
  when=rule['not_before']
  if type(when) is not int or not 1<=when<=8:raise ValueError('not_before 1..8')
  body={_literal(x) for x in rule['if']};head=_literal(rule['then']);names.add(head[0]);names.update(x[0] for x in body)
  rules.append((body,head,when))
 if len(names)>32:raise ValueError('at most32 distinct atoms')
 return facts,rules,sorted(names)
def _answer(state,query,first):
 positive=(query,1) in state;negative=(query,-1) in state
 status='both' if positive and negative else 'entailed' if positive else 'refuted' if negative else 'unknown'
 return {'status':status,'positive_round':first.get((query,1)),'negative_round':first.get((query,-1))}
def _closure(state):return [{'atom':name,'polarity':sign} for name,sign in sorted(state)]
def reference(q):
 """Independent symbolic development oracle; no candidate coefficients."""
 state,rules,names=check_request(q);first={x:0 for x in state}
 for round_ in range(1,q['horizon']+1):
  new={head for body,head,when in rules if round_>=when and body<=state}
  for literal in new-state:first[literal]=round_
  state=state|new
 return {'answer':_answer(state,q['query'],first),'closure':_closure(state)}
def graph(params):
 validate(params)
 defs={'rule_gate':{'kind':'cell','act':'step','params':{'W1':'rule.W1','b1':'rule.b1'}},
       'evidence_gate':{'kind':'cell','act':'step','params':{'W1':'any.W1','b1':'any.b1'}},
       'signed_step':{'kind':'bundle','role':'cluster','inputs':['features','conclusions','state'],'nodes':[
        {'id':'fired','use':'rule_gate','in':{'x':'$features'}},
        {'id':'votes','use':'op:matmul','in':{'a':'$conclusions','b':'fired'}},
        {'id':'derived','use':'evidence_gate','in':{'x':'votes'}},
        {'id':'next','use':'op:maximum','in':{'a':'$state','b':'derived'}}],
        'outputs':{'state':'next','fired':'fired'}}}
 g=Graph(defs,{k:np.asarray(v,float) for k,v in _params(params).items()},
         {'step':'signed_step'},{'task':'deduction','deduction_interface':INTERFACE,'construction':'historical graph interpreter with signed/clock features; candidate decisions numerical'})
 g.validate();return g
def predict(params,q):
 """Weight-driven full task execution, synchronous through bounded horizon."""
 facts,rules,names=check_request(q);g=graph(params);engine=Interpreter(g)
 ids={(name,sign):2*i+(0 if sign==1 else 1) for i,name in enumerate(names) for sign in (1,-1)}
 size=len(ids);state=np.zeros((1,size,1));first={x:0 for x in facts}
 for literal in facts:state[0,ids[literal],0]=1
 conclusions=np.zeros((1,size,len(rules)))
 for i,(_,head,_) in enumerate(rules):conclusions[0,ids[head],i]=1
 trace=[]
 for round_ in range(1,q['horizon']+1):
  features=np.zeros((1,len(rules),6))
  for i,(body,head,when) in enumerate(rules):
   pos=[x for x in body if x[1]==1];neg=[x for x in body if x[1]==-1]
   features[0,i]=[sum(state[0,ids[x],0] for x in pos),len(pos),sum(state[0,ids[x],0] for x in neg),len(neg),float(round_>=when),1.]
  # A zero-rule task has no rule decisions; avoid empty matmul interpreter stats.
  if rules:
   out=engine.run_bundle(g.entries['step'],{'features':features,'conclusions':conclusions,'state':state},'/signed-deduction')
   next_state=np.asarray(out['state']);fired=np.asarray(out['fired'])
   if next_state.shape!=state.shape or not np.isin(next_state,[0,1]).all() or fired.shape!=(1,len(rules),1) or not np.isin(fired,[0,1]).all():raise ValueError('invalid signed Boolean graph output')
   pre=features@np.asarray(_params(params)['rule.W1'],float).T+np.asarray(_params(params)['rule.b1'],float)
   for i in range(len(rules)):
    trace.append({'round':round_,'rule_index':i,'features':features[0,i].tolist(),'preactivation':float(pre[0,i,0]),'fired':bool(fired[0,i,0])})
   for literal,index in ids.items():
    if next_state[0,index,0] and not state[0,index,0]:first[literal]=round_
   state=next_state
  # No early fixed-point stop: delayed rules can become ready in future rounds.
 closed={literal for literal,index in ids.items() if state[0,index,0]}
 return {'answer':_answer(closed,q['query'],first),'closure':_closure(closed),'trace':trace,
         'rounds':q['horizon'],'graph_hash':g.hash(),'execution':engine.stats.as_dict()}
BASELINE_PARAMS={'rule.W1':[[1.,-1.,0.,0.,0.,0.]],'rule.b1':[0.5],'any.W1':[[1.]],'any.b1':[0.]}
BASELINE_CELL={'interface':INTERFACE,'params':deepcopy(BASELINE_PARAMS),'provenance':{
 'kind':'historical-positive-Horn-baseline-lifted-to-new-signed-temporal-interface',
 'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'builder_source':str(BUILDER_SOURCE),'builder_sha256':hashlib.sha256(BUILDER_SOURCE.read_bytes()).hexdigest() if BUILDER_SOURCE.exists() else None,
 'numeric_lift':'all_W[1,-1],all_b.5,any_W[1],any_b0 exactly inherited; new negative/time feature coordinates explicitly zero; not a trained or fabricated-degraded founder',
 'inherited_positions':[0,1,6,7,8],'structural_added_positions':[2,3,4,5],'training':'none','learning_claim':False}}
def lit(atom,sign=1):return {'atom':atom,'polarity':sign}
def rule(body,head,when=1):return {'if':body,'then':head,'not_before':when}
def task(id_,facts,rules,query,horizon=6):return {'id':id_,'input':{'facts':facts,'rules':rules,'query':query,'horizon':horizon}}
DEVELOPMENT_CASES=[
 task('dev-positive-three-hop',[lit('start')],[rule([lit('start')],lit('ready')),rule([lit('ready')],lit('gate')),rule([lit('gate')],lit('goal'))],'goal'),
 task('dev-negative-missing-chain',[lit('start')],[rule([lit('revoked',-1)],lit('ready')),rule([lit('ready')],lit('gate')),rule([lit('gate')],lit('goal'))],'goal'),
 task('dev-mixed-conjunction',[lit('token')],[rule([lit('token'),lit('clearance',-1)],lit('ready')),rule([lit('ready')],lit('gate')),rule([lit('gate')],lit('goal'))],'goal'),
 task('dev-supported-delayed-negative',[lit('denied',-1)],[rule([lit('denied',-1)],lit('locked'),2),rule([lit('locked')],lit('can_enter',-1),3),rule([lit('can_enter',-1)],lit('unsafe'),4)],'unsafe'),
 task('dev-contradictory-query',[lit('start'),lit('denied',-1)],[rule([lit('start')],lit('ready')),rule([lit('ready')],lit('gate'),3),rule([lit('denied',-1)],lit('gate',-1),4)],'gate'),
 task('dev-contradiction-no-explosion',[lit('a'),lit('a',-1)],[rule([lit('missing',-1)],lit('b')),rule([lit('b')],lit('goal'))],'goal'),
 task('dev-empty-premise-delayed',[],[rule([],lit('ready'),3),rule([lit('ready')],lit('gate')),rule([lit('gate')],lit('goal'))],'goal'),
 task('dev-positive-cycle-unknown',[],[rule([lit('a')],lit('b')),rule([lit('b')],lit('a')),rule([lit('b')],lit('goal'))],'goal'),
]
FROZEN_CASES=[
 task('frozen-late-negative-mixed',[lit('permit'),lit('veto',-1)],[rule([lit('permit'),lit('veto',-1)],lit('a'),4),rule([lit('a')],lit('b',-1),5),rule([lit('b',-1)],lit('target'),6)],'target',8),
 task('frozen-conflict-unrelated',[lit('x'),lit('x',-1)],[rule([lit('other',-1)],lit('next')),rule([lit('next')],lit('target'))],'target',7),
]
MANIFEST={'interface':INTERFACE,'development_ids':[c['id'] for c in DEVELOPMENT_CASES],
 'development_sha256':digest(DEVELOPMENT_CASES),'frozen_ids':[c['id'] for c in FROZEN_CASES],
 'frozen_sha256':digest(FROZEN_CASES),'frozen_evaluated':False,'source_sha256':BASELINE_CELL['provenance']['source_sha256']}
def grade(cell):
 validate(cell);traces=[];outcomes={};errors=[]
 for case in DEVELOPMENT_CASES:
  expected=reference(case['input'])
  try:
   output=predict(_params(cell),case['input']);predicted={k:output[k] for k in ('answer','closure')};error=None;steps=output['trace']
  except (ValueError,TypeError,KeyError,IndexError) as exc:predicted=None;error=type(exc).__name__+':'+str(exc);errors.append(error);steps=[]
  hit=predicted==expected;outcomes[case['id']]=hit
  traces.append({'id':case['id'],'input':case['input'],'expected':expected,'predicted':predicted,'steps':steps,'correct':hit,'error':error})
 return {'partition':'development','solved':sum(outcomes.values()),'total':len(outcomes),'outcomes':outcomes,'traces':traces,'errors':errors,'learning_claim':False}
def crossover(left,right,left_positions):
 validate(left);validate(right)
 if 'params' not in left or 'params' not in right:raise ValueError('typed parents required')
 chosen=list(left_positions)
 if len(set(chosen))!=len(chosen) or any(type(i) is not int or not 0<=i<9 for i in chosen):raise ValueError('unique scalar positions 0..8')
 chosen=set(chosen);lp=left['params'];rp=right['params'];differing={i for i,(k,j) in enumerate(POSITION_ORDER) if np.asarray(lp[k])[j]!=np.asarray(rp[k])[j]}
 if not (chosen&differing) or not (differing-chosen):raise ValueError('distinguishing contribution from both parents required')
 params={k:np.asarray(rp[k],float).copy() for k in SHAPES};ids={'left':identity(left),'right':identity(right)};inheritance=[]
 for i,(k,j) in enumerate(POSITION_ORDER):
  side='left' if i in chosen else 'right';v=float(np.asarray(lp[k] if side=='left' else rp[k])[j]);params[k][j]=v
  inheritance.append({'position':i,'parameter':k,'index':list(j),'parent':side,'parent_id':ids[side],'value':v})
 child={'interface':INTERFACE,'params':{k:v.tolist() for k,v in params.items()},'parents':ids,'inheritance':inheritance,
        'operator':'literal-scalar-inheritance','charged_positions':9,'authored_positions':0}
 validate(child);return child

DEVELOPMENT_DECIMAL_CASES=[]

