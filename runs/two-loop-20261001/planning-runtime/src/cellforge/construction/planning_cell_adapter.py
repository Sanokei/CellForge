"""Isolated typed resource-constrained planning policy:28 literal parameters.

No grid representation. Candidate logits choose every action. Domain transitions
only enforce physical rules; they do not choose a plan. BFS is a separate
reference label oracle and never executes inside candidate prediction.
"""
from collections import deque
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np
try:
 from cellforge.ir.graph import Graph
 from cellforge.runtime.interp import Interpreter
 from cellforge.runtime import interp as source_module
except ModuleNotFoundError as error:
 if error.name not in ('cellforge','cellforge.ir','cellforge.runtime'):raise
 sys.path.insert(0,str(Path(r'C:\Users\wkeif\Documents\GitHub\CellForge\src')))
 from cellforge.ir.graph import Graph
 from cellforge.runtime.interp import Interpreter
 from cellforge.runtime import interp as source_module
SOURCE=Path(source_module.__file__).resolve()
INTERFACE='resource-production-planning-cell/1'
ACTIONS=('acquire','assemble','deliver','recharge')
FEATURE_ORDER=('material_after','goods_after','remaining_goal_after','energy_after','free_cargo_after','steps_left_after')
SHAPES={'policy.W1':(4,6),'policy.b1':(4,)}
POSITION_ORDER=tuple((key,index) for key,shape in SHAPES.items() for index in np.ndindex(shape))
CONTRACT={'interface':INTERFACE,'positions':28,'shapes':{k:list(v) for k,v in SHAPES.items()},
 'weight_bounds':[-9,9],'action_order':list(ACTIONS),'feature_order':list(FEATURE_ORDER),
 'activation':'identity','selection':'each role-specific projected state scores using its corresponding affine policy row; argmax ties lowest action index; no legality masking',
 'input':'{initial:{materials,goods,delivered,energy},goal,capacity,max_energy,costs:{acquire,assemble,deliver},max_steps}',
 'output':'{status,plan,steps,state,trace}',
 'semantics':'acquire adds1 material; assemble consumes1 material adds1 goods; deliver consumes1 goods incrementsdelivered; each consumes taskdeclared energy. recharge fills battery but cannot be chosen at fullenergy. Cargo materials+goods<=capacity. Illegal selected actions fail. Complete within deadline; task scored against independent shortest BFS length.',
 'bounds':{'goal':3,'capacity':3,'max_energy':9,'cost':4,'max_steps':16}}
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _params(cell):
 if not isinstance(cell,dict):raise ValueError('cell/params dictionary required')
 if 'params' in cell:
  if cell.get('interface')!=INTERFACE:raise ValueError('planning interface mismatch')
  return cell['params']
 return cell
def validate(cell):
 p=_params(cell)
 if not isinstance(p,dict) or set(p)!=set(SHAPES):raise ValueError('complete28literal parameters required')
 for key,shape in SHAPES.items():
  try:
   raw=np.asarray(p[key],dtype=object)
   if any(type(v) not in (int,float) for v in raw.flat):raise ValueError('JSON numeric literals only')
   arr=np.asarray(p[key],float)
  except (ValueError,TypeError) as exc:raise ValueError('numeric literal tensor required') from exc
  if arr.shape!=shape or not np.isfinite(arr).all() or np.any(np.abs(arr)>9):raise ValueError('finite declared shape/[-9,9] required')
 return 28
def identity(cell):
 p=_params(cell);return digest({'interface':INTERFACE,'params':{k:np.asarray(p[k],float).tolist() for k in SHAPES}})
def _integer(value,low,high):return type(value) is int and low<=value<=high
def check_request(q):
 if not isinstance(q,dict) or set(q)!={'initial','goal','capacity','max_energy','costs','max_steps'}:raise ValueError('typed planning request schema')
 for key,low,high in [('goal',1,3),('capacity',1,3),('max_energy',1,9),('max_steps',1,16)]:
  if not _integer(q[key],low,high):raise ValueError('bounded integer '+key)
 initial=q['initial'];costs=q['costs']
 if not isinstance(initial,dict) or set(initial)!={'materials','goods','delivered','energy'}:raise ValueError('initial resource state schema')
 for key,high in [('materials',q['capacity']),('goods',q['capacity']),('delivered',q['goal']),('energy',q['max_energy'])]:
  if not _integer(initial[key],0,high):raise ValueError('bounded initial '+key)
 if initial['materials']+initial['goods']>q['capacity']:raise ValueError('initial cargo capacity exceeded')
 if not isinstance(costs,dict) or set(costs)!={'acquire','assemble','deliver'} or any(not _integer(v,1,4) for v in costs.values()):raise ValueError('integer action costs1..4')
 return deepcopy(initial)
def project(state,action,q):
 """Hypothetical physical afterstate, including negative values if illegal."""
 if action not in ACTIONS:raise ValueError('unknown typed action')
 out=dict(state)
 if action=='acquire':out['materials']+=1;out['energy']-=q['costs']['acquire']
 elif action=='assemble':out['materials']-=1;out['goods']+=1;out['energy']-=q['costs']['assemble']
 elif action=='deliver':out['goods']-=1;out['delivered']+=1;out['energy']-=q['costs']['deliver']
 else:out['energy']=q['max_energy']
 return out
def transition(state,action,q):
 out=project(state,action,q)
 if action=='recharge' and state['energy']==q['max_energy']:return None,'battery_already_full'
 if out['materials']<0 or out['goods']<0:return None,'missing_required_inventory'
 if out['energy']<0:return None,'insufficient_energy'
 if out['materials']+out['goods']>q['capacity']:return None,'cargo_capacity_exceeded'
 if out['delivered']>q['goal']:return None,'delivery_goal_exceeded'
 return out,None
def features(state,q,step):
 rows=[]
 for action in ACTIONS:
  s=project(state,action,q)
  rows.append([s['materials'],s['goods'],q['goal']-s['delivered'],s['energy'],
               q['capacity']-s['materials']-s['goods'],q['max_steps']-step-1])
 return np.asarray(rows,dtype=float)
def graph(params):
 validate(params)
 # Actual graph cell evaluates all policy rows for each projected feature row.
 # The role-aligned diagonal selects each role's own literal weight vector.
 defs={'policy':{'kind':'cell','act':'identity','params':{'W1':'policy.W1','b1':'policy.b1'}},
 'policy_scores':{'kind':'bundle','role':'cluster','inputs':['features'],'nodes':[
  {'id':'logits','use':'policy','in':{'x':'$features'}}],'outputs':{'logits':'logits'}}}
 g=Graph(defs,{k:np.asarray(v,float) for k,v in _params(params).items()},
         {'predict':'policy_scores'},{'task':'clf','planning_interface':INTERFACE})
 g.validate();return g
def predict(params,q):
 state=check_request(q);g=graph(params);engine=Interpreter(g);plan=[];trace=[]
 if state['delivered']>=q['goal']:return {'status':'complete','plan':[],'steps':0,'state':state,'trace':[],'execution':engine.stats.as_dict()}
 for step in range(q['max_steps']):
  f=features(state,q,step)
  out=engine.run_bundle(g.entries['predict'],{'features':f[None]},'/planning-policy')
  all_scores=np.asarray(out['logits'])
  if all_scores.shape!=(1,4,4) or not np.isfinite(all_scores).all():raise ValueError('finite policy logits4x4 required')
  scores=np.diag(all_scores[0]);index=int(np.argmax(scores));action=ACTIONS[index]
  after,error=transition(state,action,q)
  trace.append({'step':step+1,'state_before':dict(state),'features':f.tolist(),'scores':scores.tolist(),
                'selected_action':action,'legal':error is None,'error':error,'state_after':after})
  plan.append(action)
  if error:return {'status':'invalid_action','plan':plan,'steps':len(plan),'state':state,'trace':trace,'execution':engine.stats.as_dict()}
  state=after
  if state['delivered']==q['goal']:return {'status':'complete','plan':plan,'steps':len(plan),'state':state,'trace':trace,'execution':engine.stats.as_dict()}
 return {'status':'deadline','plan':plan,'steps':len(plan),'state':state,'trace':trace,'execution':engine.stats.as_dict()}
def reference(q):
 """Independent shortest-plan BFS labels, excluded from candidate prediction."""
 initial=check_request(q);queue=deque([(initial,[])]);keys=('materials','goods','delivered','energy')
 visited={tuple(initial[k] for k in keys)}
 while queue:
  state,plan=queue.popleft()
  if state['delivered']==q['goal']:return {'reachable':True,'shortest_steps':len(plan),'plan':plan,'final_state':state}
  if len(plan)>=q['max_steps']:continue
  for action in ACTIONS:
   out,error=transition(state,action,q)
   if error:continue
   key=tuple(out[k] for k in keys)
   if key in visited:continue
   visited.add(key);queue.append((out,plan+[action]))
 return {'reachable':False,'shortest_steps':None,'plan':None,'final_state':None}
BASELINE_PARAMS={'policy.W1':[[0.]*6 for _ in range(4)],'policy.b1':[0.]*4}
BASELINE_CELL={'interface':INTERFACE,'params':deepcopy(BASELINE_PARAMS),'provenance':{
 'kind':'explicit-uninformed-zero-weight-control','source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'training':'none','learning_claim':False,'formerly_working_founder':False,
 'purpose':'transparent initial control only; actual LLM must author fresh planning seed before learning claims'}}
def case(id_,goal=1,capacity=1,max_energy=5,energy=5,materials=0,goods=0,costs=(1,2,1),max_steps=12):
 return {'id':id_,'input':{'initial':{'materials':materials,'goods':goods,'delivered':0,'energy':energy},
 'goal':goal,'capacity':capacity,'max_energy':max_energy,'costs':dict(zip(('acquire','assemble','deliver'),costs)),'max_steps':max_steps}}
DEVELOPMENT_CASES=[
 case('dev-single-cycle'),
 case('dev-empty-battery',energy=0,max_energy=4,max_steps=5),
 case('dev-initial-goods',goods=1,max_steps=3),
 case('dev-double-tight-energy',goal=2,max_energy=4,energy=1,max_steps=10),
 case('dev-initial-material',materials=1,max_energy=3,energy=3,max_steps=5),
 case('dev-asymmetric-costs',capacity=2,max_energy=6,energy=2,costs=(2,3,1),max_steps=7),
 case('dev-batch-capacity',goal=2,capacity=2,max_energy=7,energy=7,max_steps=10),
 case('dev-three-deliveries',goal=3,capacity=2,max_energy=5,energy=2,costs=(1,1,2),max_steps=14),
]
FROZEN_CASES=[case('frozen-triple-asymmetric',goal=3,capacity=3,max_energy=8,energy=3,costs=(2,1,2),max_steps=16),
              case('frozen-partially-loaded',goal=2,capacity=2,max_energy=6,energy=2,materials=1,goods=1,costs=(1,3,2),max_steps=12)]
DEVELOPMENT_DECIMAL_CASES=[]
MANIFEST={'interface':INTERFACE,'development_ids':[c['id'] for c in DEVELOPMENT_CASES],
 'development_sha256':digest(DEVELOPMENT_CASES),'frozen_ids':[c['id'] for c in FROZEN_CASES],
 'frozen_sha256':digest(FROZEN_CASES),'frozen_evaluated':False,'positions':28,'baseline_kind':'uninformed_control'}
def grade(cell):
 validate(cell);traces=[];outcomes={};errors=[];completed=0
 for c in DEVELOPMENT_CASES:
  expected=reference(c['input'])
  try:out=predict(_params(cell),c['input']);error=None
  except (ValueError,TypeError,KeyError,IndexError) as exc:out=None;error=type(exc).__name__+':'+str(exc);errors.append(error)
  complete=out is not None and out['status']=='complete';completed+=complete
  hit=complete and expected['reachable'] and out['steps']==expected['shortest_steps'];outcomes[c['id']]=bool(hit)
  length_score=expected['shortest_steps']/out['steps'] if complete and out['steps'] else 0.
  traces.append({'id':c['id'],'input':c['input'],'expected':expected,'predicted':out,'correct':bool(hit),
                 'goal_complete':complete,'length_score':length_score,'error':error})
 return {'partition':'development','solved':sum(outcomes.values()),'total':len(outcomes),'completed':int(completed),
         'outcomes':outcomes,'traces':traces,'errors':errors,'learning_claim':False}
def crossover(left,right,left_positions):
 validate(left);validate(right)
 if 'params' not in left or 'params' not in right:raise ValueError('typed parents required')
 chosen=list(left_positions)
 if len(set(chosen))!=len(chosen) or any(type(i) is not int or not 0<=i<28 for i in chosen):raise ValueError('unique scalar positions0..27')
 chosen=set(chosen);lp=left['params'];rp=right['params'];different={i for i,(k,j) in enumerate(POSITION_ORDER) if np.asarray(lp[k])[j]!=np.asarray(rp[k])[j]}
 if not (chosen&different) or not (different-chosen):raise ValueError('distinguishing contribution from both parents required')
 p={k:np.asarray(rp[k],float).copy() for k in SHAPES};ids={'left':identity(left),'right':identity(right)};inheritance=[]
 for i,(key,index) in enumerate(POSITION_ORDER):
  side='left' if i in chosen else 'right';v=float(np.asarray(lp[key] if side=='left' else rp[key])[index]);p[key][index]=v
  inheritance.append({'position':i,'parameter':key,'index':list(index),'parent':side,'parent_id':ids[side],'value':v})
 out={'interface':INTERFACE,'params':{k:v.tolist() for k,v in p.items()},'parents':ids,'inheritance':inheritance,
      'charged_positions':28,'authored_positions':0,'operator':'literal-scalar-inheritance'}
 validate(out);return out
