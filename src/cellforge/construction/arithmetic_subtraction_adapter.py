"""Isolated development-only subtraction adapter; no training or frozen scoring."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import numpy as np
try:
 _actual=importlib.import_module('cellforge.construction.arithmetic_cell')
 SOURCE=Path(_actual.__file__).resolve()
except ModuleNotFoundError as error:
 if error.name not in ('cellforge','cellforge.construction','cellforge.construction.arithmetic_cell'):raise
 SOURCE=Path(r'C:\Users\wkeif\Documents\GitHub\CellForge\src\cellforge\construction\arithmetic_cell.py')
 _spec=importlib.util.spec_from_file_location('subtraction_actual_arithmetic_cell',SOURCE)
 _actual=importlib.util.module_from_spec(_spec)
 _spec.loader.exec_module(_actual)
BUILDER_SOURCE=SOURCE.parents[3]/'scripts'/'build_arithmetic_specialist.py'
SHAPES=dict(_actual.SHAPES)
INTERFACE='decimal-subtraction-borrow-cell/1'
POSITION_ORDER=tuple((key,index) for key,shape in SHAPES.items() for index in np.ndindex(shape))
CONTRACT={'interface':INTERFACE,'structure':'affine-total/scalar-step/affine-readout',
 'input_order':['a','b','borrow'],'input_bounds':{'a':[0,9],'b':[0,9],'borrow':[0,1]},
 'output_order':['digit','borrow'],'output_bounds':{'digit':[0,9],'borrow':[0,1]},
 'shapes':{k:list(v) for k,v in SHAPES.items()},'positions':12,'weight_bounds':[-10,10],
 'semantics':'d=a-b-borrow_in; digit=d%10; borrow_out=int(d<0)',
 'third_input_rebinding':'historical carry input explicitly receives borrow',
 'readout_rebinding':'historical carry output explicitly named borrow'}
def digest(v):
 return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def identity(cell):
 p=_params(cell)
 return digest({'interface':INTERFACE,'params':{k:np.asarray(p[k],float).tolist() for k in SHAPES}})
def _params(cell):
 if not isinstance(cell,dict):raise ValueError('cell/params dictionary required')
 if 'params' in cell:
  if cell.get('interface')!=INTERFACE:raise ValueError('subtraction interface mismatch')
  return cell['params']
 return cell
def validate(cell):
 p=_params(cell)
 if not isinstance(p,dict) or set(p)!=set(SHAPES):raise ValueError('complete actual parameter keys required')
 for key,shape in SHAPES.items():
  try:
   raw=np.asarray(p[key],dtype=object)
   if any(type(v) not in (int,float) for v in raw.flat):raise ValueError('JSON numerical literals required')
   a=np.asarray(p[key],float)
  except (TypeError,ValueError) as e:raise ValueError('numeric literals required') from e
  if a.shape!=shape or not np.isfinite(a).all() or np.any(np.abs(a)>10):raise ValueError('finite declared shapes/bounds required')
 return len(POSITION_ORDER)
def _input(q):
 if not isinstance(q,dict) or set(q)!={'a','b','borrow'}:raise ValueError('typed a,b,borrow input only')
 if any(type(q[k]) is not int or not 0<=q[k]<=limit for k,limit in [('a',9),('b',9),('borrow',1)]):raise ValueError('digit/borrow bounds')
def predict(params,request):
 """Actual arithmetic execution, only explicitly declared port renaming."""
 validate(params);_input(request)
 out=_actual.predict_adder(_params(params),{'a':request['a'],'b':request['b'],'carry':request['borrow']})
 return {'digit':out['digit'],'borrow':out['carry']}
def reference(request):
 """Development label oracle; never enters candidate execution."""
 _input(request);d=request['a']-request['b']-request['borrow']
 return {'digit':d%10,'borrow':int(d<0)}
def _decimal(x):
 if not isinstance(x,str) or not 1<=len(x)<=4096 or any(c not in '0123456789' for c in x):raise ValueError('unsigned ASCII decimals 1..4096 digits')
def predict_decimal(params,a,b):
 """Width-limited residue plus final underflow borrow, not signed output.
 Candidate digits execute actual cells; no reference subtraction enters here.
 """
 validate(params);_decimal(a);_decimal(b)
 borrow=0;digits=[];trace=[]
 for i in range(max(len(a),len(b))):
  q={'a':int(a[-1-i]) if i<len(a) else 0,'b':int(b[-1-i]) if i<len(b) else 0,'borrow':borrow}
  out=predict(params,q);trace.append({'column':i,'input':q,'output':out})
  digits.append(str(out['digit']));borrow=out['borrow']
 return {'digits':''.join(reversed(digits)),'borrow':borrow,'status':'underflow' if borrow else 'ok','trace':trace}
BASELINE_PARAMS={'sum.W':[[1.,1.,1.]],'sum.b':[0.],'carry.W':[[1.]],'carry.b':[-9.5],
 'readout.W':[[1.,-10.],[0.,1.]],'readout.b':[0.,0.]}
BASELINE_CELL={'interface':INTERFACE,'params':deepcopy(BASELINE_PARAMS),'provenance':{
 'kind':'historical-addition-baseline-on-incompatible-task','source':str(SOURCE),'builder_source':str(BUILDER_SOURCE),
 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'builder_sha256':hashlib.sha256(BUILDER_SOURCE.read_bytes()).hexdigest() if BUILDER_SOURCE.exists() else None,
 'lift':'builder digit_W/digit_b -> first readout row; [0,1] second row passes historical carry',
 'training':'none','learning_claim':False}}
# Declare before scoring; frozen cases are never evaluated by this adapter.
DEVELOPMENT_CASES=[{'id':name,'input':{'a':a,'b':b,'borrow':c}} for name,a,b,c in [
 ('dev-zero',0,0,0),('dev-no-borrow',7,2,0),('dev-new-borrow',2,7,0),
 ('dev-propagate',0,0,1),('dev-clear-borrow',8,3,1),('dev-equal-incoming',4,4,1)]]
DEVELOPMENT_DECIMAL_CASES=[
 {'id':'dev-chain','a':'1000','b':'1','expected':{'digits':'0999','borrow':0}},
 {'id':'dev-leading-zero','a':'0070','b':'0025','expected':{'digits':'0045','borrow':0}},
 {'id':'dev-underflow','a':'12','b':'19','expected':{'digits':'93','borrow':1}}]
FROZEN_CASES=[{'id':'frozen-digit-1','input':{'a':9,'b':6,'borrow':1}},
 {'id':'frozen-digit-2','input':{'a':1,'b':8,'borrow':0}}]
FROZEN_DECIMAL_CASES=[{'id':'frozen-chain','a':'20000','b':'17'},
 {'id':'frozen-underflow','a':'99','b':'100'}]
def baseline_development():
 """Measure authentic historical addition baseline on development only."""
 traces=[]
 for case in DEVELOPMENT_CASES:
  expected=reference(case['input']);predicted=predict(BASELINE_PARAMS,case['input'])
  traces.append(dict(case,expected=expected,predicted=predicted,correct=predicted==expected))
 return {'partition':'development','solved':sum(t['correct'] for t in traces),'total':len(traces),'traces':traces,'learning_claim':False}
def crossover(left,right,left_positions):
 """Exact scalar inheritance; both parents must contribute differing positions.
 Indices refer to POSITION_ORDER. No averaging or new authored values.
 """
 validate(left);validate(right)
 if 'params' not in left or 'params' not in right:raise ValueError('explicit typed parent cells required')
 chosen=list(left_positions)
 if any(type(i) is not int or not 0<=i<12 for i in chosen) or len(set(chosen))!=len(chosen):raise ValueError('unique scalar indices required')
 chosen=set(chosen);lp=left['params'];rp=right['params']
 differing={i for i,(k,j) in enumerate(POSITION_ORDER) if np.asarray(lp[k])[j]!=np.asarray(rp[k])[j]}
 if not (differing&chosen) or not (differing-chosen):raise ValueError('distinct numerical contribution from both parents required')
 child={k:np.asarray(rp[k],float).copy() for k in SHAPES}
 ids={'left':identity(left),'right':identity(right)}
 inheritance=[]
 for i,(key,index) in enumerate(POSITION_ORDER):
  side='left' if i in chosen else 'right';v=float(np.asarray(lp[key] if side=='left' else rp[key])[index]);child[key][index]=v
  inheritance.append({'position':i,'parameter':key,'index':list(index),'parent':side,'parent_id':ids[side],'value':v})
 out={'interface':INTERFACE,'params':{k:v.tolist() for k,v in child.items()},'parents':ids,'inheritance':inheritance,
 'charged_positions':12,'authored_positions':0,'operator':'literal-scalar-inheritance'}
 validate(out);return out
