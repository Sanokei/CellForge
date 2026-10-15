"""Literal request-only tool/action router. No filesystem or tool execution."""
import copy, math, re
import numpy as np
from cellforge.construction.joint_weight_population import digest

ACTIONS=('grep_todo','grep_import','grep_def','glob_py','script_count_lines','read_file','none','write_file','edit_in_place')
FEATURES=('find','search','mentions','todo','import','def','list','files','python','count','lines','run','how','many','show','open','read','contents','print','say','display','file','has_filename','write','create','edit','replace','append')

def validate(tensors):
    if set(tensors)!={'W','b'}:raise ValueError('literal router W and b required; no slots')
    def numeric(v):
        return all(numeric(x) for x in v) if isinstance(v,list) else type(v) in (int,float) and math.isfinite(v)
    if not all(numeric(v) for v in tensors.values()):raise ValueError('finite literal numerical tensors required')
    w,b=np.asarray(tensors['W']),np.asarray(tensors['b'])
    if w.shape!=(9,28) or b.shape!=(9,):raise ValueError('fixed action/feature contract requires W9x28,b9')
    return copy.deepcopy(tensors)

def features(request):
    if type(request) is not str:raise ValueError('request text only')
    words=set(re.findall(r'[a-z]+',request.lower()))
    v=[int(word in words) for word in FEATURES]
    v[22]=int(bool(re.search(r'(?<![\w./\\])[A-Za-z0-9_-]+\.(?:py|txt)(?![\w.])',request)))
    return v

def route(tensors, request):
    t=validate(tensors);v=features(request)
    scores=np.asarray(t['W'])@np.asarray(v)+np.asarray(t['b'])
    index=int(np.argmax(scores))
    return {'action':ACTIONS[index],'features':v,'scores':scores.tolist(),'row':index}

def inherited_parent(baseline):
    if baseline['actions']!=list(ACTIONS[:7]) or baseline['vocab']!=list(FEATURES[:18]):raise ValueError('baseline interface mismatch')
    w=np.zeros((9,28));b=np.zeros(9)
    w[:7,:18]=np.asarray(baseline['W']);b[:7]=baseline['b']
    return validate({'W':w.tolist(),'b':b.tolist()})

def safe_path(path):
    return type(path) is str and re.fullmatch(r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*\.(?:txt|py)',path) is not None

def action_arguments(action,args):
    if type(args) is not dict:raise ValueError('explicit typed action arguments required')
    if action=='write_file':
        if set(args)!={'path','content'} or not safe_path(args['path']) or type(args['content']) is not str:
            raise ValueError('safe relative fixture path and literal text required')
    elif action=='edit_in_place':
        if set(args)!={'path','old','new','expected_matches'} or not safe_path(args['path']) or type(args['old']) is not str or not args['old'] or type(args['new']) is not str or type(args['expected_matches']) is not int or args['expected_matches']<=0:
            raise ValueError('bounded exact replacement arguments required')
    else:raise ValueError('only new typed mutation action arguments handled here')
    return copy.deepcopy(args)

def route_action(tensors,request,args):
    raw=route(tensors,request)
    if raw['action'] in ('write_file','edit_in_place'):
        try:return dict(raw,arguments=action_arguments(raw['action'],args),raw_action=raw['action'],arguments_valid=True)
        except ValueError as error:return dict(raw,action='none',raw_action=raw['action'],arguments_valid=False,reason=str(error))
    return dict(raw,raw_action=raw['action'])

def score(tensors,rows):
    traces=[];outcomes={};negative=wrong_negative=0
    for row in rows:
        trace=route(tensors,row['request']);hit=trace['action']==row['expected']
        outcomes[row['id']]=hit;negative+=int(row['expected']=='none');wrong_negative+=int(row['expected']=='none' and trace['action']!='none')
        traces.append(dict(row,predicted=trace['action'],features=trace['features'],scores=trace['scores'],correct=hit))
    return {'manifest':digest(rows),'outcomes':outcomes,'solves':sum(outcomes.values()),'tasks':len(rows),'negative_count':negative,'false_action_count':wrong_negative,'traces':traces}

def admission(parent,candidate):
    reasons=[]
    for label in ('fresh','archive','anchor'):
        a,b=parent[label],candidate[label]
        if a['manifest']!=b['manifest'] or a['outcomes'].keys()!=b['outcomes'].keys():raise ValueError('matched frozen phrase contract required')
        losses=[k for k in a['outcomes'] if a['outcomes'][k] and not b['outcomes'][k]]
        if losses:reasons.append({'gate':label+'_regression','phrases':losses})
        if b['false_action_count']>a['false_action_count']:reasons.append({'gate':label+'_false_action'})
    if candidate['fresh']['solves']<=parent['fresh']['solves']:reasons.append({'gate':'strict_fresh_gain'})
    return {'accepted':not reasons,'reasons':reasons,'scope':'bounded frozen request classification; no action-cell competency or ARC claim'}



def route_checked_action(tensors, request, args, *, allowed_root):
    """Adapt numerical action to the actual glue transport; validate without executing."""
    from cellforge.construction.action_packet import validate_file_action
    result=route_action(tensors,request,args)
    if result['action'] not in ('write_file','edit_in_place'):
        return result
    packet={'type':'call','name':result['action'],'arguments':result['arguments']}
    try:
        checked=validate_file_action(packet,allowed_root=allowed_root)
        return dict(result,packet=packet,checked=checked,validation_scope='installed glue, explicit fixture root; no execution')
    except (ValueError,OSError) as error:
        return dict(result,action='none',arguments_valid=False,reason=str(error),validation_scope='installed glue rejected packet')
