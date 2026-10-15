"""Literal joint genomes over the existing weights-pop1 predictors.

Demonstration error features contain no query targets or niche identifiers.
Cross-structure inheritance is an explicit eight-neighbor weight broadcast.
"""
import copy,hashlib,importlib.util,json
from pathlib import Path
import numpy as np

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def load_core(path):
    spec=importlib.util.spec_from_file_location('pop1_frozen_core',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def experts(tensors,core):
    groups={}
    for name,value in tensors.items():
        if name.startswith('route.'):continue
        index,key=name.split(':',1)
        groups.setdefault(index,{})[key]=value
    result=[]
    for index,params in sorted(groups.items()):
        template=next((t for t,shapes in core.SHAPES.items() if set(shapes)==set(params)
                       and all(np.asarray(params[k]).shape==shape for k,shape in shapes.items())),None)
        if template is None:raise ValueError('unknown or malformed inherited expert')
        result.append((index,template,params))
    if not result:raise ValueError('nonempty joint cells required')
    return result

def expert_predictions(tensors,grid,core):
    values=[]
    for _,template,params in experts(tensors,core):
        try:values.append(core.PREDICTORS[template](params,grid,False)[0])
        except (ValueError,IndexError,KeyError,TypeError):values.append(None)
    return values

def router_features(tensors,view,core):
    if set(view)!={'demos','input'} or not view['demos']:
        raise ValueError('routing view permits demonstrations and query input only')
    n=len(experts(tensors,core));error=np.zeros(n)
    for demo in view['demos']:
        if set(demo)!={'input','output'}:raise ValueError('complete demonstrations required')
        expected=np.asarray(demo['output'])
        for i,pred in enumerate(expert_predictions(tensors,demo['input'],core)):
            observed=np.asarray(pred) if pred is not None else None
            error[i]+=1. if observed is None or observed.shape!=expected.shape else float(np.mean(observed!=expected))
    return -error/len(view['demos'])

def predict_routed(tensors,router,view,core,trace=False):
    f=router_features(tensors,view,core);w=np.asarray(router['route.W']);b=np.asarray(router['route.b'])
    if w.shape!=(len(f),len(f)) or b.shape!=(len(f),) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError('finite aligned literal router tensors required')
    scores=w@f+b;index=int(np.argmax(scores))
    value=expert_predictions(tensors,view['input'],core)[index]
    return (value,{'features':f.tolist(),'scores':scores.tolist(),'chosen_expert':index}) if trace else value

def uniform_router(router):
    return {k:np.zeros_like(np.asarray(v)).tolist() for k,v in router.items()}

def inherit_neighbors(recipient,donor,*,recipient_prefix,donor_prefix,router_row=None):
    """Count -> conv broadcast exactly matches eight-neighbor count semantics.

    Center detector and output paint remain from recipient. No averaging of
    unlike shapes. Conv -> count requires identical eight directional rows.
    """
    child=copy.deepcopy(recipient);r=recipient_prefix;d=donor_prefix
    if d+'cnt.W' in donor and r+'conv.W' in recipient:
        row=np.asarray(donor[d+'cnt.W'])
        w=np.asarray(recipient[r+'conv.W']).copy()
        if row.shape!=(1,4) or w.shape!=(9,4):raise ValueError('four-color neighborhood contract required')
        for i in range(9):
            if i!=4:w[i]=row[0]
        child[r+'conv.W']=w.tolist();child[r+'conv.b']=copy.deepcopy(donor[d+'cnt.b'])
        mapping={'donor':d+'cnt.W','recipient':r+'conv.W','broadcast_rows':[0,1,2,3,5,6,7,8]}
        retained=[r+'conv.W[4]',r+'paint']
    elif d+'conv.W' in donor and r+'cnt.W' in recipient:
        w=np.asarray(donor[d+'conv.W']);rows=w[[0,1,2,3,5,6,7,8]]
        if w.shape!=(9,4) or not np.array_equal(rows,np.broadcast_to(rows[0],rows.shape)):
            raise ValueError('directional convolution cannot become isotropic count without loss; no arbitrary averaging')
        child[r+'cnt.W']=[rows[0].tolist()];child[r+'cnt.b']=copy.deepcopy(donor[d+'conv.b'])
        mapping={'donor':d+'conv.W','recipient':r+'cnt.W','exact_identical_rows':True}
        retained=[r+'self.W',r+'paint']
    elif d+'conv.W' in donor and r+'conv.W' in recipient:
        w=np.asarray(recipient[r+'conv.W']).copy();dw=np.asarray(donor[d+'conv.W'])
        if w.shape!=dw.shape or w.shape!=(9,4):raise ValueError('homologous convolution contract required')
        for i in range(9):
            if i!=4:w[i]=dw[i]
        child[r+'conv.W']=w.tolist();child[r+'conv.b']=copy.deepcopy(donor[d+'conv.b'])
        mapping={'donor':d+'conv.W','recipient':r+'conv.W','rows':[0,1,2,3,5,6,7,8]}
        retained=[r+'conv.W[4]',r+'paint']
    else:raise ValueError('unsupported structural inheritance contract')
    if router_row is not None:
        child['route.W'][router_row]=copy.deepcopy(donor['route.W'][router_row])
        child['route.b'][router_row]=donor['route.b'][router_row]
    changes={k:int(np.count_nonzero(np.asarray(child[k])!=np.asarray(recipient[k]))) for k in child}
    receipt={'recipient_hash':digest(recipient),'donor_hash':digest(donor),'child_hash':digest(child),
        'mapping':mapping,'recipient_retained':retained,'router_row_from_donor':router_row,
        'changed_scalars':{k:v for k,v in changes.items() if v},
        'limitations':'four-color grid cells only; inherited detection features, not guaranteed phenotype equivalence'}
    return child,receipt
