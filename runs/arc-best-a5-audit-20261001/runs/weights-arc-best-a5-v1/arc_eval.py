"""Frozen graph topology; every numeric tensor comes from the supplied weights."""
import hashlib,json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor

ROOT=Path(__file__).resolve().parent
_predictors={}

def predict_with_template(weights,task,template):
    if any(set(pair)!={'input'} for pair in task['test']):
        raise ValueError('test outputs or metadata reached inference')
    structure=json.loads((ROOT/template).read_text(encoding='utf-8'))
    if set(weights)!=set(structure['shapes']):raise ValueError('all literal graph weights required')
    params={k:np.asarray(v,dtype=np.float64) for k,v in weights.items()}
    for key,value in params.items():
        if list(value.shape)!=structure['shapes'][key] or not np.isfinite(value).all():
            raise ValueError('literal tensor shape/finite contract')
    key=hashlib.sha256(json.dumps([structure,weights],sort_keys=True,allow_nan=False).encode()).hexdigest()
    if key not in _predictors:
        graph=Graph(structure['defs'],params,structure['entries'],structure['meta'])
        graph.validate()
        _predictors.clear() # one immutable compiled model in memory
        _predictors[key]=GridPredictor(graph,backend='compiled')
    model=_predictors[key]
    demonstrations=[{'input':p['input'],'output':p['output']} for p in task['train']]
    return [model(demonstrations,pair['input']) for pair in task['test']]

def predict(weights,task):
    return predict_with_template(weights,task,'template-final.json')
