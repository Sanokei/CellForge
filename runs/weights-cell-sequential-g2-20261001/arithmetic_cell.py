"""Literal twelve-parameter decimal arithmetic cell, no language parser."""
import numpy as np
SHAPES={'sum.W':(1,3),'sum.b':(1,),'carry.W':(1,1),'carry.b':(1,),'readout.W':(2,2),'readout.b':(2,)}
def predict_adder(params,request):
    if set(request)!={'a','b','carry'}:raise ValueError('typed arithmetic input only')
    if any(type(request[k]) is not int or not 0<=request[k]<=limit for k,limit in [('a',9),('b',9),('carry',1)]):
        raise ValueError('decimal digit/carry bounds')
    if set(params)!=set(SHAPES):raise ValueError('complete literal weights required')
    for key,shape in SHAPES.items():
        a=np.asarray(params[key],float)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('finite declared arithmetic shapes')
    s=float(np.asarray(params['sum.W'])[0]@np.asarray([request['a'],request['b'],request['carry']])+params['sum.b'][0])
    c=float(params['carry.W'][0][0]*s+params['carry.b'][0]>0)
    y=np.asarray(params['readout.W'])@np.asarray([s,c])+np.asarray(params['readout.b'])
    if np.any(y!=np.floor(y)) or not 0<=y[0]<=9 or y[1] not in (0,1):raise ValueError('invalid digit/carry output')
    return {'digit':int(y[0]),'carry':int(y[1])}

