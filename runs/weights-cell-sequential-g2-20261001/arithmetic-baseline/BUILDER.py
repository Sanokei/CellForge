"""Author a decimal full-adder from small, fixed affine/threshold cells."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph


def build_arithmetic():
    params = {'sum_W':np.array([[1.,1.,1.]]),'sum_b':np.array([0.]),
              'carry_W':np.array([[1.]]),'carry_b':np.array([-9.5]),
              'digit_W':np.array([[1.,-10.]]),'digit_b':np.array([0.])}
    defs = {}
    for name, act in [('sum','identity'),('carry','step'),('digit','identity')]:
        defs[name] = {'kind':'cell','act':act,'params':{'W1':name+'_W','b1':name+'_b'}}
    defs['full_adder'] = {'kind':'bundle','role':'cluster','inputs':['digits'], 'nodes':[
        {'id':'total','use':'sum','in':{'x':'$digits'}},
        {'id':'carry','use':'carry','in':{'x':'total'}},
        {'id':'pair','use':'op:concat','in':{'x0':'total','x1':'carry'},'args':{'axis':0}},
        {'id':'digit','use':'digit','in':{'x':'pair'}}],
        'outputs':{'digit':'digit','carry':'carry'}}
    defs['addition_step'] = {'kind':'bundle','role':'cluster','inputs':['a','b','carry'],'nodes':[
        {'id':'digits','use':'op:concat','in':{'x0':'$a','x1':'$b','x2':'$carry'},'args':{'axis':0}},
        {'id':'adder','use':'full_adder','in':{'digits':'digits'}}],
        'outputs':{'digit':'adder.digit','carry':'adder.carry'}}
    return Graph(defs,params,{'step':'addition_step'},{'task':'arithmetic',
        'arithmetic_interface':'decimal-addition-graph/1','max_digits':4096,
        'construction':'authored decimal full adder; no fitting or data access',
        'limitations':'unsigned decimal addition only; no language parsing or learned reasoning'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    out = Path(p.parse_args().out)
    if out.exists():
        raise SystemExit('output must be new')
    graph = build_arithmetic()
    graph.validate()
    digest = graph.save(out)
    source = Path(__file__).read_bytes()
    (out/'BUILDER.py').write_bytes(source)
    (out/'MODEL_CARD.json').write_text(json.dumps({'model_hash':digest,
        'builder_sha256':hashlib.sha256(source).hexdigest(),'training':'none',
        'numeric_provenance':'base-ten full-adder coefficients and carry threshold',
        'limitations':graph.meta['limitations']},indent=2))
    print(digest)


if __name__ == '__main__':
    main()
