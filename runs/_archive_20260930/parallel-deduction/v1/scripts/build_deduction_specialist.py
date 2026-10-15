"""Directly author bounded positive Horn closure; no fitted parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph


def build_deduction():
    defs = {
        'all_present':{'kind':'cell','act':'step','params':{'W1':'all_W','b1':'all_b'}},
        'any_present':{'kind':'cell','act':'step','params':{'W1':'any_W','b1':'any_b'}}}
    params = {'all_W':np.array([[1.,-1.]]),'all_b':np.array([0.5]),
              'any_W':np.array([[1.]]),'any_b':np.array([0.])}
    defs['rule_firing'] = {'kind':'bundle','role':'cluster',
        'inputs':['facts','antecedents','required','active'],'nodes':[
        {'id':'counts','use':'op:matmul','in':{'a':'$antecedents','b':'$facts'}},
        {'id':'pair','use':'op:concat','in':{'x0':'counts','x1':'$required'},'args':{'axis':1}},
        {'id':'ready','use':'all_present','in':{'x':'pair'}},
        {'id':'fire','use':'op:mul','in':{'a':'ready','b':'$active'}}],
        'outputs':{'fired':'fire'}}
    defs['closure_step'] = {'kind':'bundle','role':'cluster',
        'inputs':['facts','antecedents','required','active','conclusions','query'],'nodes':[
        {'id':'rules','use':'rule_firing','in':{k:'$'+k for k in ('facts','antecedents','required','active')}},
        {'id':'votes','use':'op:matmul','in':{'a':'$conclusions','b':'rules.fired'}},
        {'id':'derived','use':'any_present','in':{'x':'votes'}},
        {'id':'closure','use':'op:maximum','in':{'a':'$facts','b':'derived'}},
        {'id':'answer','use':'op:matmul','in':{'a':'$query','b':'closure'}}],
        'outputs':{'facts':'closure','fired':'rules.fired','entailed':'answer'}}
    return Graph(defs,params,{'step':'closure_step'},{'task':'deduction',
        'deduction_interface':'horn-closure-graph/1','max_atoms':32,'max_rules':64,
        'max_antecedents':4,'construction':'authored positive propositional Horn fixed point; no fitting',
        'policy':'positive ASCII identifiers only; negation unsupported; not_proven does not mean false',
        'limitations':'bounded supplied Horn rules only; no natural-language understanding or rule induction'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    out = Path(p.parse_args().out)
    if out.exists():
        raise SystemExit('output must be new')
    graph = build_deduction()
    graph.validate()
    digest = graph.save(out)
    source = Path(__file__).read_bytes()
    (out/'BUILDER.py').write_bytes(source)
    (out/'MODEL_CARD.json').write_text(json.dumps({'model_hash':digest,
        'builder_sha256':hashlib.sha256(source).hexdigest(),'training':'none',
        'numeric_provenance':'fixed presence thresholds and bounded tensor dimensions',
        'policy':graph.meta['policy'],'limitations':graph.meta['limitations']},indent=2))
    print(digest)


if __name__ == '__main__':
    main()
