"""Encode symbols and schedule graph-owned positive Horn closure."""
import re
import numpy as np
from ..ir.graph import GraphError
from .interp import Interpreter


class HornDeductionPredictor:
    def __init__(self,graph):
        graph.validate()
        if graph.meta.get('deduction_interface') != 'horn-closure-graph/1':
            raise GraphError('requires horn-closure-graph/1')
        if tuple(graph.meta.get(k) for k in ('max_atoms','max_rules','max_antecedents')) != (32,64,4):
            raise GraphError('unsupported deduction bounds')
        self.graph = graph

    def __call__(self,facts,rules,query):
        def atom(value):
            if not isinstance(value,str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,63}',value):
                raise GraphError('positive ASCII identifiers only; negation is unsupported')
            return value
        if not isinstance(facts,list) or len(facts)>32 or not isinstance(rules,list) or len(rules)>64:
            raise GraphError('facts/rules must be lists bounded by 32/64')
        names = {atom(query)}
        names.update(atom(x) for x in facts)
        for rule in rules:
            if not isinstance(rule,dict) or set(rule) != {'if','then'} or not isinstance(rule['if'],list) or len(rule['if'])>4:
                raise GraphError('rule requires if list of at most four atoms and then atom')
            names.add(atom(rule['then']))
            names.update(atom(x) for x in rule['if'])
        if len(names)>32:
            raise GraphError('at most 32 distinct atoms')
        ids = {name:i for i,name in enumerate(sorted(names))}
        state = np.zeros((1,32,1))
        for name in facts:
            state[0,ids[name],0] = 1
        antecedents,required,active = np.zeros((1,64,32)),np.zeros((1,64,1)),np.zeros((1,64,1))
        conclusions = np.zeros((1,32,64))
        for i,rule in enumerate(rules):
            unique = set(rule['if'])
            for name in unique:
                antecedents[0,i,ids[name]] = 1
            required[0,i,0] = len(unique)
            active[0,i,0] = 1
            conclusions[0,ids[rule['then']],i] = 1
        query_tensor = np.zeros((1,1,32))
        query_tensor[0,0,ids[query]] = 1
        engine = Interpreter(self.graph)
        provenance = []
        seen = set()
        for iteration in range(33):
            out = engine.run_bundle(self.graph.entries['step'],{'facts':state,
                'antecedents':antecedents,'required':required,'active':active,
                'conclusions':conclusions,'query':query_tensor},'/deduction')
            next_state,fired,answer = (np.asarray(out[k]) for k in ('facts','fired','entailed'))
            for value,shape in ((next_state,(1,32,1)),(fired,(1,64,1)),(answer,(1,1,1))):
                if value.shape != shape or not np.isfinite(value).all() or not np.isin(value,[0,1]).all():
                    raise GraphError('invalid deduction output')
            for i in np.flatnonzero(fired[0,:,0]):
                if int(i) not in seen:
                    provenance.append({'rule_index':int(i),'round':iteration+1})
                    seen.add(int(i))
            if np.array_equal(state,next_state):
                break
            state = next_state
        else:
            raise GraphError('graph closure failed to stabilize')
        entailed = bool(answer[0,0,0])
        return {'status':'entailed' if entailed else 'not_proven','entailed':entailed,
            'rounds':iteration+1,'graph_hash':self.graph.hash(),'fired_rules':provenance,
            'policy':self.graph.meta['policy'],'execution':engine.stats.as_dict()}
