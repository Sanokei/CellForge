"""Exact teacher-forced byte fitness with certified recurrent-prefix reuse.

64-byte windows are execution batches, never context resets. No corpus fitting,
answer construction, loss pruning or changed-state reuse occurs here.
"""
import hashlib
import json
import time
import numpy as np
from ..ir.graph import GraphError, parse_ref, array_hash
from ..runtime.compiled import CompiledInterpreter
from ..runtime.lm import encode_text, _log2_softmax
from ..runtime.text_codec import BYTE_CODEC, BYTE_VOCAB


def _check(graph):
    if graph.meta.get('text_codec') != BYTE_CODEC or graph.meta.get('vocab') != BYTE_VOCAB:
        raise GraphError('canonical UTF8 byte graph required')


def _state_certificate(graph):
    """Conservative root dependency closure, entire nested definitions included."""
    definition = graph.defs[graph.entries['step']]
    nodes = {n['id']: n for n in definition['nodes']}
    required, defs, params = set(), set(), set()

    def visit_def(name):
        if name in defs: return
        defs.add(name)
        params.update(graph.params_of(name))
        for child in graph.children(name): visit_def(child)

    def visit_ref(ref):
        kind, target, _ = parse_ref(ref)
        if kind != 'node' or target in required: return
        required.add(target)
        node = nodes[target]
        use = node['use']
        if use.startswith('param:'): params.add(use[6:])
        elif use == 'dispatch':
            for branch in node['args']['branches']: visit_def(branch)
        elif not use.startswith('op:'): visit_def(use)
        for value in node.get('in', {}).values(): visit_ref(value)

    visit_ref(definition.get('outputs', {}).get('state', '$state'))
    payload = {'entry': graph.entries['step'],
               'meta': {k: graph.meta.get(k) for k in ('task','text_codec','vocab','ctx_len','state_dim')},
               'inputs': definition.get('inputs', []),
               'state_output': definition.get('outputs', {}).get('state', '$state'),
               'nodes': [n for n in definition['nodes'] if n['id'] in required],
               'defs': {k: graph.defs[k] for k in sorted(defs)},
               'params': {k: array_hash(graph.params[k]) for k in sorted(params)}}
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return digest, required


def _batch_independent(graph):
    # Negative axes can alias the batch axis at a particular rank. Reject all
    # such explicit/default reductions conservatively until rank-certified.
    allowed = {'patchify','reshape','transpose','add','sub','mul','maximum','scale','bias',
               'relu','tanh','sigmoid','abs','square','neg','exp','step','sum','mean','max','min',
               'cumsum','cummax','softmax','matmul','concat','slice','pad','shift','coords',
               'onehot_argmax','expand','broadcast_to','csr_lookup','log','divide','power'}
    for definition in graph.defs.values():
        for node in definition.get('nodes', []):
            if node['use'].startswith('op:'):
                if node['use'][3:] not in allowed: return False
                args=node.get('args', {})
                default=-1 if node['use'] in ('op:softmax','op:onehot_argmax') else 0
                if int(args.get('axis', default)) < 0: return False
    return True


def _inputs(ctx, state, current, position):
    ctx = np.roll(ctx, -1, axis=1)
    ctx[:, -1] = 0.
    ctx[np.arange(len(current)), -1, current] = 1.
    return {'ctx':ctx, 'prev':ctx[:, -1], 'state':state,
            'step':np.full((len(current),1),float(position))}


def _selection_masks(items):
    masks=[]
    for item in items:
        text=item.get('text')
        if not isinstance(text,str) or not text:raise ValueError('nonempty next-byte text required')
        n=len(text.encode('utf8'));mask=np.ones(n,dtype=bool)
        if 'score_byte_ranges' in item:
            ranges=item['score_byte_ranges'];mask[:]=False;end=0
            if not isinstance(ranges,list) or not ranges:raise ValueError('nonempty score byte ranges required')
            for interval in ranges:
                if not isinstance(interval,list) or len(interval)!=2 or any(type(v) is not int for v in interval):
                    raise ValueError('integer [start,stop] byte intervals required')
                start,stop=interval
                if not 0<=start<stop<=n or start<end:raise ValueError('sorted nonoverlapping bounded byte intervals required')
                mask[start:stop]=True;end=stop
        masks.append(mask)
    return masks


def _result(graph, items, bits, failed, elapsed, startup=0., cache_mode='compiled_full', evaluated=None):
    lengths = np.array([int(mask.sum()) for mask in _selection_masks(items)])
    penalty = max(8., float(np.log2(len(BYTE_VOCAB))+1))
    charged = np.where(failed, penalty*lengths, bits)
    total = int(lengths.sum())
    return {'fitness_bits_per_byte':float(charged.sum()/total),
            'eligible_for_selection':not bool(failed.any()),
            'next_byte':{'content_bits':float(charged.sum()),'content_bytes':total,
                         'failed_documents':int(failed.sum()),'eos_excluded':True,
                         'per_document_content_bits':charged.tolist(),
                         'selection_unit':'selected UTF8 content bytes',
                         'full_content_bytes':sum(len(item['text'].encode('utf8')) for item in items)},
            'timing':{'startup_seconds':startup,'evaluation_seconds':elapsed,
                      'evaluations_per_second':1/elapsed if elapsed else None,
                      'evaluated_bytes':int(evaluated if evaluated is not None else total+len(items))},
            'cache_mode':cache_mode,'model_hash':graph.hash()}


def compiled_content_fitness(graph, items, *, interpreter=None):
    """Sequential compiled fallback; same EOS failure check as reference."""
    _check(graph)
    if not items or any(not isinstance(i.get('text'),str) or not i['text'] for i in items):
        raise ValueError('nonempty next-byte texts required')
    started=time.perf_counter()
    it=interpreter or CompiledInterpreter(graph)
    startup=time.perf_counter()-started
    seqs=[encode_text(BYTE_VOCAB,i['text'],BYTE_CODEC) for i in items]
    masks=_selection_masks(items)
    B,V=len(items),len(BYTE_VOCAB);S=int(graph.meta.get('state_dim',0));K=int(graph.meta.get('ctx_len',8))
    ctx=np.zeros((B,K,V));state=np.zeros((B,S));bits=np.zeros(B);failed=np.zeros(B,dtype=bool)
    tick=time.perf_counter();count=0
    for t in range(max(map(len,seqs))-1):
        rows=np.array([j for j,s in enumerate(seqs) if t<len(s)-1],dtype=int)
        inp=_inputs(ctx[rows],state[rows],np.array([seqs[j][t] for j in rows]),t)
        ctx[rows]=inp['ctx'];out=it.run_bundle(graph.entries['step'],inp,'/step')
        lg=np.asarray(out['logits']).reshape(len(rows),V)
        ns=np.asarray(out.get('state',np.zeros((len(rows),S)))).reshape(len(rows),S)
        failed[rows]|=~np.isfinite(lg).all(axis=1)|~np.isfinite(ns).all(axis=1)
        lp=_log2_softmax(np.where(np.isfinite(lg),lg,0.))
        for r,j in enumerate(rows):
            if t<len(masks[j]) and masks[j][t]:bits[j]+=-lp[r,seqs[j][t+1]]
        state[rows]=np.where(np.isfinite(ns),ns,0.);count+=len(rows)
    result=_result(graph,items,bits,failed,time.perf_counter()-tick,startup,evaluated=count)
    result['timing']['end_to_end_seconds']=time.perf_counter()-started
    result['timing']['end_to_end_evaluations_per_second']=1/result['timing']['end_to_end_seconds']
    return result


class EvolutionFitness:
    """Own a frozen curriculum and verified founder state-prefix cache.

    Changed state/read/write/context structure or values always falls back.
    Even supported candidates execute all logits, including every expert
    selected by their own router. Cached values are state ancestors only.
    """
    def __init__(self, founder, items, window_bytes=64):
        _check(founder)
        if type(window_bytes) is not int or not 1<=window_bytes<=64:raise ValueError('window <=64 required')
        if not items or any(not i.get('text') or i.get('kind','next_byte')!='next_byte' for i in items):
            raise ValueError('nonempty next-byte items required')
        self.items=json.loads(json.dumps(items));self.founder=founder.copy();self.window_bytes=window_bytes
        masks=_selection_masks(self.items)
        self.certificate,self.cached_nodes=_state_certificate(founder)
        started=time.perf_counter()
        it=CompiledInterpreter(founder,trace=True,trace_filter=lambda path:path.count('/')==2)
        self.records=[];self.prefix_failed=np.zeros(len(items),dtype=bool)
        V=len(BYTE_VOCAB);S=int(founder.meta.get('state_dim',0));K=int(founder.meta.get('ctx_len',8))
        for j,item in enumerate(self.items):
            seq=encode_text(BYTE_VOCAB,item['text'],BYTE_CODEC)
            ctx=np.zeros((1,K,V));state=np.zeros((1,S))
            for t in range(len(seq)-1):
                inp=_inputs(ctx,state,np.array([seq[t]]),t);ctx=inp['ctx']
                it.trace.clear()
                out=it.run_bundle_outputs(founder.entries['step'],inp,('state',),'/step') if 'state' in founder.defs[founder.entries['step']]['outputs'] else {'state':state}
                ns=np.asarray(out['state']).reshape(1,S)
                self.prefix_failed[j]|=not np.isfinite(ns).all()
                cached={}
                for node in self.cached_nodes:
                    for port in it._ports(next(n for n in founder.defs[founder.entries['step']]['nodes'] if n['id']==node)):
                        cached[(node,port)]=it.trace['/step/'+node+'.'+port].copy()
                selected=t<len(masks[j]) and bool(masks[j][t])
                self.records.append((j,t,seq[t+1],selected,{k:v.copy() for k,v in inp.items()},cached))
                state=np.where(np.isfinite(ns),ns,0.)
        self.startup_seconds=time.perf_counter()-started
        self._template=it
        self._plans={}  # bounded to the most recent candidate, not population

    def evaluate(self,candidate):
        boundary_started=time.perf_counter()
        _check(candidate)
        digest,nodes=_state_certificate(candidate)
        if digest!=self.certificate or nodes!=self.cached_nodes or not _batch_independent(candidate) or self.prefix_failed.any():
            result=compiled_content_fitness(candidate,self.items)
            result['cache_mode']='compiled_full_unsafe_prefix_fallback'
            result['timing']['end_to_end_seconds']=time.perf_counter()-boundary_started
            result['timing']['end_to_end_evaluations_per_second']=1/result['timing']['end_to_end_seconds']
            return result
        started=time.perf_counter();key=candidate.hash()
        cold=key not in self._plans
        if cold:
            it=CompiledInterpreter(candidate,reuse_from=self._template)
            fn,bindings=it.cached_node_plan(candidate.entries['step'],self.cached_nodes)
            it.guard_batched_matmul()
            self._plans={key:(it,fn,bindings)}
        it,fn,bindings=self._plans[key];startup=time.perf_counter()-started
        bits=np.zeros(len(self.items));failed=np.zeros(len(self.items),dtype=bool);tick=time.perf_counter()
        for start in range(0,len(self.records),self.window_bytes):
            records=self.records[start:start+self.window_bytes]
            inp={k:np.concatenate([record[4][k] for record in records]) for k in ('ctx','prev','state','step')}
            for nodeport,binding in bindings.items():inp[binding]=np.concatenate([record[5][nodeport] for record in records])
            try:
                out=fn(it,inp,'/step')
            except (GraphError,ValueError,IndexError):
                # A batch-shape ambiguity must never become a reused score.
                result=compiled_content_fitness(candidate,self.items)
                result['cache_mode']='compiled_full_batch_semantics_fallback'
                result['timing']['end_to_end_seconds']=time.perf_counter()-boundary_started
                result['timing']['end_to_end_evaluations_per_second']=1/result['timing']['end_to_end_seconds']
                return result
            lg=np.asarray(out['logits']).reshape(len(records),len(BYTE_VOCAB))
            ns=np.asarray(out.get('state',np.zeros((len(records),int(candidate.meta.get('state_dim',0))))))
            bad=~np.isfinite(lg).all(axis=1)|~np.isfinite(ns).all(axis=1)
            lp=_log2_softmax(np.where(np.isfinite(lg),lg,0.))
            for r,(j,t,target,content,_,_) in enumerate(records):
                failed[j]|=bad[r]
                if content:bits[j]+=-lp[r,target]
        result=_result(candidate,self.items,bits,failed,time.perf_counter()-tick,startup,
                       'state_ancestor_cache_cold' if cold else 'state_ancestor_cache_warm',len(self.records))
        result['prefix_cache_startup_seconds']=self.startup_seconds
        result['timing']['end_to_end_seconds']=time.perf_counter()-boundary_started
        result['timing']['end_to_end_evaluations_per_second']=1/result['timing']['end_to_end_seconds']
        result['plan_reuse']={'reused_bundles':it.reused_bundle_plans,'compiled_bundles':it.compiled_bundle_plans}
        return result
