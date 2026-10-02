# SPDX-License-Identifier: MIT-0
"""Experimental per-step pure geometry memoization; no routing or model edits."""
import hashlib
import numpy as np
from cellforge.runtime.compiled import CompiledInterpreter
from cellforge.runtime.grid import GridPredictor


class GeometryMemoInterpreter(CompiledInterpreter):
    def __init__(self,graph,**kwargs):
        self._memo={};self._eligible=set();self.memo_hits=0;self.memo_misses=0
        super().__init__(graph,**kwargs)
        ports={'input','input_height','input_width'}
        for name,d in self.g.defs.items():
            if d.get('role')!='cluster' or set(d.get('inputs',[]))!=ports:continue
            # No nested bundle, state, dispatch or adaptation is eligible.
            if all(n['use'].startswith(('op:','param:')) or self.g.defs.get(n['use'],{}).get('kind')=='cell' for n in d['nodes']):
                self._eligible.add(name)

    def run_bundle(self,name,inputs,path=''):
        if name in self.g.entries.values():self._memo.clear()
        if name not in self._eligible or self.trace_enabled:return super().run_bundle(name,inputs,path)
        def fingerprint(value):
            a=np.asarray(value)
            return (a.dtype.str,a.shape,hashlib.sha256(a.tobytes(order='C')).digest())
        key=(self._definition_signatures[name],tuple((p,fingerprint(inputs[p])) for p in sorted(self.g.defs[name]['inputs'])))
        if key in self._memo:
            output,nodes,macs,peak=self._memo[key];self.memo_hits+=1
            # Preserve semantic counters; they are not physical work counters.
            self.stats.node_execs+=nodes;self.stats.macs+=macs
            self.stats.peak_elements=max(self.stats.peak_elements,peak)
            return output
        self.memo_misses+=1;before=(self.stats.node_execs,self.stats.macs)
        result=super().run_bundle(name,inputs,path)
        saved={k:np.array(v,copy=True) for k,v in result.items()}
        for v in saved.values():v.flags.writeable=False
        self._memo[key]=(saved,self.stats.node_execs-before[0],self.stats.macs-before[1],self.stats.peak_elements)
        return result


class GeometryMemoGridPredictor(GridPredictor):
    def __init__(self,graph):
        super().__init__(graph,backend='compiled')
        self._compiled=GeometryMemoInterpreter(graph,reuse_from=self._compiled)
