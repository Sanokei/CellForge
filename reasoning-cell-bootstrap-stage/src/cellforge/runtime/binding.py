"""Bounds checking and byte packing for graph-owned exact binding retrieval."""
import numpy as np
from ..ir.graph import GraphError
from .interp import Interpreter


class BindingPredictor:
    def __init__(self, graph):
        graph.validate()
        if graph.meta.get('binding_interface') != 'binding-graph/1':
            raise GraphError('requires binding-graph/1')
        if (graph.meta.get('max_bindings'),graph.meta.get('key_bytes'),graph.meta.get('value_bytes')) != (8,16,32):
            raise GraphError('unsupported binding bounds')
        self.graph = graph

    @staticmethod
    def _pack(text, length):
        if not isinstance(text,str):
            raise GraphError('bindings must be strings')
        raw = text.encode('utf-8')
        if not raw or len(raw) >= length:
            raise GraphError(f'nonempty text must fit {length-1} UTF-8 bytes')
        encoded = np.zeros((length,257))
        encoded[:,256] = 1
        for i,b in enumerate(raw):
            encoded[i,256] = 0
            encoded[i,b] = 1
        return encoded

    def __call__(self, bindings, query):
        if not isinstance(bindings,list) or len(bindings)>8:
            raise GraphError('bindings must be a list of at most eight entries')
        keys,values = np.zeros((1,8,16,257)),np.zeros((1,8,32,257))
        active = np.zeros((1,8))
        for i,item in enumerate(bindings):
            if not isinstance(item,dict) or set(item) != {'key','value'}:
                raise GraphError('each binding requires key and value')
            keys[0,i] = self._pack(item['key'],16)
            values[0,i] = self._pack(item['value'],32)
            active[0,i] = 1
        out = Interpreter(self.graph).run_bundle(self.graph.entries['predict'],
              {'keys':keys,'values':values,'query':self._pack(query,16)[None],'active':active},'/binding')
        logits,count = np.asarray(out['value_logits']),np.asarray(out['matches'])
        if logits.shape != (1,32,257) or count.shape != (1,1) or not np.isfinite(logits).all() or not np.isfinite(count).all():
            raise GraphError('invalid binding output')
        matches = int(count[0,0])
        if count[0,0] != matches or not 0<=matches<=8:
            raise GraphError('invalid match count')
        if matches != 1:
            if np.any(logits):
                raise GraphError('nonunique graph must mask answer')
            return {'status':'missing' if matches==0 else 'ambiguous','value':None,'matches':matches}
        tokens = logits[0].argmax(-1).tolist()
        if 256 not in tokens:
            raise GraphError('unterminated binding output')
        try:
            value = bytes(tokens[:tokens.index(256)]).decode('utf-8')
        except UnicodeDecodeError as exc:
            raise GraphError('invalid UTF-8 binding output') from exc
        return {'status':'ok','value':value,'matches':matches}
