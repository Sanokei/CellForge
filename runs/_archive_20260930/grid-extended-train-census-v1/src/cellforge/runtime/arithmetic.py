"""Decimal encoding and scheduling; all addition and carries execute in graph."""
import re
import numpy as np
from ..ir.graph import GraphError
from .interp import Interpreter


class DecimalAdditionPredictor:
    def __init__(self, graph):
        graph.validate()
        if graph.meta.get('arithmetic_interface') != 'decimal-addition-graph/1':
            raise GraphError('requires decimal-addition-graph/1')
        if graph.meta.get('max_digits') != 4096:
            raise GraphError('unsupported arithmetic bounds')
        self.graph = graph

    def __call__(self, a, b):
        for value in (a,b):
            if not isinstance(value,str) or not re.fullmatch('[0-9]{1,4096}',value):
                raise GraphError('operands must be unsigned ASCII decimal strings of 1..4096 digits')
        engine = Interpreter(self.graph)
        carry = np.zeros((1,1))
        digits = []
        for i in range(max(len(a),len(b))):
            out = engine.run_bundle(self.graph.entries['step'], {
                'a':np.array([[int(a[-1-i]) if i<len(a) else 0]],dtype=float),
                'b':np.array([[int(b[-1-i]) if i<len(b) else 0]],dtype=float),
                'carry':carry},'/addition')
            digit,carry = np.asarray(out['digit']),np.asarray(out['carry'])
            if digit.shape != (1,1) or carry.shape != (1,1):
                raise GraphError('invalid arithmetic output shape')
            if not np.isfinite(digit).all() or not np.isfinite(carry).all():
                raise GraphError('nonfinite arithmetic output')
            if digit[0,0] != int(digit[0,0]) or not 0<=digit[0,0]<=9 or carry[0,0] not in (0,1):
                raise GraphError('invalid digit or carry')
            digits.append(str(int(digit[0,0])))
        if carry[0,0]:
            digits.append(str(int(carry[0,0])))
        return {'status':'ok','sum':''.join(reversed(digits)).lstrip('0') or '0',
                'graph_hash':self.graph.hash(),'steps':max(len(a),len(b)),
                'execution':engine.stats.as_dict()}
