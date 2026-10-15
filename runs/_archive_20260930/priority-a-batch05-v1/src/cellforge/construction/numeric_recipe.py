"""Explicit gradient-free numerical recipes over homologous authored graphs.

The caller supplies every mask, row and scalar. This module evaluates no fitness,
estimates no gradients and samples no proposals; charged evaluation belongs to
MakerOperator. A returned graph is an unadmitted candidate.
"""
from __future__ import annotations

import copy
import numpy as np

from .transactions import TransactionError, apply_ops
from ..ir.graph import Graph


def apply_numeric_recipe(left: Graph, right: Graph, recipe: list[dict]) -> Graph:
    """Materialize bounded tensor crossover/row scaling from two exact parents."""
    if left.meta.get('gradients') is not False or right.meta.get('gradients') is not False:
        raise TransactionError('numeric recipes require explicit no-gradient parent provenance')
    if (left.defs != right.defs or left.entries != right.entries
            or left.meta.get('state_dim') != right.meta.get('state_dim')
            or set(left.params) != set(right.params)):
        raise TransactionError('numeric parents must have homologous definitions, ports and parameters')
    for name in left.params:
        if left.params[name].shape != right.params[name].shape:
            raise TransactionError('numeric parent parameter shapes differ')
    if not recipe or len(recipe) > 32:
        raise TransactionError('numeric recipe must contain 1..32 explicit edits')
    working = left.copy()
    ops = []
    for edit in recipe:
        name = edit.get('name')
        if name not in working.params:
            raise TransactionError('unknown numerical parameter')
        a = working.params[name].copy()
        if edit.get('op') == 'homologous_mask':
            mask = np.asarray(edit.get('mask'))
            if mask.shape != a.shape or not np.all((mask == 0) | (mask == 1)):
                raise TransactionError('crossover mask must be binary and exactly tensor-shaped')
            a = np.where(mask.astype(bool), right.params[name], a)
        elif edit.get('op') == 'scale_rows':
            rows, factor = edit.get('rows'), edit.get('factor')
            if (a.ndim != 2 or not isinstance(rows, list) or not rows
                    or any(type(i) is not int or not 0 <= i < a.shape[0] for i in rows)
                    or len(set(rows)) != len(rows)):
                raise TransactionError('row scaling requires distinct valid matrix rows')
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not np.isfinite(factor):
                raise TransactionError('row scale must be a finite explicit scalar')
            a[rows] *= factor
        else:
            raise TransactionError('unsupported numerical recipe operation')
        if not np.all(np.isfinite(a)):
            raise TransactionError('numerical recipe produced non-finite weights')
        working.params[name] = a
        ops.append(dict(op='set_param', name=name, value=a.tolist()))
    candidate, _ = apply_ops(left, ops)
    candidate.validate()
    candidate.meta['numeric_recipe'] = dict(
        protocol='explicit-numerical-recipe/1', parents=[left.hash(), right.hash()],
        edits=copy.deepcopy(recipe), gradients=False, status='unadmitted')
    return candidate
