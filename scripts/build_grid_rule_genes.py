"""Reusable authored property/rule genes. All query computation is generic IR.

Colors are explicit (including zero), updates simultaneous, padding excluded.
Floods and source-color components use bounded propagation with identity fallback.
No benchmark data, fitted labels, runtime Python callbacks or new semantic ops.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from cellforge.ir.graph import Graph, Limits
from cellforge.ir.profiles import check_profile
from build_grid_local_fill_gene import Nodes


RULES = (
    'touching', 'not_touching', 'neighbor_count', 'isolated', 'interior',
    'boundary', 'border', 'not_border', 'row', 'column', 'top', 'bottom',
    'left', 'right', 'row_parity', 'column_parity', 'checkerboard',
    'color_size', 'majority', 'minority', 'row_majority', 'row_minority',
    'column_majority', 'column_minority', 'neighbor_majority',
    'neighbor_minority', 'local_table', 'flood_marker', 'flood_border',
    'enclosed', 'component_size', 'component_parity', 'component_touching',
    'component_not_touching', 'component_border', 'component_not_border',
    'component_largest', 'component_smallest', 'contained_bbox',
)


def add_rule_cluster(graph, rule, source=2, target=3, marker=4,
                     value=1, connectivity=4, iterations=64, table=None,
                     prefix='rule'):
    """Install a composable grid-graph/1 cluster; returns its definition name.

    Size means exact pixel count. Component rules concern source-color objects.
    Contained_bbox tests strict containment in the global marker bounding box;
    enclosed instead tests topological disconnection from the valid grid border.
    Local table maps exact marker-neighbor counts to output colors; omitted
    counts preserve source. Frequency ties choose the lowest present color.
    """
    if rule not in RULES:
        raise ValueError('unknown rule')
    if any(type(c) is not int or not 0 <= c < 10 for c in (source, target, marker)):
        raise ValueError('colors must be integers 0..9')
    if connectivity not in (4, 8) or type(iterations) is not int or not 1 <= iterations <= 900:
        raise ValueError('connectivity must be 4/8 and iterations 1..900')
    if type(value) is not int or not 0 <= value <= 900:
        raise ValueError('value must be an integer 0..900')
    if rule.endswith('parity') or rule == 'checkerboard':
        if value not in (0, 1):
            raise ValueError('parity value must be 0 or 1')
    table = {} if table is None else dict(table)
    if any(type(k) is not int or not 0 <= k <= connectivity or
           type(v) is not int or not 0 <= v < 10 for k, v in table.items()):
        raise ValueError('table maps neighbor counts to colors')
    if any(name == prefix or name.startswith(prefix + '.') for name in graph.defs):
        raise ValueError('prefix collision')
    n = Nodes()
    def op(name, **inputs): return n.add('op:' + name, inputs)
    def bias(x, c): return n.add('op:bias', {'x': x}, c=c)
    def scale(x, c): return n.add('op:scale', {'x': x}, c=c)
    def reshape(x, shape): return n.add('op:reshape', {'x': x}, shape=shape)
    def inv(x): return bias(op('neg', x=x), 1)
    def eq(a, b): return op('relu', x=bias(op('neg', x=op('abs', x=op('sub', a=a, b=b))), 1))
    def const(v):
        key = prefix + '.constant' + str(len(graph.params))
        graph.params[key] = np.asarray(v, dtype=float)
        return n.add('param:' + key)
    def reduce(x, name='sum', axis=2, keepdims=True):
        return n.add('op:' + name, {'x': x}, axis=axis, keepdims=keepdims)
    def total(x): return reduce(reshape(x, [900]), axis=0)
    def add_all(xs):
        result = xs[0]
        for x in xs[1:]: result = op('add', a=result, b=x)
        return result
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8: offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    def shift(x, dr, dc, fill=0):
        for axis, k in ((0, dr), (1, dc)):
            if k: x = n.add('op:shift', {'x': x}, axis=axis, k=k, value=fill)
        return x
    def neighbors(x): return [shift(x, *offset) for offset in offsets]
    def adjacent(x): return op('step', x=add_all(neighbors(x)))
    valid = reduce('$input')
    sm = n.add('op:slice', {'x': '$input'}, axis=2, start=source, stop=source+1)
    mm = n.add('op:slice', {'x': '$input'}, axis=2, start=marker, stop=marker+1)
    rows = reshape(n.add('op:coords', n=30), [30, 1, 1])
    cols = reshape(n.add('op:coords', n=30), [1, 30, 1])
    # Border derives from validity, not zero-valued color or padded dimensions.
    orth_valid = [shift(valid, *o) for o in offsets[:4]]
    border = op('mul', a=valid, b=op('step', x=op('sub', a=const(4), b=add_all(orth_valid))))
    count = add_all(neighbors(mm))
    gate = sm
    settled = const(1)
    replacement = const(np.eye(10)[target].reshape(1, 1, 10))
    if rule in ('touching', 'not_touching', 'neighbor_count'):
        condition = eq(count, const(value)) if rule == 'neighbor_count' else op('step', x=count)
        if rule == 'not_touching': condition = inv(condition)
    elif rule in ('isolated', 'interior', 'boundary'):
        own = add_all(neighbors(sm))
        condition = eq(own, const(0 if rule == 'isolated' else connectivity))
        if rule == 'boundary': condition = inv(condition)
    elif rule in ('border', 'not_border'):
        condition = border if rule == 'border' else inv(border)
    elif rule in ('row', 'column', 'top', 'bottom', 'left', 'right'):
        coordinate = rows if rule in ('row', 'top', 'bottom') else cols
        location = const(value if rule in ('row', 'column') else 0)
        if rule in ('bottom', 'right'):
            dimension = '$input_height' if rule == 'bottom' else '$input_width'
            location = reduce(op('mul', a=dimension, b=reshape(n.add('op:coords', n=30), [30])), axis=0)
        condition = eq(coordinate, location)
    elif rule in ('row_parity', 'column_parity', 'checkerboard'):
        parity = (np.arange(30) % 2).astype(float)
        rp = const(parity.reshape(30,1,1))
        cp = const(parity.reshape(1,30,1))
        if rule == 'checkerboard':
            condition = eq(rp, cp)
            if value: condition = inv(condition)
        else:
            condition = eq(rp if rule == 'row_parity' else cp, const(value))
    elif rule == 'color_size': condition = eq(total(sm), const(value))
    elif rule in ('majority', 'minority', 'row_majority', 'row_minority',
                  'column_majority', 'column_minority', 'neighbor_majority', 'neighbor_minority'):
        if rule.startswith('row_'): hist = reduce('$input', axis=1)
        elif rule.startswith('column_'): hist = reduce('$input', axis=0)
        elif rule.startswith('neighbor_'): hist = add_all(neighbors('$input'))
        else: hist = reduce(reduce('$input', axis=0), axis=0)
        present = op('step', x=hist)
        scores = hist
        if rule.endswith('minority'):
            scores = op('sub', a=op('neg', x=hist), b=scale(inv(present), 1000))
        winner = n.add('op:onehot_argmax', {'x': scores}, axis=2)
        condition = reduce(op('mul', a=winner, b=const(np.eye(10)[source])))
        condition = op('mul', a=condition, b=op('step', x=reduce(hist)))
    elif rule == 'local_table':
        gates = [eq(count, const(k)) for k in sorted(table)]
        condition = add_all(gates) if gates else scale(sm, 0)
        replacement = add_all([op('mul', a=g, b=const(np.eye(10)[table[k]].reshape(1, 1, 10)))
                               for k, g in zip(sorted(table), gates)]) if gates else scale('$input', 0)
    elif rule in ('flood_marker', 'flood_border', 'enclosed'):
        reached = op('mul', a=sm, b=adjacent(mm) if rule == 'flood_marker' else border)
        def advance(x): return op('mul', a=sm, b=op('step', x=op('add', a=x, b=adjacent(x))))
        for _ in range(iterations): reached = advance(reached)
        settled = inv(op('step', x=total(op('abs', x=op('sub', a=reached, b=advance(reached))))))
        condition = inv(reached) if rule == 'enclosed' else reached
    elif rule == 'contained_bbox':
        condition = op('step', x=total(mm))
        for coordinate in (rows, cols):
            coords = op('add', a=scale(valid, 0), b=coordinate)
            low = reduce(reshape(op('add', a=coords, b=scale(inv(mm), 30)), [900]), name='min', axis=0)
            high = reduce(reshape(op('mul', a=coords, b=mm), [900]), name='max', axis=0)
            condition = op('mul', a=condition, b=op('mul',
                a=op('step', x=op('sub', a=coordinate, b=low)),
                b=op('step', x=op('sub', a=high, b=coordinate))))
    else:
        labels_axis = n.add('op:coords', n=900, start=1.)
        labels = op('add', a=op('mul', a=reshape(labels_axis, [30,30,1]), b=sm), b=scale(inv(sm), 901))
        def advance(x):
            candidates = [x] + [op('add', a=op('mul', a=shift(x, *o, fill=901), b=shift(sm, *o)),
                                  b=scale(inv(shift(sm, *o)), 901)) for o in offsets]
            minimum = reduce(n.add('op:concat', {f'x{i}':v for i,v in enumerate(candidates)}, axis=2), name='min')
            return op('add', a=op('mul', a=minimum, b=sm), b=scale(inv(sm), 901))
        for _ in range(iterations): labels = advance(labels)
        settled = inv(op('step', x=total(op('abs', x=op('sub', a=labels, b=advance(labels))))))
        membership = eq(reshape(labels, [900,1]), reshape(labels_axis, [1,900]))
        areas = reduce(membership, axis=0, keepdims=False)
        def broadcast_components(x):
            return reshape(op('matmul', a=membership, b=reshape(x, [900,1])), [30,30,1])
        if rule == 'component_size': choice = eq(areas, const(value))
        elif rule == 'component_parity':
            sizes = n.add('op:coords', n=451 if value == 0 else 450, start=float(value), step=2.)
            choice = reduce(eq(reshape(areas, [900,1]), reshape(sizes, [1,451 if value == 0 else 450])), axis=1, keepdims=False)
        elif rule in ('component_largest', 'component_smallest'):
            scores = areas if rule == 'component_largest' else op('sub', a=op('neg', x=areas), b=scale(inv(op('step', x=areas)), 1000))
            choice = n.add('op:onehot_argmax', {'x':scores}, axis=0)
        else:
            property_mask = adjacent(mm) if 'touching' in rule else border
            transposed = n.add('op:transpose', {'x':membership}, perm=[1,0])
            choice = op('step', x=reshape(op('matmul', a=transposed, b=reshape(property_mask,[900,1])), [900]))
            if '_not_' in rule: choice = inv(choice)
        condition = broadcast_components(choice)
    gate = op('mul', a=sm, b=op('mul', a=condition, b=settled))
    colors = op('add', a=op('mul', a='$input', b=inv(gate)), b=op('mul', a=gate, b=replacement))
    graph.defs[prefix+'.gene'] = n.bundle('cluster', ['input','input_height','input_width'],
        {'colors':colors, 'height':'$input_height', 'width':'$input_width'})
    return prefix+'.gene'


def build_rule_gene(rule, **kwargs):
    g = Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':1,
        'state_dim':0,'profile':'ladder-micro/1','construction':'authored property rule/1',
        'gene_contract':{'cluster':'rule.gene','rule':rule,'configuration':kwargs,
                         'inputs':['input','input_height','input_width'],
                         'outputs':['colors','height','width'],
                         'updates':'simultaneous; dimensions preserved; bounded propagation identity fallback'},
        'limitations':'explicit authored configuration; no demonstration adaptation; source-color components; frequency ties lowest color; component size ties row-major'})
    gene = add_rule_cluster(g, rule, **kwargs)
    g.meta['gene_contract']['cluster'] = gene
    ports = ['input','input_mask','input_height','input_width','output','output_mask',
             'output_height','output_width','phase','index','state']
    n = Nodes(); call = n.add(gene, {p:'$'+p for p in ('input','input_height','input_width')})
    outputs = {'state':'$state'}
    for port, name, shape in [('color_logits','colors',[1,30,30,10]),
                             ('height_logits','height',[1,30]),('width_logits','width',[1,30])]:
        outputs[port] = n.add('op:reshape', {'x':call+'.'+name}, shape=shape)
    g.defs['rule_expert0'] = n.bundle('expert',ports,outputs)
    g.defs['rule_expert1'] = copy.deepcopy(g.defs['rule_expert0'])
    n = Nodes(); scores = n.add('op:coords',n=2,start=0.,step=0.)
    g.defs['rule_router'] = n.bundle('router',[],{'scores':scores})
    n = Nodes(); router = n.add('rule_router')
    dispatch = n.add('dispatch',{**{p:'$'+p for p in ports},'scores':router+'.scores'},
                     branches=['rule_expert0','rule_expert1'],k=1)
    g.defs['rule_root'] = n.bundle('root',ports,{p:dispatch+'.'+p for p in outputs})
    g.entries = {'grid_step':'rule_root'}
    g.validate(Limits(strict_microcells=True)); check_profile(g,'ladder-micro/1')
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rule',choices=RULES,required=True); p.add_argument('--out',required=True)
    for name, default in [('source',2),('target',3),('marker',4),('value',1),('connectivity',4),('iterations',64)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--table',default='{}',help='JSON neighbor-count to color mapping')
    args = p.parse_args(); out = Path(args.out)
    if out.exists(): raise ValueError('preserve existing artifact')
    options = {k:getattr(args,k) for k in ('source','target','marker','value','connectivity','iterations')}
    options['table'] = {int(k):v for k,v in json.loads(args.table).items()}
    g = build_rule_gene(args.rule,**options); digest = g.save(out)
    receipt = {'hash':digest,'stats':g.stats(),'gene_contract':g.meta['gene_contract'],
               'benchmark_tasks_read':0,'curriculum_scored':False}
    (out/'MODEL_CARD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
