"""Authored spatial genes using only existing tensor IR primitives.

Zero is background/missing. Original foreground wins; collisions between new
pixels choose the highest color. No demonstrations or benchmark data are read.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from cellforge.ir.graph import Graph, Limits
from cellforge.ir.profiles import check_profile
from build_grid_associative import Nodes
from build_grid_repeat_gene import build_repeat_gene


class Spatial:
    def __init__(self):
        self.g = Graph(meta=dict(task='grid', grid_interface='grid-graph/1',
            grid_attempts=1, state_dim=0, profile='ladder-micro/1',
            construction='authored spatial tensor gene/1', gradients=False,
            benchmark_tasks_read=0))
        self.n = Nodes()
        self.g.params['dimension.W'] = np.arange(1, 31, dtype=float)[None]
        self.g.params['dimension.b'] = np.zeros(1)
        self.g.defs['dimension'] = dict(kind='cell', act='identity',
            params=dict(W1='dimension.W', b1='dimension.b'))
        self.h = self.n.add('dimension', {'x': '$input_height'})
        self.w = self.n.add('dimension', {'x': '$input_width'})
        self.coords = self.n.add('op:coords', n=30)
        self.fg = self.n.add('op:slice', {'x': '$input'}, axis=2, start=1, stop=10)

    def op(self, name, **inputs):
        return self.n.op(name, **inputs)

    def bias(self, x, c):
        return self.n.add('op:bias', {'x': x}, c=c)

    def eq(self, a, b):
        return self.op('relu', x=self.bias(self.op('neg', x=self.op('abs', x=self.op('sub', a=a, b=b))), 1))

    def shift(self, x, dr, dc):
        for axis, k in ((0, dr), (1, dc)):
            if k:
                x = self.n.add('op:shift', {'x': x}, axis=axis, k=k)
        return x

    def reduce(self, x, op='max', axes=(2, 1, 0)):
        for axis in axes:
            x = self.n.add('op:'+op, {'x': x}, axis=axis, keepdims=True)
        return x

    def union(self, refs):
        result = refs[0]
        for ref in refs[1:]:
            result = self.op('maximum', a=result, b=ref)
        return result

    def render(self, fg):
        # Collapse competing nonzero channels to their highest color.
        weights = self.n.reshape(self.n.add('op:coords', n=9, start=1), [1, 1, 9])
        values = self.reduce(self.op('mul', a=fg, b=weights), axes=(2,))
        empty = self.n.add('op:slice', {'x': '$input'}, axis=2, start=0, stop=1)
        original = self.reduce(self.op('mul', a=self.fg, b=weights), axes=(2,))
        values = self.op('add', a=original, b=self.op('mul', a=values, b=empty))
        colors = self.n.reshape(self.n.add('op:coords', n=10), [1, 1, 10])
        return self.op('mul', a=self.eq(values, colors), b='$input_mask')

    def finish(self, name, colors, height='$input_height', width='$input_width', rule=''):
        ports = ['input', 'input_mask', 'input_height', 'input_width']
        self.g.defs['spatial.gene'] = self.n.bundle('cluster', ports,
            dict(colors=colors, height=height, width=width))
        n = Nodes()
        call = n.add('spatial.gene', {p: '$'+p for p in ports})
        outputs = {'state': '$state'}
        for dest, src, shape in [('color_logits', 'colors', [1,30,30,10]),
                                 ('height_logits', 'height', [1,30]),
                                 ('width_logits', 'width', [1,30])]:
            outputs[dest] = n.reshape(call+'.'+src, shape)
        ports = ports + ['state']
        self.g.defs['expert0'] = n.bundle('expert', ports, outputs)
        self.g.defs['expert1'] = copy.deepcopy(self.g.defs['expert0'])
        n = Nodes()
        scores = n.add('op:coords', n=2, start=0, step=0)
        self.g.defs['router'] = n.bundle('router', [], {'scores': scores})
        n = Nodes()
        r = n.add('router')
        d = n.add('dispatch', {**{p:'$'+p for p in ports}, 'scores':r+'.scores'}, branches=['expert0','expert1'], k=1)
        self.g.defs['root'] = n.bundle('root', ports, {p:d+'.'+p for p in outputs})
        self.g.entries = {'grid_step':'root'}
        self.g.meta.update(gene_name=name, gene_contract=dict(cluster='spatial.gene', inputs=ports[:-1], outputs=['colors','height','width'], rule=rule))
        self.g.validate(Limits(strict_microcells=True))
        check_profile(self.g, 'ladder-micro/1')
        return self.g

    def mirror(self, x, axis):
        size = self.h if axis == 0 else self.w
        src = self.n.reshape(self.coords, [1,30])
        dst = self.n.reshape(self.coords, [30,1])
        matrix = self.eq(self.op('add', a=src, b=dst), self.bias(size,-1))
        if axis == 1:
            x = self.n.add('op:transpose', {'x':x}, perm=[1,0,2])
        x = self.n.reshape(self.op('matmul', a=matrix, b=self.n.reshape(x,[30,270])), [30,30,9])
        return self.n.add('op:transpose', {'x':x}, perm=[1,0,2]) if axis == 1 else x


def build_symmetry_gene(mode):
    if mode not in ('mirror_h','mirror_v','rotate180','diagonal','antidiagonal','rotate90','dihedral'):
        raise ValueError('unknown symmetry')
    s = Spatial()
    def transform(x, kind):
        if kind == 'mirror_h': return s.mirror(x, 1)
        if kind == 'mirror_v': return s.mirror(x, 0)
        if kind == 'rotate180': return s.mirror(s.mirror(x,0),1)
        x = s.n.add('op:transpose', {'x':x}, perm=[1,0,2])
        if kind == 'antidiagonal': return s.mirror(s.mirror(x,0),1)
        if kind == 'rotate90': return s.mirror(x,1)
        return x
    square = s.eq(s.h,s.w)
    refs = [s.fg]
    kinds = ['mirror_h','mirror_v','rotate180','diagonal','antidiagonal','rotate90'] if mode == 'dihedral' else [mode]
    for kind in kinds:
        x = transform(s.fg,kind)
        if kind in ('diagonal','antidiagonal','rotate90'):
            x = s.op('mul', a=x, b=square)
        refs.append(x)
        if kind == 'rotate90':
            half = transform(x,'rotate90')
            refs.extend([half, transform(half,'rotate90')])
    return s.finish(mode, s.render(s.union(refs)), rule='complete symmetry into zero; diagonal/quarter turns require square; foreground preserved; highest color collision')


def build_periodic_gene(axis='both', max_period=6, period=None):
    if axis not in ('row','column','both','independent') or not 1 <= max_period <= 30 or (period is not None and not 1 <= period <= 30):
        raise ValueError('invalid period/axis')
    s = Spatial()
    result = s.fg
    taken = s.n.reshape(s.n.add('op:coords', n=1, start=0), [1,1,1])
    periods = ([period] if period is not None else range(1,max_period+1))
    candidates = (sorted(((r,c) for r in periods for c in periods), key=lambda rc:(rc[0]*rc[1],sum(rc),rc))
                  if axis=='independent' else [(p,p) for p in periods])
    for pr,pc in candidates:
        candidate = s.fg
        for ax in ([0,1] if axis in ('both','independent') else [0] if axis=='row' else [1]):
            p = pr if ax==0 else pc
            candidate = s.union([candidate]+[s.n.add('op:shift', {'x':candidate}, axis=ax, k=k) for k in range(-29,30) if k and k % p == 0])
        # A residue class must not contain two different observed colors.
        count = s.reduce(candidate, op='sum', axes=(2,))
        conflict = s.reduce(s.op('relu', x=s.bias(count,-1)))
        valid = s.bias(s.op('neg', x=s.n.add('op:step', {'x':conflict})),1)
        use = s.op('mul', a=valid, b=s.bias(s.op('neg',x=taken),1))
        result = s.op('add', a=s.op('mul', a=candidate,b=use), b=s.op('mul',a=result,b=s.bias(s.op('neg',x=use),1)))
        taken = s.op('maximum', a=taken,b=valid)
    return s.finish(f'periodic_{axis}_{period or "infer"}', s.render(result), rule=f'zero is missing; first conflict-free period <= {max_period}; independent periods rank by area, sum, row then column; observed colors persist; unseen residues stay zero')


DIRECTIONS = {'n':(-1,0),'ne':(-1,1),'e':(0,1),'se':(1,1),
              's':(1,0),'sw':(1,-1),'w':(0,-1),'nw':(-1,-1)}


def build_drawing_gene(mode, directions=('e',), steps=29):
    if mode not in ('ray','connect','bounce') or not 1 <= steps <= 29 or not directions or any(d not in DIRECTIONS for d in directions):
        raise ValueError('invalid drawing specification')
    s = Spatial()
    refs = [s.fg]
    empty = s.n.add('op:slice', {'x':'$input'}, axis=2,start=0,stop=1)
    for direction in directions:
        dr,dc = DIRECTIONS[direction]
        if mode == 'connect':
            forward = s.union([s.shift(s.fg,dr*k,dc*k) for k in range(1,30)])
            backward = s.union([s.shift(s.fg,-dr*k,-dc*k) for k in range(1,30)])
            refs.append(s.op('mul', a=forward,b=backward))
        elif mode == 'ray':
            tip = s.fg
            for _ in range(steps):
                tip = s.op('mul', a=s.shift(tip,dr,dc),b=empty)
                refs.append(tip)
        else:
            # Four direction states reflect at the actual rectangle boundary.
            states = {(r,c):s.op('mul',a=s.fg,b=s.n.reshape(s.n.add('op:coords',n=1,start=float((r,c)==(dr,dc))),[1,1,1]))
                      for r,c in {(dr,dc),(-dr,dc),(dr,-dc),(-dr,-dc)}}
            row = s.n.reshape(s.coords,[30,1,1]); col = s.n.reshape(s.coords,[1,30,1])
            for _ in range(steps):
                incoming = {key:[] for key in states}
                for (r,c),tip in states.items():
                    rb = s.eq(row, s.bias(s.h,-1) if r>0 else s.n.add('op:coords',n=1,start=0)) if r else s.n.reshape(s.n.add('op:coords',n=1,start=0),[1,1,1])
                    cb = s.eq(col, s.bias(s.w,-1) if c>0 else s.n.add('op:coords',n=1,start=0)) if c else s.n.reshape(s.n.add('op:coords',n=1,start=0),[1,1,1])
                    for flipr,flipc in ((0,0),(1,0),(0,1),(1,1)):
                        rr,cc = (-r if flipr else r),(-c if flipc else c)
                        mask = s.op('mul',a=rb if flipr else s.bias(s.op('neg',x=rb),1),b=cb if flipc else s.bias(s.op('neg',x=cb),1))
                        moved = s.op('mul',a=tip,b=mask)
                        for ax,delta,size in ((0,rr,s.h),(1,cc,s.w)):
                            if delta:
                                singleton=s.eq(size,s.n.add('op:coords',n=1,start=1))
                                shifted=s.n.add('op:shift',{'x':moved},axis=ax,k=delta)
                                moved=s.op('add',a=s.op('mul',a=moved,b=singleton),b=s.op('mul',a=shifted,b=s.bias(s.op('neg',x=singleton),1)))
                        incoming[(rr,cc)].append(s.op('mul',a=moved,b='$input_mask'))
                states = {key:s.union(items) for key,items in incoming.items()}
                refs.extend(states.values())
    return s.finish(f'{mode}_{"_".join(directions)}_{steps}',s.render(s.union(refs)),rule='ray stops at foreground; connect joins collinear same-color dots; bounce reflects at rectangle edges, passes foreground; original foreground wins')


def build_kron_gene(stencil='diagonal', factor=2):
    if factor not in (2,3,4) or stencil not in ('diagonal','antidiagonal','cross','border','checker'):
        raise ValueError('invalid stencil/factor')
    s = Spatial()
    src = s.n.reshape(s.coords,[1,30]); dst = s.n.reshape(s.coords,[30,1])
    scaled = s.n.add('op:scale',{'x':src},c=factor)
    refs = []
    for r in range(factor):
        for c in range(factor):
            active = {'diagonal':r==c,'antidiagonal':r+c==factor-1,
                'cross':r==factor//2 or c==factor//2,'border':r in (0,factor-1) or c in (0,factor-1),'checker':(r+c)%2==0}[stencil]
            if not active: continue
            rm=s.eq(dst,s.bias(scaled,r)); cm=s.eq(dst,s.bias(scaled,c))
            x=s.n.reshape(s.op('matmul',a=rm,b=s.n.reshape(s.fg,[30,270])),[30,30,9])
            x=s.n.add('op:transpose',{'x':x},perm=[1,0,2])
            x=s.n.reshape(s.op('matmul',a=cm,b=s.n.reshape(x,[30,270])),[30,30,9])
            refs.append(s.n.add('op:transpose',{'x':x},perm=[1,0,2]))
    fg=s.union(refs)
    occupied=s.reduce(fg,axes=(2,))
    background=s.bias(s.op('neg',x=occupied),1)
    colors=s.n.add('op:concat',{'x0':background,'x1':fg},axis=2)
    dims=[]; overflow=[]
    for size in (s.h,s.w):
        size=s.n.add('op:scale',{'x':size},c=factor)
        dims.append(s.eq(s.bias(s.coords,1),size))
        overflow.append(s.op('relu',x=s.bias(size,-30)))
    bad=s.n.add('op:step',{'x':s.op('add',a=overflow[0],b=overflow[1])})
    good=s.bias(s.op('neg',x=bad),1)
    choose=lambda a,b:s.op('add',a=s.op('mul',a=a,b=good),b=s.op('mul',a=b,b=bad))
    return s.finish(f'kron_{stencil}_{factor}',choose(colors,'$input'),choose(dims[0],'$input_height'),choose(dims[1],'$input_width'),rule='replace each pixel with its color times a binary stencil; zero background; overflow returns identity')


def build_scale_gene(row_factor=2, column_factor=2, mode='enlarge'):
    if row_factor not in (1,2,3,4) or column_factor not in (1,2,3,4) or mode not in ('enlarge','sample'):
        raise ValueError('invalid scaling specification')
    s=Spatial(); x='$input'; dims=[]; invalid=[]
    src=s.n.reshape(s.coords,[1,30]); dst=s.n.reshape(s.coords,[30,1])
    for ax,factor,size,port in ((0,row_factor,s.h,'height'),(1,column_factor,s.w,'width')):
        if mode=='enlarge':
            positions=s.n.add('op:scale',{'x':src},c=factor)
            matrix=s.union([s.eq(dst,s.bias(positions,k)) for k in range(factor)])
            outsize=s.n.add('op:scale',{'x':size},c=factor)
        else:
            matrix=s.eq(src,s.n.add('op:scale',{'x':dst},c=factor))
            outsize=s.n.add('op:scale',{'x':size},c=1/factor)
        hot=s.eq(s.bias(s.coords,1),outsize); dims.append(hot)
        # Sampling requires exact divisibility; any invalid size is identity.
        present=s.n.add('op:sum',{'x':hot},axis=0,keepdims=True)
        invalid.append(s.bias(s.op('neg',x=present),1))
        if ax==1: x=s.n.add('op:transpose',{'x':x},perm=[1,0,2])
        x=s.n.reshape(s.op('matmul',a=matrix,b=s.n.reshape(x,[30,300])),[30,30,10])
        if ax==1: x=s.n.add('op:transpose',{'x':x},perm=[1,0,2])
    bad=s.n.add('op:step',{'x':s.op('add',a=invalid[0],b=invalid[1])}); good=s.bias(s.op('neg',x=bad),1)
    choose=lambda a,b:s.op('add',a=s.op('mul',a=a,b=good),b=s.op('mul',a=b,b=bad))
    return s.finish(f'{mode}_{row_factor}_{column_factor}',choose(x,'$input'),choose(dims[0],'$input_height'),choose(dims[1],'$input_width'),rule='independent axis nearest-neighbor enlargement or top-left strided sampling; overflow or nondivisible sampling returns identity')


def gene_specs():
    specs={}
    for mode in ('mirror_h','mirror_v','rotate180','diagonal','antidiagonal','rotate90','dihedral'):
        specs[mode]=('symmetry',dict(mode=mode))
    for axis in ('row','column','both'):
        specs[f'periodic_{axis}_infer']=('periodic',dict(axis=axis))
        for p in range(2,7): specs[f'periodic_{axis}_{p}']=('periodic',dict(axis=axis,period=p))
    specs['periodic_independent_infer']=('periodic',dict(axis='independent'))
    for mode in ('ray','connect','bounce'):
        for d in DIRECTIONS: specs[f'{mode}_{d}']=('drawing',dict(mode=mode,directions=(d,)))
        for label,ds in [('orthogonal',('n','e','s','w')),('diagonal',('ne','se','sw','nw')),('all',tuple(DIRECTIONS))]:
            specs[f'{mode}_{label}']=('drawing',dict(mode=mode,directions=ds))
    for factor in (2,3,4):
        for stencil in ('diagonal','antidiagonal','cross','border','checker'):
            specs[f'kron_{stencil}_{factor}']=('kron',dict(stencil=stencil,factor=factor))
    for factor in (2,3):
        for mode in ('repeat','tile'): specs[f'{mode}_{factor}']=('repeat',dict(factor=factor,mode=mode))
    for mode in ('enlarge','sample'):
        for r in (1,2,3,4):
            for c in (1,2,3,4):
                if (r,c)!=(1,1): specs[f'{mode}_{r}_{c}']=('scale',dict(row_factor=r,column_factor=c,mode=mode))
    return specs


def build_named_gene(name):
    family,args=gene_specs()[name]
    return {'symmetry':build_symmetry_gene,'periodic':build_periodic_gene,
            'drawing':build_drawing_gene,'kron':build_kron_gene,'repeat':build_repeat_gene,'scale':build_scale_gene}[family](**args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--gene',choices=tuple(gene_specs()))
    args=p.parse_args(); out=Path(args.out)
    if out.exists(): raise ValueError('fresh output directory required')
    out.mkdir(parents=True)
    inventory=[]
    for name in ([args.gene] if args.gene else gene_specs()):
        g=build_named_gene(name)
        inventory.append(dict(name=name,path=name,hash=g.save(out/name),stats=g.stats()))
    (out/'INVENTORY.json').write_text(json.dumps(dict(protocol='spatial-gene-library/1',benchmark_tasks_read=0,genes=inventory),indent=2)+'\n')
    print(json.dumps(dict(genes=len(inventory),inventory=str(out/'INVENTORY.json'))))


if __name__=='__main__': main()
