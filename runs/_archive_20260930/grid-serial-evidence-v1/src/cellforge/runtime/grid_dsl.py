"""REJECTED DRAFT: never executed or benchmarked; not an authorized primary.

Latest owner correction requires CellForge cells/wiring. Preserved only as a
negative proposal; use the graph gene builders instead.

Bounded CPU grid-program induction, separate from neural graph inference.

Only supplied demonstration pairs select programs. Test answers and task IDs
are absent from this interface. Enumeration/evolution are discrete program
search, with no parameter gradients or gradient estimates.
"""
from dataclasses import dataclass
import hashlib
import json
import time
import numpy as np
from ..reference.arc_grid_operator import validate_grid


class SearchTimeout(RuntimeError): pass


@dataclass(frozen=True)
class Operation:
    name: str
    args: tuple = ()


@dataclass(frozen=True)
class Program:
    steps: tuple = ()

    def record(self):return [[op.name,list(op.args)] for op in self.steps]
    def key(self):return json.dumps(self.record(),separators=(',',':'))
    def digest(self):return hashlib.sha256(self.key().encode()).hexdigest()


def _deadline(deadline):
    if deadline is not None and time.perf_counter()>=deadline:raise SearchTimeout('task CPU wall budget reached')


def _background(a,bg):return int(np.bincount(a.ravel(),minlength=10).argmax()) if bg==-1 else bg


def _objects(a,bg,same_color=True,diagonal=False,deadline=None):
    seen=np.zeros(a.shape,dtype=bool);objects=[]
    directions=[(-1,0),(1,0),(0,-1),(0,1)]
    if diagonal:directions += [(-1,-1),(-1,1),(1,-1),(1,1)]
    H,W=a.shape
    for y,x in zip(*np.nonzero(a!=bg)):
        if seen[y,x]:continue
        _deadline(deadline);color=int(a[y,x]);queue=[(int(y),int(x))];seen[y,x]=True;pixels=[]
        while queue:
            cy,cx=queue.pop();pixels.append((cy,cx))
            if len(pixels)%32==0:_deadline(deadline)
            for dy,dx in directions:
                ny,nx=cy+dy,cx+dx
                if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and a[ny,nx]!=bg and (not same_color or a[ny,nx]==color):
                    seen[ny,nx]=True;queue.append((ny,nx))
        objects.append(pixels)
    return objects


def _crop_mask(a,mask):
    y,x=np.nonzero(mask)
    if not len(y):raise ValueError('empty selection')
    return a[y.min():y.max()+1,x.min():x.max()+1].copy()


def apply_operation(a,op,deadline=None):
    _deadline(deadline);name,args=op.name,op.args
    if name=='identity':out=a.copy()
    elif name=='rotate':out=np.rot90(a,int(args[0])).copy()
    elif name=='flip':out=np.flip(a,int(args[0])).copy()
    elif name=='transpose':out=a.T.copy()
    elif name=='shift_colors':out=(a+int(args[0]))%10
    elif name=='color_map':out=np.asarray(args,dtype=np.int8)[a]
    elif name=='replace':out=np.where(a==args[0],args[1],a)
    elif name=='crop':out=_crop_mask(a,a!=_background(a,args[0]))
    elif name=='crop_color':out=_crop_mask(a,a==args[0])
    elif name=='interior':
        if min(a.shape)<=2:raise ValueError('no interior')
        out=a[1:-1,1:-1].copy()
    elif name=='scale':out=np.repeat(np.repeat(a,args[0],axis=0),args[1],axis=1)
    elif name=='tile':out=np.tile(a,args)
    elif name=='sample':out=a[args[2]::args[0],args[3]::args[1]].copy()
    elif name=='mirror_tile':
        horizontal=np.concatenate([a,np.fliplr(a)],axis=1)
        out=np.concatenate([horizontal,np.flipud(horizontal)],axis=0)
    elif name=='gravity':
        bg=_background(a,args[0]);direction=args[1];out=np.full(a.shape,bg,dtype=np.int8)
        if direction in ('down','up'):
            for x in range(a.shape[1]):
                values=a[:,x][a[:,x]!=bg]
                if len(values):out[-len(values):,x] = values if direction=='down' else out[-len(values):,x]
                if direction=='up':out[:len(values),x]=values
        else:
            for y in range(a.shape[0]):
                values=a[y][a[y]!=bg]
                if len(values):out[y,-len(values):] = values if direction=='right' else out[y,-len(values):]
                if direction=='left':out[y,:len(values)]=values
    elif name=='symmetry':
        bg=_background(a,args[0]);partner=np.flip(a,args[1]);out=np.where(a==bg,partner,a)
    elif name=='fill_holes':
        bg=_background(a,args[0]);fill=args[1];H,W=a.shape
        reachable=np.zeros(a.shape,dtype=bool);queue=[]
        for y in range(H):
            for x in range(W):
                if (y in (0,H-1) or x in (0,W-1)) and a[y,x]==bg:
                    reachable[y,x]=True;queue.append((y,x))
        while queue:
            _deadline(deadline);y,x=queue.pop()
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny,nx=y+dy,x+dx
                if 0<=ny<H and 0<=nx<W and not reachable[ny,nx] and a[ny,nx]==bg:
                    reachable[ny,nx]=True;queue.append((ny,nx))
        out=np.where((a==bg)&~reachable,fill,a)
    elif name in ('select_object','keep_object','count_objects'):
        bg=_background(a,args[0]);objects=_objects(a,bg,bool(args[2]),bool(args[3]),deadline)
        if not objects:raise ValueError('no foreground objects')
        if name=='count_objects':
            n=len(objects);color=int(a[objects[0][0]])
            out=np.full((1,n) if args[1]=='wide' else (n,1),color,dtype=np.int8)
        else:
            selected=sorted(objects,key=lambda p:(len(p),p[0]))[-1 if args[1]=='largest' else 0]
            mask=np.zeros(a.shape,dtype=bool)
            for y,x in selected:mask[y,x]=True
            canvas=np.where(mask,a,bg)
            out=_crop_mask(canvas,mask) if name=='select_object' else canvas
    elif name=='count_color':
        n=int((a==args[0]).sum())
        if n==0:raise ValueError('empty color count')
        out=np.full((1,n) if args[1]=='wide' else (n,1),args[0],dtype=np.int8)
    elif name=='fill_bbox':
        color=args[0];out=a.copy();y,x=np.nonzero(a==color)
        if not len(y):raise ValueError('empty color')
        out[y.min():y.max()+1,x.min():x.max()+1]=color
    elif name=='connect':
        color,axis=args;out=a.copy();work=out if axis==1 else out.T
        for row in work:
            indices=np.flatnonzero(row==color)
            if len(indices)>1:row[indices[0]:indices[-1]+1]=color
    else:raise ValueError('unknown DSL primitive')
    if out.ndim!=2 or not 1<=out.shape[0]<=30 or not 1<=out.shape[1]<=30 or np.any((out<0)|(out>9)):
        raise ValueError('invalid output grid shape/color')
    return np.asarray(out,dtype=np.int8)


def execute(program,grid,deadline=None):
    a=np.asarray(validate_grid(grid),dtype=np.int8)
    for op in program.steps:a=apply_operation(a,op,deadline)
    return a.tolist()


def _primitives(train):
    colors=sorted({v for pair in train for row in pair['input']+pair['output'] for v in row})
    backgrounds=sorted(set([0,-1]+[int(np.bincount(np.asarray(p['input']).ravel(),minlength=10).argmax()) for p in train]))
    ops=[Operation('rotate',(k,)) for k in (1,2,3)]+[Operation('flip',(a,)) for a in (0,1)]+[Operation('transpose'),Operation('interior'),Operation('mirror_tile')]
    ops += [Operation('shift_colors',(n,)) for n in range(1,10)]
    for bg in backgrounds:
        ops.append(Operation('crop',(bg,)))
        for direction in ('down','up','left','right'):ops.append(Operation('gravity',(bg,direction)))
        for axis in (0,1):ops.append(Operation('symmetry',(bg,axis)))
        for same in (0,1):
            for diagonal in (0,1):
                for size in ('largest','smallest'):
                    for name in ('select_object','keep_object'):ops.append(Operation(name,(bg,size,same,diagonal)))
                for shape in ('wide','tall'):ops.append(Operation('count_objects',(bg,shape,same,diagonal)))
        for color in colors:
            if color!=bg:ops.append(Operation('fill_holes',(bg,color)))
    for color in colors:
        ops.extend([Operation('crop_color',(color,)),Operation('fill_bbox',(color,))])
        for shape in ('wide','tall'):ops.append(Operation('count_color',(color,shape)))
        for axis in (0,1):ops.append(Operation('connect',(color,axis)))
        for other in colors:
            if color!=other:ops.append(Operation('replace',(color,other)))
    for h in (1,2,3):
        for w in (1,2,3):
            if h*w>1:
                ops.extend([Operation('scale',(h,w)),Operation('tile',(h,w)),Operation('sample',(h,w,0,0))])
    return tuple(ops)


def _infer_color_map(predictions,targets):
    mapping={}
    for a,b in zip(predictions,targets):
        if a.shape!=b.shape:return None
        for src,dst in zip(a.ravel(),b.ravel()):
            if int(src) in mapping and mapping[int(src)]!=int(dst):return None
            mapping[int(src)]=int(dst)
    result=tuple(mapping.get(i,i) for i in range(10))
    return result if result!=tuple(range(10)) else None


class GridDSLSolver:
    """Enumerative beam + optional seeded discrete program evolution.

    Ranking: exact training pairs, average cell agreement, matching dimensions,
    fewer operations, then canonical program. No test labels enter ranking.
    """
    def __init__(self,seconds=1.,max_programs=6000,max_depth=3,beam_size=32,seed=0,evolution_fraction=.25):
        raise RuntimeError('Rejected draft: owner requires graph-owned cells/wiring; no DSL program search is authorized')
        if not 0<seconds<=60 or not 1<=max_programs<=100000 or not 1<=max_depth<=6 or not 1<=beam_size<=128 or not 0<=evolution_fraction<1:
            raise ValueError('bounded DSL search configuration required')
        self.seconds=seconds;self.max_programs=max_programs;self.max_depth=max_depth;self.beam_size=beam_size;self.seed=seed;self.evolution_fraction=evolution_fraction
        self.last_result=None

    def solve(self,demonstrations,test_inputs):
        if not demonstrations or not test_inputs:raise ValueError('training pairs and test inputs required')
        train=[{'input':validate_grid(p['input']),'output':validate_grid(p['output'])} for p in demonstrations]
        xs=[np.asarray(p['input'],dtype=np.int8) for p in train];ys=[np.asarray(p['output'],dtype=np.int8) for p in train]
        queries=[validate_grid(x) for x in test_inputs]
        start=time.perf_counter();deadline=start+self.seconds;ops=_primitives(demonstrations)
        archive=[];seen=set();ledger=hashlib.sha256();counts={'verified':0,'invalid':0,'timeout':0,'exact_train':0};negative_examples=[]
        rng=np.random.default_rng(self.seed)

        def verify(program,origin,parents=()):
            if program.key() in seen:return None
            _deadline(deadline)
            if counts['verified']>=self.max_programs:raise SearchTimeout('program cap reached')
            seen.add(program.key());counts['verified']+=1
            try:
                predicted=[]
                for x in xs:
                    a=x.copy()
                    for op in program.steps:a=apply_operation(a,op,deadline)
                    predicted.append(a)
                matches=[a.shape==b.shape for a,b in zip(predicted,ys)]
                exact=sum(m and np.array_equal(a,b) for m,a,b in zip(matches,predicted,ys))
                agreement=sum(float((a==b).mean()) if m else 0. for m,a,b in zip(matches,predicted,ys))/len(ys)
                rank=(-exact,-agreement,-sum(matches),len(program.steps),program.key())
                record={'program':program,'rank':rank,'exact_pairs':int(exact),'agreement':agreement,'origin':origin,'parents':parents}
                archive.append(record);archive.sort(key=lambda r:r['rank']);del archive[max(self.beam_size*4,128):]
                counts['exact_train']+=int(exact==len(ys))
                ledger.update(json.dumps({'hash':program.digest(),'exact':int(exact),'agreement':agreement,'origin':origin,'parents':parents},sort_keys=True).encode())
                # Directly induced color mapping is constrained across ALL demos.
                mapping=_infer_color_map(predicted,ys)
                if mapping is not None and len(program.steps)<self.max_depth:
                    verify(Program(program.steps+(Operation('color_map',mapping),)),'train_consistent_color_map',(program.digest(),))
                return record
            except SearchTimeout:raise
            except (ValueError,IndexError,TypeError) as error:
                counts['invalid']+=1;ledger.update((program.digest()+':invalid:'+type(error).__name__).encode())
                if len(negative_examples)<8:negative_examples.append({'hash':program.digest(),'reason':str(error),'program':program.record()})
                return None

        try:
            verify(Program(),'identity')
            # Give all single primitives a chance before deeper expansion.
            for op in ops:verify(Program((op,)),'enumeration')
            enum_deadline=start+self.seconds*(1-self.evolution_fraction)
            for depth in range(2,self.max_depth+1):
                frontier=[r for r in archive if len(r['program'].steps)==depth-1][:self.beam_size]
                for record in frontier:
                    for op in ops:
                        if time.perf_counter()>=enum_deadline:break
                        verify(Program(record['program'].steps+(op,)),'beam',(record['program'].digest(),))
                    if time.perf_counter()>=enum_deadline:break
                if time.perf_counter()>=enum_deadline:break
            if self.evolution_fraction:
                while time.perf_counter()<deadline and counts['verified']<self.max_programs:
                    elite=archive[:self.beam_size];parent=elite[int(rng.integers(len(elite)))]['program'];steps=list(parent.steps)
                    choice=int(rng.integers(4));parents=[parent.digest()]
                    if choice==0 and len(steps)<self.max_depth:steps.insert(int(rng.integers(len(steps)+1)),ops[int(rng.integers(len(ops)))])
                    elif choice==1 and steps:steps[int(rng.integers(len(steps)))]=ops[int(rng.integers(len(ops)))]
                    elif choice==2 and steps:steps.pop(int(rng.integers(len(steps))))
                    else:
                        donor=elite[int(rng.integers(len(elite)))]['program'];parents.append(donor.digest())
                        cut=int(rng.integers(len(steps)+1));steps=(steps[:cut]+list(donor.steps[cut:]))[:self.max_depth]
                    verify(Program(tuple(steps)),'discrete_evolution',tuple(parents))
        except SearchTimeout:
            counts['timeout']+=1
        # Execute final candidates outside training search; bound every primitive
        # structurally to 30x30 and max_depth; choose distinct outputs, at most2.
        attempts=[];ranked=[]
        for record in archive:
            if len(ranked)>=32:break
            try:
                outputs=[execute(record['program'],[list(row) for row in q]) for q in queries]
            except (ValueError,IndexError):continue
            if outputs in [r[1] for r in ranked]:continue
            ranked.append((record,outputs))
        for j,q in enumerate(queries):
            grids=[]
            for _,outputs in ranked:
                if outputs[j] not in grids:grids.append(outputs[j])
                if len(grids)==2:break
            if not grids:grids=[[list(row) for row in q]]
            attempts.append(grids)
        result={'attempts':attempts,'seconds':time.perf_counter()-start,'counts':counts,
                'training_pairs':len(train),'exact_training_programs':counts['exact_train'],
                'candidate_ledger_sha256':ledger.hexdigest(),'negative_examples':negative_examples,
                'ranked_archive':[{'hash':r['program'].digest(),'program':r['program'].record(),'exact_pairs':r['exact_pairs'],
                                   'agreement':r['agreement'],'origin':r['origin'],'parents':list(r['parents'])} for r,_ in ranked[:8]],
                'config':{'seconds':self.seconds,'max_programs':self.max_programs,'max_depth':self.max_depth,'beam_size':self.beam_size,
                          'seed':self.seed,'evolution_fraction':self.evolution_fraction},
                'scope':'CPU symbolic program search; exact training fit is not test success or neural LLM reasoning'}
        self.last_result=result;return result

    def __call__(self,demonstrations,test_input):return self.solve(demonstrations,[test_input])['attempts'][0]
