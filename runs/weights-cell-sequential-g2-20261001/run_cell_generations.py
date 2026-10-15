"""Cell-only bounded sequential generations; model proposals are external receipts."""
import argparse,copy,datetime,hashlib,json,os,time,resource
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from cellforge.construction.cell_population import CellPopulation
from cellforge.construction.joint_weight_population import digest
from cellforge.construction.typed_population import PaletteExecutor,parity_row_stencil,measure_resources,calibrate_caps,resource_pass
from cellforge.construction.component_count import SHAPES as COMPONENT_SHAPES,predict_component_count
from cellforge.construction.patch_cell import PATCH_SHAPES,predict_patch
ROOT=Path.cwd();OLD=Path('/srv/cellforge-inbox/weights-real-learning-v2-20261001')
V3=Path('/srv/cellforge-inbox/weights-real-learning-v3-20261001')
palette=PaletteExecutor(OLD/'pop1_core.py',10)
def count_pre(p,grid):
    a=np.asarray(grid);h,w=a.shape;f=np.asarray(p['self.W'])[0,a]+p['cnt.b'][0]
    for dr,dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        for r in range(h):
            for c in range(w):
                rr,cc=r+dr,c+dc
                if 0<=rr<h and 0<=cc<w:f[r,c]+=p['cnt.W'][0][a[rr,cc]]
    return f
def count_patch(p,grid):
    a=np.asarray(grid);colors=set(a[a!=0].tolist());density=p['patch.guard.density'][0]
    if not 0<=density<=1:raise ValueError('typed density range')
    if not (np.any(a==0) and len(colors)==1 and 1 not in colors and np.count_nonzero(a)/a.size>density):return a.tolist()
    anchors=count_pre(p,grid)>0;out=a.copy();h,w=a.shape
    for r in range(h):
        for c in range(w):
            f=p['patch.emit.b'][0]
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    rr,cc=r-dr,c-dc
                    if 0<=rr<h and 0<=cc<w:f+=p['patch.emit.W'][dr+1][dc+1]*anchors[rr,cc]
            if f>0:out[r,c]=int(np.argmax(p['paint']))
    return out.tolist()
def rectangle_filter(p,grid):
    a=np.asarray(grid);h,w=a.shape
    body=int(np.argmax(p['filter.body.W']));marker=int(np.argmax(p['filter.marker.W']))
    colors=set(a[a!=0].tolist())
    if body==marker or body==0 or marker==0 or not np.any(a==0) or colors!={body,marker}:return count_patch(p,grid)
    unseen={(r,c) for r in range(h) for c in range(w) if a[r,c] in (body,marker)}
    components=[]
    while unseen:
        start=min(unseen);unseen.remove(start);stack=[start];component=[start]
        while stack:
            r,c=stack.pop()
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                q=(r+dr,c+dc)
                if q in unseen:unseen.remove(q);stack.append(q);component.append(q)
        rs=[x[0] for x in component];cs=[x[1] for x in component]
        if (max(rs)-min(rs)+1)*(max(cs)-min(cs)+1)!=len(component):return count_patch(p,grid)
        components.append(component)
    out=np.zeros_like(a)
    for component in components:
        count=sum(p['filter.marker.W'][0][a[r,c]] for r,c in component)
        score=p['filter.component.W'][0][0]+p['filter.component.W'][0][1]*count+p['filter.component.b'][0]
        if score>0:
            for r,c in component:out[r,c]=a[r,c]
    return out.tolist()
SHAPES=dict(palette.core.SHAPES)
SHAPES['parity_row_stencil_v1']={'stencil.W':(2,3),'foreground.W':(1,10),'activation.b':(1,)}
SHAPES['component_count_unary_v1']=COMPONENT_SHAPES
SHAPES['guarded_conv_patch_emit_v1']=dict(SHAPES['conv3x3_detect'],**PATCH_SHAPES)
SHAPES['guarded_count_patch_emit_v1']=dict(SHAPES['neighbor_count'],**{'patch.emit.W':(3,3),'patch.emit.b':(1,),'patch.guard.density':(1,)})
SHAPES['guarded_count_patch_emit_v2_component_filter']=dict(SHAPES['guarded_count_patch_emit_v1'],**{'filter.body.W':(1,10),'filter.marker.W':(1,10),'filter.component.W':(1,2),'filter.component.b':(1,)})
def predict(structure,p,g):
    if structure=='parity_row_stencil_v1':return parity_row_stencil(p,g)
    if structure=='component_count_unary_v1':return predict_component_count(p,g)
    if structure=='guarded_conv_patch_emit_v1':return predict_patch(p,g,lambda pp,gg:palette.predict('conv3x3_detect',pp,gg))
    if structure=='guarded_count_patch_emit_v1':return count_patch(p,g)
    if structure=='guarded_count_patch_emit_v2_component_filter':return rectangle_filter(p,g)
    return palette.predict(structure,p,g)
def validate(s,p):
    if s not in SHAPES or set(p)!=set(SHAPES[s]):raise ValueError('known typed cell parameters required')
    for k,shape in SHAPES[s].items():
        a=np.asarray(p[k],float)
        if a.shape!=shape or not np.isfinite(a).all() or np.max(np.abs(a))>9:raise ValueError('invalid literal tensor')
    if 'patch.guard.density' in p and not 0<=p['patch.guard.density'][0]<=1:raise ValueError('invalid density')
dev=json.loads((V3/'development.json').read_text())
anchors=json.loads((OLD/'old-anchors.json').read_text(encoding='utf-8-sig'))
PARTS={'fresh':dev[4:],'archive':dev[:4],'anchor':[{'id':i,'task':t} for i,t in anchors]}
def scores(s,p):
    result={}
    for label,rows in PARTS.items():
        outcomes={};consistent=wrong=0
        for row in rows:
            t=row['task'];dc=all(predict(s,p,pair['input'])==pair['output'] for pair in t['train'])
            hits=[predict(s,p,pair['input'])==pair['output'] for pair in t['test']]
            outcomes[row['id']]=all(hits);consistent+=int(dc);wrong+=int(dc and not all(hits))
        result[label]={'partition':'development','manifest':digest([r['id'] for r in rows]),
        'source':'public-training-real','outcomes':outcomes,'false_positive_given_consistency':wrong/consistent if consistent else None}
    return result
def evaluate(job):
    s,p=job
    try:
        validate(s,p)
        result,probe=measure_resources(lambda:scores(s,p))
        return {'scores':result,'resources':probe,'valid':True}
    except (ValueError,IndexError,TypeError) as error:return {'valid':False,'reason':str(error)}
def depth(cells,cid):
    parents=cells[cid]['parents'];return 0 if not parents else 1+max(depth(cells,p) for p in parents)
def stage(generation,proposal_file):
    started=time.perf_counter();started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps({'event':'generation-start','generation':generation,'pid':os.getpid(),'utc':started_utc}),flush=True);plan=json.loads((ROOT/'PREDECLARED.json').read_text())
    response=json.loads(Path(proposal_file).read_text(encoding='utf-8-sig'));entries=[response['numerical'],response['structural']]
    previous=ROOT/f'generation-{generation-1}.json'
    if generation>1 and not previous.exists():raise ValueError('sequential prior generation required')
    if (ROOT/f'generation-{generation}.json').exists():raise ValueError('never overwrite executed generation')
    prior=json.loads(previous.read_text()) if previous.exists() else None
    seed_snapshot=json.loads((ROOT/'founders.json').read_text())
    populations={}
    for seed in plan['seeds']:
        for arm in ('llm','random'):
            key=f'{seed}/{arm}'
            populations[key]=CellPopulation.from_snapshot(prior['populations'][key] if prior else seed_snapshot)
    capfile=ROOT/'RESOURCE_CAPS.json'
    if not capfile.exists():
        founder=next(c for c in seed_snapshot['cells'].values() if c['state']=='live' and c['structure']=='neighbor_count')
        baseline=[]
        for _ in range(3):
            _,probe=measure_resources(lambda:scores(founder['structure'],founder['params']));baseline.append(probe)
        capfile.write_text(json.dumps(calibrate_caps(baseline),indent=2))
    cache=json.loads((ROOT/'cache.json').read_text()) if (ROOT/'cache.json').exists() else {}
    candidates=[]
    for seed in plan['seeds']:
        for arm in ('llm','random'):
            population=populations[f'{seed}/{arm}']
            # Generation feedback may nominate only an actually live parent.
            parent=next(cid for ids in population.live.values() for cid in ids if population.cells[cid]['structure']=='neighbor_count')
            for index,e in enumerate(entries):
                p=copy.deepcopy(e['params']);recipient=population.cells[parent]['params']
                if arm=='random':
                    rng=np.random.default_rng(seed+generation*10000+index*1000)
                    for name,v in p.items():
                        a=np.asarray(v,float);old=np.asarray(recipient[name],float) if name in recipient else None
                        mask=np.ones(a.shape,bool) if old is None else a!=old
                        new=rng.integers(-9,10,size=a.shape).astype(float)
                        if name=='patch.guard.density':new=rng.random(size=a.shape)
                        value=a.copy() if old is None else old.copy();value[mask]=new[mask];p[name]=value.tolist()
                cid=digest({'structure':e['structure'],'params':p})
                candidates.append({'context':f'{seed}/{arm}','seed':seed,'arm':arm,'slot':index,
                'structure':e['structure'],'params':p,'parent':parent,'id':cid})
    unique={c['id']:(c['structure'],c['params']) for c in candidates if c['id'] not in cache}
    batch_started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        results=list(pool.map(evaluate,unique.values()))
    batch_wall=time.perf_counter()-batch_started
    for cid,result in zip(unique,results):cache[cid]=result
    serial_wall=None
    if generation==1:
        timer=time.perf_counter();serial=list(map(evaluate,unique.values()));serial_wall=time.perf_counter()-timer
        if [r.get('scores') for r in serial]!=[r.get('scores') for r in results]:raise ValueError('parallel/serial prediction discrepancy')
    caps=json.loads(capfile.read_text())
    records=[];feedback=[]
    for c in candidates:
        pop=populations[c['context']];measured=cache[c['id']]
        if not measured['valid']:
            pop.charges.append({'arm':c['arm'],'child':c['id'],'charged':True})
            pop.events.append({'event':'invalid','cell':c['id'],'reason':measured['reason']})
            admission={'accepted':False,'invalid':True,'reasons':[measured['reason']]}
        else:
            authored=list(c['params']) if c['slot']==0 else [k for k in c['params'] if k not in pop.cells[c['parent']]['params']]
            provenance={'raw_response':Path(proposal_file).name,'model':response.get('model','gpt-6.1-sol'),'newly_authored':authored,
                        'typed_io':'Grid10 -> same-shape Grid10','generation':generation,'seed':c['seed'],
                        'operator':'LLM mutation' if c['arm']=='llm' else 'matched random mutation',
                        'failed_child_inspiration_not_parent':True,'shared_maker_response':c['arm']=='llm'}
            admission=pop.birth(c['structure'],c['params'],[c['parent']],provenance,measured['scores'],
                caps_passed=resource_pass(measured['resources'],caps),fixtures_passed=True,arm=c['arm'])
        record={k:v for k,v in c.items() if k!='params'}
        record.update(admission=admission,scores=measured.get('scores'),cached=c['id'] not in unique,
            ancestry_depth=depth(pop.cells,c['id']) if c['id'] in pop.cells else None)
        records.append(record)
    # Only development feedback. Include full live recipient and concrete expected/predicted failures.
    pop=populations[f"{plan['seeds'][0]}/llm"]
    for record in records[:2]:
        c=next(c for c in candidates if c['context']==record['context'] and c['slot']==record['slot'])
        failures=[]
        for row in PARTS['fresh']:
            for pair in row['task']['test']:
                prediction=predict(c['structure'],c['params'],pair['input'])
                if prediction!=pair['output']:failures.append({'task':row['id'],'demos':row['task']['train'],'input':pair['input'],'expected':pair['output'],'predicted':prediction})
        feedback.append({'candidate':c,'admission':record['admission'],'development_failures':failures})
    output={'generation':generation,'start_utc':started_utc,'pid':os.getpid(),
    'records':records,'populations':{k:p.snapshot() for k,p in populations.items()},
    'physical_author_responses':1,'seeds_are_contexts_not_independent_maker_samples':True,
    'timing':{'parallel_batch_seconds':batch_wall,'serial_verification_seconds':serial_wall,
              'server_stage_seconds':time.perf_counter()-started,'worker_count':plan['workers'],'new_unique_evaluations':len(unique),
              'charged_candidates':len(candidates),'cache_entries':len(cache),'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    'live_counts':{k:sum(map(len,p.live.values())) for k,p in populations.items()},
    'admitted':sum(r['admission']['accepted'] for r in records),'reporting_used':False}
    (ROOT/'cache.json').write_text(json.dumps(cache))
    (ROOT/f'generation-{generation}.json').write_text(json.dumps(output,indent=2))
    packet={'generation':generation,'feedback':feedback,'live_cells':[v for v in pop.cells.values() if v['state']=='live'],
      'request':'fresh literal numerical and structural proposals from live neighbor_count parent; rejected children are evidence only, never breeding parents; no reporting'}
    (ROOT/f'feedback-{generation}.json').write_text(json.dumps(packet,indent=2))
    print(json.dumps({'event':'generation-finished','generation':generation,'admitted':output['admitted'],'live_counts':output['live_counts'],'timing':output['timing']}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generation',type=int,required=True);parser.add_argument('--proposal',required=True)
    args=parser.parse_args()
    if args.generation not in (1,2):raise ValueError('bounded two-generation plan only')
    stage(args.generation,args.proposal)

