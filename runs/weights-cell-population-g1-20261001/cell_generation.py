"""Bounded independent cell generation; no held-out labels used in selection."""
import copy,json,datetime,os,time,hashlib
from pathlib import Path
import numpy as np
from cellforge.construction.cell_population import CellPopulation
from cellforge.construction.joint_weight_population import digest,experts,predict_routed,expert_predictions
from cellforge.construction.typed_population import PaletteExecutor,parity_row_stencil,measure_resources,calibrate_caps,resource_pass
from cellforge.construction.component_count import SHAPES,predict_component_count
from cellforge.construction.patch_cell import PATCH_SHAPES,predict_patch
ROOT=Path.cwd();OLD=Path('/srv/cellforge-inbox/weights-real-learning-v2-20261001');V3=Path('/srv/cellforge-inbox/weights-real-learning-v3-20261001')
start=time.perf_counter(); manifest=json.loads((V3/'PREDECLARED.json').read_text())
parent=json.loads((V3/'parent.json').read_text())['tensors'];dev=json.loads((V3/'development.json').read_text())
oldanchors=json.loads((OLD/'old-anchors.json').read_text(encoding='utf-8-sig'))
parts={'fresh':dev[4:],'archive':dev[:4],'anchor':[{'id':i,'task':t} for i,t in oldanchors]}
policy={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'model':'gpt-6.1-sol low configured task',
'cell_level':True,'two_level_required':True,'heldout_selection':False,'development_manifest':manifest['development'],
'reserved_reporting':manifest['reserved_reporting'],'budget':'one authored count, one authored conv structural repair, one aligned count-to-conv crossover; each has one matched random control; six charged children',
'limits':'six grid structural niches, not OCR/reasoning; assembly only measured successful cells; no new costs'}
(ROOT/'PREDECLARED.json').write_text(json.dumps(policy,indent=2))
print(json.dumps({'event':'start','pid':os.getpid(),'run':str(ROOT)}),flush=True)
palette=PaletteExecutor(OLD/'pop1_core.py',10);core=palette.core
core.PREDICTORS={n:(lambda p,g,trace=False,n=n:(palette.predict(n,p,g),{})) for n in core.PREDICTORS}
core.SHAPES['parity_row_stencil_v1']={'stencil.W':(2,3),'foreground.W':(1,10),'activation.b':(1,)}
core.PREDICTORS['parity_row_stencil_v1']=lambda p,g,trace=False:(parity_row_stencil(p,g),{})
core.SHAPES['component_count_unary_v1']=SHAPES
core.PREDICTORS['component_count_unary_v1']=lambda p,g,trace=False:(predict_component_count(p,g),{})
groups=experts(parent,core);conv_name=groups[2][1]
core.SHAPES['guarded_conv_patch_emit_v1']=dict(core.SHAPES[conv_name],**PATCH_SHAPES)
core.PREDICTORS['guarded_conv_patch_emit_v1']=lambda p,g,trace=False:(predict_patch(p,g,lambda pp,gg:palette.predict(conv_name,pp,gg)),{})
def predict(structure,params,g):return core.PREDICTORS[structure](params,g,False)[0]
def score(structure,params,rows):
    outcomes={};consistent=wrong=0
    for row in rows:
        task=row['task'];dc=all(predict(structure,params,p['input'])==p['output'] for p in task['train'])
        hits=[predict(structure,params,p['input'])==p['output'] for p in task['test']]
        outcomes[row['id']]=all(hits)
        consistent+=int(dc);wrong+=int(dc and not all(hits))
    return {'partition':'development','manifest':digest([r['id'] for r in rows]),
    'source':'public-training-real','outcomes':outcomes,'false_positive_given_consistency':wrong/consistent if consistent else None}
def scores(s,p):return {k:score(s,p,rows) for k,rows in parts.items()}
population=CellPopulation();ids=[]
for i,structure,p in groups:
    ids.append(population.seed(structure,p,{'source':'weights-real-learning-v2/verified-parent','joint_hash':digest(parent),'expert_index':i,'founder_status':'extracted historical LLM-authored cell, not new author call'},scores(structure,p)))
(ROOT/'FOUNDERS.json').write_text(json.dumps(population.snapshot(),indent=2))
baseline=[]
for _ in range(3):
    _,r=measure_resources(lambda:[scores(s,p) for _,s,p in groups]);baseline.append(r)
caps=calibrate_caps(baseline);(ROOT/'RESOURCE_CAPS.json').write_text(json.dumps(caps,indent=2))
author=json.loads((ROOT/'count-author.json').read_text(encoding='utf-8-sig'))
patch=json.loads((ROOT/'patch-author.json').read_text(encoding='utf-8-sig'))
conv=copy.deepcopy(groups[2][2]);patchparams=copy.deepcopy(conv)
for op in patch['ops']:patchparams[op['name'].split(':',1)[1]]=op['value']
cross=copy.deepcopy(conv);donor=groups[3][2]
for i in [0,1,2,3,5,6,7,8]:cross['conv.W'][i]=copy.deepcopy(donor['cnt.W'][0])
cross['conv.b']=copy.deepcopy(donor['cnt.b'])
proposals=[('llm_count',groups[3][1],author['params'],[ids[3]],{'raw_response':'count-author.json','model':author['model'],'newly_authored':list(author['params'])}),
('llm_patch','guarded_conv_patch_emit_v1',patchparams,[ids[2]],{'raw_response':'patch-author.json','model':patch['model'],'inherited':list(conv),'newly_authored':list(PATCH_SHAPES),'typed_io':'Grid10 -> same-shape Grid10'}),
('crossover',conv_name,cross,[ids[2],ids[3]],{'mapping':{'donor':'cnt.W[0]','recipient':'conv.W rows0,1,2,3,5,6,7,8','bias':'cnt.b -> conv.b'},'retained':['conv.W[4]','paint'],'no_new_glue':True})]
rng=np.random.default_rng(20261001)
for label,s,p,parents,prov in list(proposals):
    recipient=population.cells[parents[0]]['params'];random=copy.deepcopy(recipient)
    for key,v in p.items():
        arr=np.asarray(v);old=np.asarray(recipient[key]) if key in recipient else None
        mask=np.ones(arr.shape,bool) if old is None else arr!=old
        new=arr.copy() if old is None else old.copy()
        values=rng.integers(-9,10,size=arr.shape).astype(float)
        if key=='patch.guard.density':values=rng.random(size=arr.shape)
        new[mask]=values[mask];random[key]=new.tolist()
    proposals.append(('random_'+label,s,random,parents,{'control_for':label,'same_structure_and_changed_positions':True,'seed':20261001}))
results=[]
for arm,s,p,parents,prov in proposals:
    # Admission requires valid complete tensors; invalid proposals remain charged.
    if set(p)!=set(core.SHAPES[s]) or any(np.asarray(p[k]).shape!=shape for k,shape in core.SHAPES[s].items()):raise ValueError('invalid tensor contract')
    evidence,resource=measure_resources(lambda:scores(s,p))
    decision=population.birth(s,p,parents,prov,evidence,caps_passed=resource_pass(resource,caps),fixtures_passed=True,arm=arm)
    results.append({'arm':arm,'cell':digest({'structure':s,'params':p}),'structure':s,'scores':evidence,'admission':decision,'resources':resource})
    print(json.dumps({'event':'cell-evaluated','arm':arm,'fresh':sum(evidence['fresh']['outcomes'].values()),'accepted':decision['accepted']}),flush=True)
    (ROOT/'population.json').write_text(json.dumps(population.snapshot(),indent=2))
# Cooperation consumes only live cells with observed capability; keep separate ensemble competition.
eligible=[cid for cid in ids if any(v for section in population.cells[cid]['scores'].values() for v in section['outcomes'].values())]
assembly=None;ensembles=[]
if eligible:
    positions=[ids.index(cid) for cid in eligible];w=np.asarray(parent['route.W'])[np.ix_(positions,positions)].tolist();b=np.asarray(parent['route.b'])[positions].tolist()
    assembly=population.assemble({'route.W':w,'route.b':b},eligible)
    joint={}
    for n,cid in enumerate(eligible):
        joint.update({str(n)+':'+k:v for k,v in population.cells[cid]['params'].items()})
    joint.update(assembly['router'])
    def ensemble_score(t,rows):
        hits=oracle=uniform=0;route={k:v for k,v in t.items() if k.startswith('route.')}
        for row in rows:
            actual=[];oh=[];uh=[]
            for pair in row['task']['test']:
                view={'demos':row['task']['train'],'input':pair['input']}
                actual.append(predict_routed(t,route,view,core)==pair['output'])
                oh.append(any(p==pair['output'] for p in expert_predictions(t,pair['input'],core)))
                uh.append(predict_routed(t,{k:np.zeros_like(v).tolist() for k,v in route.items()},view,core)==pair['output'])
            hits+=all(actual);oracle+=all(oh);uniform+=all(uh)
        return {'actual':hits,'oracle':oracle,'uniform':uniform,'tasks':len(rows)}
    ensembles=[{'id':digest(parent),'kind':'historical_full_parent','development':ensemble_score(parent,dev)},
               {'id':digest(joint),'kind':'evidence_earned_assembly','development':ensemble_score(joint,dev)}]
    (ROOT/'assembly.json').write_text(json.dumps(assembly,indent=2))
    # Candidates already fixed. Reporting reserved earlier, evaluated now only for final reporting.
    data=Path('/srv/cellforge-private/arcagi2/data/training');report=[]
    for row in manifest['reserved_reporting']:
        raw=(data/(row['id']+'.json')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('report hash mismatch')
        report.append(dict(row,task=json.loads(raw)))
    for row,t in zip(ensembles,[parent,joint]):row['reporting']=ensemble_score(t,report)
snapshot=population.snapshot();live=sum(map(len,snapshot['live'].values()))
summary={'cells':results,'live_count':live,'live_by_structure':{k:len(v) for k,v in snapshot['live'].items()},
'births':sum(e['event']=='birth' for e in snapshot['events']),'admitted':sum(e['event']=='admitted' for e in snapshot['events']),
'rejected':sum(e['event']=='rejected' for e in snapshot['events']),'retired':sum(e['event']=='retired' for e in snapshot['events']),
'charged':len(snapshot['charges']),'eligible_assembly_cells':eligible,'ensembles':ensembles,
'two_level_complete':False,'remaining':'whole-ensemble candidates compared but no separate ensemble lineage/admission archive yet; router numeric mutation not justified by current author; OCR/reasoning absent',
'runtime_seconds':time.perf_counter()-start,'running':False}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'event':'finished','live':live,'births':summary['births'],'admitted':summary['admitted'],'seconds':summary['runtime_seconds']}),flush=True)

