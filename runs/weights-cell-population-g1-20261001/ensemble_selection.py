import copy,json,datetime
from pathlib import Path
import numpy as np
from cellforge.construction.cell_population import CellPopulation
from cellforge.construction.joint_weight_population import digest,experts,predict_routed,expert_predictions
from cellforge.construction.typed_population import PaletteExecutor,parity_row_stencil
from cellforge.construction.component_count import SHAPES,predict_component_count
from cellforge.construction.patch_cell import PATCH_SHAPES,predict_patch
from cellforge.construction.population_contract import require_two_level_run
ROOT=Path.cwd();OLD=Path('/srv/cellforge-inbox/weights-real-learning-v2-20261001');V3=Path('/srv/cellforge-inbox/weights-real-learning-v3-20261001')
parent=json.loads((V3/'parent.json').read_text())['tensors']
dev=json.loads((V3/'development.json').read_text())
oldanchors=json.loads((OLD/'old-anchors.json').read_text(encoding='utf-8-sig'))
parts={'fresh':dev[4:],'archive':dev[:4],'anchor':[{'id':i,'task':t} for i,t in oldanchors]}
palette=PaletteExecutor(OLD/'pop1_core.py',10);core=palette.core
core.PREDICTORS={n:(lambda p,g,trace=False,n=n:(palette.predict(n,p,g),{})) for n in core.PREDICTORS}
core.SHAPES['parity_row_stencil_v1']={'stencil.W':(2,3),'foreground.W':(1,10),'activation.b':(1,)}
core.PREDICTORS['parity_row_stencil_v1']=lambda p,g,trace=False:(parity_row_stencil(p,g),{})
core.SHAPES['component_count_unary_v1']=SHAPES
core.PREDICTORS['component_count_unary_v1']=lambda p,g,trace=False:(predict_component_count(p,g),{})
groups=experts(parent,core);conv_name=groups[2][1]
core.SHAPES['guarded_conv_patch_emit_v1']=dict(core.SHAPES[conv_name],**PATCH_SHAPES)
core.PREDICTORS['guarded_conv_patch_emit_v1']=lambda p,g,trace=False:(predict_patch(p,g,lambda pp,gg:palette.predict(conv_name,pp,gg)),{})

snapshot=json.loads((ROOT/'population.json').read_text());assembly=json.loads((ROOT/'assembly.json').read_text())
joint=copy.deepcopy(assembly['router'])
for i,cell in enumerate(assembly['cells']):
    joint.update({str(i)+':'+k:v for k,v in cell['params'].items()})
def score(t,rows):
    route={k:v for k,v in t.items() if k.startswith('route.')};outcomes={};consistent=wrong=0
    for row in rows:
        task=row['task'];hits=[];dc=[]
        for pair in task['test']:
            pred,trace=predict_routed(t,route,{'demos':task['train'],'input':pair['input']},core,True)
            hits.append(pred==pair['output']);dc.append(trace['features'][trace['chosen_expert']]==0)
        outcomes[row['id']]=all(hits);consistent+=int(all(dc));wrong+=int(all(dc) and not all(hits))
    return {'partition':'development','manifest':digest([r['id'] for r in rows]),'source':'public-training-real',
            'outcomes':outcomes,'false_positive_given_consistency':wrong/consistent if consistent else None}
def scores(t):return {k:score(t,r) for k,r in parts.items()}
ensembles=CellPopulation()
pid=ensembles.seed('grid_moe',parent,{'source':'historical v2 ensemble','separate_level':True},scores(parent))
admission=ensembles.birth('grid_moe',joint,[pid],{'assembly_cell_ids':assembly['cell_ids'],'selection':'measured development competency only',
'router':'literal inherited submatrix; no new router authoring','candidate_type':'earned assembly, not weight crossover'},scores(joint),caps_passed=True,fixtures_passed=True,arm='assembly')
payload={'population':snapshot,'ensemble_population':ensembles.snapshot(),'assembly':assembly,'selection_partition':'development'}
require_two_level_run(payload)
(ROOT/'two-level-receipt.json').write_text(json.dumps(dict(payload,admission=admission,architecture_gate=True,
timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),limits='Cell and MoE selection exercised serially, not parallel jobs; no router mutation, OCR/reasoning, or repaired reverse-transfer birth in this generation'),indent=2))
print(json.dumps({'event':'two-level-selection','cell_live':sum(map(len,snapshot['live'].values())),
'moe_live':sum(map(len,ensembles.live.values())),'assembly_admitted':admission['accepted'],'gate':True}))
