import json,hashlib
from pathlib import Path
import numpy as np
from cellforge.construction.joint_weight_population import experts,digest
from cellforge.construction.typed_population import PaletteExecutor,parity_row_stencil
from cellforge.construction.component_count import SHAPES,predict_component_count
from cellforge.construction.patch_cell import PATCH_SHAPES,predict_patch
ROOT=Path.cwd();OLD=Path('/srv/cellforge-inbox/weights-real-learning-v2-20261001');V3=Path('/srv/cellforge-inbox/weights-real-learning-v3-20261001')
parent=json.loads((V3/'parent.json').read_text())['tensors']
palette=PaletteExecutor(OLD/'pop1_core.py',10);core=palette.core
core.PREDICTORS={n:(lambda p,g,trace=False,n=n:(palette.predict(n,p,g),{})) for n in core.PREDICTORS}
core.SHAPES['parity_row_stencil_v1']={'stencil.W':(2,3),'foreground.W':(1,10),'activation.b':(1,)}
core.PREDICTORS['parity_row_stencil_v1']=lambda p,g,trace=False:(parity_row_stencil(p,g),{})
core.SHAPES['component_count_unary_v1']=SHAPES
core.PREDICTORS['component_count_unary_v1']=lambda p,g,trace=False:(predict_component_count(p,g),{})
groups=experts(parent,core);conv_name=groups[2][1]
core.SHAPES['guarded_conv_patch_emit_v1']=dict(core.SHAPES[conv_name],**PATCH_SHAPES)
core.PREDICTORS['guarded_conv_patch_emit_v1']=lambda p,g,trace=False:(predict_patch(p,g,lambda pp,gg:palette.predict(conv_name,pp,gg)),{})

manifest=json.loads((V3/'PREDECLARED.json').read_text());rows=[]
for row in manifest['reserved_reporting']:
    raw=(Path('/srv/cellforge-private/arcagi2/data/training')/(row['id']+'.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('report hash mismatch')
    rows.append(dict(row,task=json.loads(raw)))
population=json.loads((ROOT/'population.json').read_text());scores={}
for cid,c in population['cells'].items():
    scores[cid]={r['id']:all(core.PREDICTORS[c['structure']](c['params'],pair['input'],False)[0]==pair['output'] for pair in r['task']['test']) for r in rows}
(ROOT/'REPORTING_TASKS.json').write_text(json.dumps(rows))
(ROOT/'cell-reporting.json').write_text(json.dumps(scores,indent=2))
print(json.dumps({'event':'frozen-cell-reporting','solved':{c:sum(v.values()) for c,v in scores.items()},'tasks':8}))
