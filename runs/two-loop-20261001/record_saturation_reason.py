import json,datetime,pathlib
b=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/runs')
for name in ('weights-arithmetic-subtraction-continuous','weights-spatial-hardlane'):
 r=b/name;v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actor':'Codex orchestration agent','reason':'saturation_pause','user_cancelled':False,'original_generic_status_preserved':True,'cause':'all three LLM survivors solve all declared development cases; avoid neutral rewrite spending','stopfile_created_by':'Codex orchestration agent','next_lane':'weights-hard-signed-deduction','frozen_reporting_accessed':False}
 p=r/'ORCHESTRATION_PAUSE_CORRECTION.json'
 if p.exists():raise ValueError('preserve existing correction')
 p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
print('two additive saturation-provenance corrections written; original STATUS unchanged')
