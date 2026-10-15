"""Oracle capacity-only inspection on already exposed public DEV; no policy/scorer calls."""
import datetime,hashlib,json,pathlib,sys
base=pathlib.Path(__file__).parent
read=lambda n:json.loads((base/n).read_text(encoding='utf-8'))
h=lambda n:hashlib.sha256((base/n).read_bytes()).hexdigest()
abi=read('ABI_FREEZE.json');genome=read('GENOME_FREEZE.json')
assert h('template/native_slots4.py')==abi['source_sha256'];assert h('GENOME.json')==genome['genome_sha256'];assert h('ABI_FREEZE.json')==genome['source_ABI_sha256']
sys.path.insert(0,str(base/'template'));import native_slots4 as N;import native_adapter as A
cases=read('SIX_INPUTS.json');answers={row['id']:row for row in read('SIX_ANSWERS.json')};rows=[]
for case in cases:
    public=A.candidate_input(case);surfaces=N.spans(public['request']);missing=[];represented=[]
    for annotation in answers[case['id']]['ground_truth']:
        for function,parameters in annotation.items():
            schema=next(f['parameters'] for f in public['functions'] if f['name']==function)
            for key in schema['required']:
                typed=N.typed_candidates(public['request'],schema['properties'][key],surfaces);found=[c for c in typed if any(type(c['value']) is type(value) and c['value']==value for value in parameters[key])]
                if found:represented.append({'parameter':key,'origin_types':sorted({c['kind'] for c in found})})
                else:missing.append({'parameter':key,'type':schema['properties'][key]['type']})
    rows.append({'case_id':case['id'],'all_required_annotation_values_represented':not missing,'missing_required_values':missing,'represented_origins':represented})
result={'schema':'exposed-DEV-capacity-only/4','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_frozen_before_inspection':True,'genome_frozen_before_inspection':True,'cohort_role':'EXPOSED_PUBLIC_DEV','oracle_annotation_access':True,'method':'Only enumerate input/schema-derived typed candidates and inspect required public annotations. No ranking, solve, packet, baseline or scorer calls.','per_case':rows,'all_six_required_value_sets_represented':all(row['all_required_annotation_values_represented'] for row in rows),'selected_policy_predictions':0,'solve_calls':0,'baseline_calls':0,'scored_packets':0,'accuracy_measured':False,'common_genome_joint_achievability_established':False,'qualified':False}
(base/'CAPACITY_DIAGNOSTIC.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({'all_six_required_value_sets_represented':result['all_six_required_value_sets_represented'],'selected_policy_predictions':0,'solve_calls':0,'baseline_calls':0,'accuracy_measured':False}))