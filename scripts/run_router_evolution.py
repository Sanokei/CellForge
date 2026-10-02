"""Bounded router-only pilot. No API, downloads, gradients or cell-state edits."""
import argparse, copy, datetime, hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
from cellforge.construction.joint_weight_population import digest
from cellforge.construction.typed_population import measure_resources, calibrate_caps
from cellforge.construction.population_contract import require_two_level_run
try:
    from cellforge.construction.router_evolution import assemble_snapshot, literal_router, score_joint, random_matched, MoECompetition
except ModuleNotFoundError:
    from router_evolution import assemble_snapshot, literal_router, score_joint, random_matched, MoECompetition

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(v,f,indent=2,allow_nan=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--author',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    source=args.repo/'runs/weights-cell-population-g1'
    author=read(args.author);proposal=author['router_mutation'];seeds=[101,202,303]
    a=read(source/'assembly.json');p=read(source/'population.json');dev=read(source/'development.json')
    anchors=read(args.repo/'runs/weights-real-learning-v2/old-anchors.json')
    parts={'fresh':dev[4:],'archive':dev[:4],'anchor':[{'id':i,'task':t} for i,t in anchors]}
    if proposal['cell_order'] != a['cell_ids']:raise ValueError('author router cell order mismatch')
    if author['model'] != 'gpt-6.1-sol' or not author['prompt'] or author['provenance']['fresh_report_access']:
        raise ValueError('actual permitted author provenance required')
    input_paths=[source/'assembly.json',source/'population.json',source/'development.json',args.author,
                 args.repo/'runs/weights-real-learning-v2/old-anchors.json',source/'pop1_core.py',source/'cell_eval.py',Path(__file__),
                 args.repo/'src/cellforge/construction/router_evolution.py',
                 args.repo/'src/cellforge/construction/joint_weight_population.py',
                 args.repo/'src/cellforge/construction/typed_population.py',
                 args.repo/'src/cellforge/construction/red_queen_history.py',
                 args.repo/'src/cellforge/construction/repair_acceptance.py']
    write(root/'PREDECLARED.json',{'seeds':seeds,'before_candidate_execution':True,'proposal_budget_per_arm_per_seed':1,
          'physical_maker_responses':1,'independent_maker_replicates':False,'development_only':True,
          'plan':'one supplied actual scale router trial, one matched random edit per seed; zero-weight and true uniform expectation and oracle diagnostics',
          'selection':'strict fresh gain; per-task fresh/archive/anchor and false-positive nonregression; caps; history; zero-weight ablation',
          'inputs':{str(f):sha(f) for f in input_paths},'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'limitations':['positive-scale trial preserves routing; no new founder authoring','author source says gpt-6.1-sol; effort setting not embedded in raw author artifact','no reserved reporting read or scored','three random contexts share one maker response']})
    frozen_sources=root/'source';frozen_sources.mkdir()
    for f in input_paths:
        if f.suffix=='.py':
            with (frozen_sources/f.name).open('xb') as target:target.write(f.read_bytes())
    for name,value in [('assembly.json',a),('population.json',p),('development-parts.json',parts),('author.json',author)]:write(root/name,value)
    spec=importlib.util.spec_from_file_location('router_frozen_cell_eval',source/'cell_eval.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);core=m.core
    physical_calls={'cell_predictions':0}
    def counted(predictor):
        def invoke(*args,**kwargs):
            physical_calls['cell_predictions']+=1
            return predictor(*args,**kwargs)
        return invoke
    core.PREDICTORS={k:counted(v) for k,v in core.PREDICTORS.items()}
    joint=assemble_snapshot(p,a,core,parts);authored=literal_router(proposal['tensors'],len(a['cell_ids']))
    (_,r1)=measure_resources(lambda:{k:score_joint(joint,v,core) for k,v in parts.items()})
    (_,r2)=measure_resources(lambda:{k:score_joint(joint,v,core) for k,v in parts.items()})
    (baseline,r3)=measure_resources(lambda:{k:score_joint(joint,v,core) for k,v in parts.items()})
    caps=calibrate_caps([r1,r2,r3]);write(root/'RESOURCE_CAPS.json',caps);write(root/'parent-scores.json',baseline)
    weights=root/'weights';weights.mkdir();parent_hash=digest(joint);write(weights/(parent_hash+'.json'),{'tensors':joint})
    records=[];genomes=[];ledger=[];previous=None;histories={}
    for seed in seeds:
        tournaments={}
        random,positions=random_matched(a['router'],authored,seed)
        prov={'model':author['model'],'actor':author['actor'],'raw_response':'author.json','prompt':author['prompt'],
              'response_tensors':authored,'response_sha256':sha(root/'author.json'),'rationale':proposal['rationale'],
              'cell_order':proposal['cell_order'],'physical_response':1,'fresh_report_access':False}
        for arm,router,pr in [('llm',authored,prov),('random',random,{'seed':seed,'control_for':'authored scale trial','positions':positions,'vocabulary':[-9,9]})]:
            tournament=MoECompetition(joint,core,parts,caps)
            tournaments[arm]=tournament
            result=tournament.propose(router,arm=arm,provenance=pr,resource_measure=measure_resources)
            result['seed']=seed;result['edited_positions']=positions
            records.append(result)
            if 'joint' in result:
                h=result['hash'];path=weights/(h+'.json')
                if not path.exists():write(path,{'tensors':result['joint']})
                genomes.append({'hash':h,'niche':'selected_live_grid_moe','origin':'llm_mutation' if arm=='llm' else 'random_mutation',
                    'parents':[parent_hash],'weights_file':'weights/'+h+'.json','accuracy':sum(result['scores']['fresh']['outcomes'].values())/len(parts['fresh']),
                    'accuracy_partition':'development','generation':1,'arm':arm,'seed':seed})
            event={'arm':arm,'seed':seed,'hash_or_null':result.get('hash'),'charged':True,'outcome':result['state'],
                   'edited_scalar_count':sum(len(x) for x in positions.values()),'full_router_scalar_count':6,'previous':previous}
            previous=digest(event);event['hash']=previous;ledger.append(event)
        histories[str(seed)]={arm:t.history.records for arm,t in tournaments.items()}
    write(root/'candidate-receipts.json',records);write(root/'moe-opponent-history.json',histories)
    payload={'population':p,'assembly':a,'ensemble_population':{'histories':histories,'events':[{'event':'birth','child':r.get('hash'),'state':r['state']} for r in records],'charges':[{'arm':r['arm'],'seed':r['seed'],'charged':True} for r in records]},'selection_partition':'development'}
    require_two_level_run(payload)
    write(root/'two-level-execution.json',dict(payload,gate=True,scope='Executed router proposals over historical independently competed selected live cells; no scientific-gain claim'))
    with (root/'ledger.jsonl').open('x',encoding='utf-8') as f:
        for event in ledger:f.write(json.dumps(event,sort_keys=True)+'\n')
    write(root/'STANDARD.json',{'arms':{'llm':{'maker':'actual supplied offline response'},'random':{'same_edited_positions':True}},
        'seeds':seeds,'seeds_declared_before_run':True,'budget_per_arm':1,'niches':['selected_live_grid_moe'],
        'founders':[{'hash':parent_hash,'niche':'selected_live_grid_moe','author':'llm','actor':'historical selected-cell founders',
                     'rationale':'inherited measured eligible cells and router; no new author call','weights_file':'weights/'+parent_hash+'.json'}],
        'genomes':genomes,'proposals':[{k:e[k] for k in ('arm','seed','hash_or_null','charged','outcome')} for e in ledger],
        'status':'bounded development-only router pilot; scientific milestone incomplete','physical_maker_responses':1,
        'independent_maker_replicates':False,'ledger_file':'ledger.jsonl','limitations':'No new founder routers, independent maker seeds, whole-MoE crossover, native fresh gain, reserved report, OCR or reasoning. This STANDARD cannot pass the full semantic/scaling milestone.'})
    summary={'selected_cells':a['cell_ids'],'parent':{k:{'solves':sum(v['outcomes'].values()),'tasks':len(v['outcomes']),
              'oracle_query_upper_bound':v['oracle_query_upper_bound'],'zero_weight_first_cell':v['zero_weight_first_cell'],
              'uniform_random_cell_expected_solves':v['uniform_random_cell_expected_solves']} for k,v in baseline.items()},
        'candidates':[{'arm':r['arm'],'seed':r['seed'],'state':r['state'],'admission':r['admission'],
                       'solves':{k:sum(r['scores'][k]['outcomes'].values()) for k in parts} if 'scores' in r else None} for r in records],
        'promoted':any(r['state']=='admitted' for r in records),'scientific_gain':False,'physical_author_responses':1,
        'scientific_status':'failure: no strict fresh router gain; scale mutation is behaviorally neutral',
        'actual_physical_cell_prediction_calls_including_competence_caps_parent_and_candidates':physical_calls['cell_predictions'],
        'source_snapshot_hash':sha(root/'population.json'),'selection_partition':'development'}
    write(root/'summary.json',summary)
    files={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()}
    write(root/'INTEGRITY.json',{'files':files,'ledger_head':previous,'immutable_policy':'exclusive creation; refuse output directory reuse; verify file hashes before reuse'})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()



