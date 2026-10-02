import json
import pathlib
import sys
import time
import hashlib
import numpy as np
import arc3_tensor_effect_cells as T
import arc3cell as C
import arc3_env as E
import maker as M

ROOT = pathlib.Path(sys.argv[1])
GAMES = ['ar25', 'lp85', 'sc25']
BUDGET = 300

def save(name, value):
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

ROOT.mkdir(exist_ok=False)
start = {'utc': M.now(), 'scope': 'two tensor-only reusable cells, bounded shared-policy pilot, not full ONE_STANDARD population',
         'maker_calls_cap':0 if len(sys.argv)>3 and sys.argv[2]=='runtime-replay' else 1, 'candidates': 2, 'positions_each': 26, 'games': GAMES, 'development_seed': 0, 'reporting_seed': 1,
         'actions_per_play': BUDGET, 'max_plays': 33, 'same_runtime_all_tasks': T.RUNTIME_SHA256,
         'reporting_limit': 'fresh seed, already public development games; not unseen-game transfer', 'selection': 'development levels only', 'contract_repair':len(sys.argv)>2 and sys.argv[2]=='contract-repair'}
save('START.json', start)
save('STANDARD.json', {'scope': start['scope'], 'qualified': False, 'limitations': ['single proposal batch', 'not full population competition', 'no crossover', 'public development games', 'not ONE_STANDARD qualification']})
(ROOT / 'RUNTIME.py').write_text(T.SOURCE)
prompt = '''Write literal numerical weights ONLY. Do not write code, programs, lookup tables, task IDs, sample IDs or explanations.
Return a JSON array with exactly two candidates {"id":"A","params":{...}} and {"id":"B","params":{...}}.
Fixed reusable model: two online transition cells and numeric action-selection router. This runtime is identical for every task.
Features in order: bias, action's EMA changed-pixel fraction, action's EMA no-op fraction, action exposure/(1+exposure), repeated last action indicator, global change EMA, 1/sqrt(1+exposure), consecutive no-op bounded trace.
Effect tensor shape[8] is sigmoid(dot(weights, features)) predicting useful observable action effect. Recovery tensor shape[8] predicts no-op probability the same way.
Router shape[4] scores [effect_prediction, -noop_prediction, exposure_uncertainty, -noop_EMA]. Argmax among legal actions.
Rates shape[2]: sigmoid(rate) controls action-conditioned EMA learning of changed-fraction/no-op outcomes. No gradient updates or hidden solver.
Click shape[4] sets generic coordinates abs(w0*step+w1)%64 and abs(w2*step+w3)%64. No frame/task-specific pixel constants.
All tensors finite numbers in [-20,20], exact keys effect,recovery,router,rates,click. 26 positions each. Propose two distinct meaningful hypotheses, not all-zero or duplicate.
Prior shared generic policy often repeated ineffective actions and earned zero levels. No game observations or target labels are supplied.
These cells must learn generic effects from observation/action/next-observation history and avoid ineffective actions while retaining exploration. Shape-8 effect/recovery must differ so swap ablation is meaningful.
'''
if len(sys.argv)>2 and sys.argv[2]=='contract-repair':
    prompt += '\nContract-only repair of preceding rejected batch: ALL coefficients, including click offsets, MUST lie in [-20,20]. Previous click offsets 29,23,41 violated this fixed bound. Do not relax bounds or return programs. No game evaluation feedback exists. This is one separate charged repair call, then stop.\n'
(ROOT / 'PROMPT.txt').write_text(prompt)
t0 = time.monotonic()
try:
    replay=len(sys.argv)>3 and sys.argv[2]=='runtime-replay'
    if replay:
        import shutil
        prior=pathlib.Path(sys.argv[3])
        shutil.copytree(prior/'AUTHOR',ROOT/'AUTHOR')
        prompt=(prior/'PROMPT.txt').read_text()
        (ROOT/'PROMPT.txt').write_text(prompt)
        raw=json.loads((ROOT/'AUTHOR'/'response.json').read_text())
        save('REPLAY.json',{'prior':str(prior),'new_maker_calls':0,'reason':'generic legal action ID range fix; identical frozen coefficients','already_exposed_reporting':True})
    else:
        raw = M.ask(prompt, ROOT / 'AUTHOR', track='arc3-tensor-effect-recovery', timeout=360, model='gpt-6.1-sol', effort='low')
    if not isinstance(raw, list) or len(raw) != 2 or {p.get('id') for p in raw} != {'A','B'}:
        raise ValueError('exactly A/B required')
    save('PROPOSAL_CHARGE.json',{'utc':M.now(),'proposals':0 if replay else 2,'positions_each':26,'invalid_candidates_still_charged':True,'literal_response':'AUTHOR/response.json'})
    candidates = {p['id']: T.validate_proposal(p) for p in raw}
    if candidates['A'] == candidates['B']:
        raise ValueError('duplicate candidates')
    author_start=json.loads((ROOT/'AUTHOR'/'START.json').read_text())
    literal=json.loads((ROOT/'AUTHOR'/'response.json').read_text())
    T.validate_release(literal,candidates,prompt,author_start,T.RUNTIME_SHA256)
    author_prompt=ROOT/'AUTHOR'/'PROMPT.txt'
    if author_prompt.exists() and author_prompt.read_text()!=prompt:
        raise ValueError('stale author prompt')
    optional_receipt=ROOT/'AUTHOR'/'RECEIPT.json'
    if optional_receipt.exists():
        receipt=json.loads(optional_receipt.read_text())
        if receipt.get('prompt_sha256')!=author_start['prompt_sha256'] or receipt.get('parent_sha256') is not None:
            raise ValueError('stale prompt or unexpected founder parent provenance')
    save('PREFLIGHT.json',{'utc':M.now(),'literal_response_verified':True,'prompt_verified':True,'fixed_runtime_verified':True,'founder_parent':None})
    save('CANDIDATES.json', candidates)
    for cid, weights in candidates.items():
        save('WEIGHTS_'+cid+'.json', weights)
    save('AUTHOR_RECEIPT.json', {'utc':M.now(), 'seconds':time.monotonic()-t0, 'charged_proposals':0 if replay else 2, 'literal_positions':0 if replay else 52,
                              'source':'AUTHOR/response.json', 'model':'gpt-6.1-sol', 'effort':'low', 'runtime_sha256':T.RUNTIME_SHA256,
                              'parent':'new founders; no inherited weights claimed', 'numeric_only':True})
    ok, why = C.validate(T.SOURCE)
    if not ok:
        raise ValueError('fixed runtime rejected: ' + why)
    rows=[]
    scores={}
    for cid, params in candidates.items():
        scores[cid]=0
        for game in GAMES:
            tick=time.monotonic()
            result=E.play(T.SOURCE, params, game, BUDGET, seed=0, wall=90)
            scores[cid]+=result['levels'] if not result['error'] else 0
            rows.append({'split':'development','candidate':cid,'arm':'full','evaluation_seconds':time.monotonic()-tick,'result':result})
            save('RESULTS.json', rows)
    if replay:
        chosen=json.loads((prior/'SELECTION_SEALED.json').read_text())['chosen']
        locked={cid:int(cid==chosen) for cid in candidates}
        seal=T.seal_selection(ROOT,candidates,locked,[g+':seed0' for g in GAMES],[g+':seed1' for g in GAMES])
        save('SELECTION_ORIGIN.json',{'prior_seal':str(prior/'SELECTION_SEALED.json'),'selected_before_runtime_replay':chosen,'no_reselection':True})
    else:
        seal=T.seal_selection(ROOT,candidates,scores,[g+':seed0' for g in GAMES],[g+':seed1' for g in GAMES])
    params=candidates[seal['chosen']]
    arms={'full':params, 'zero':C.zeros_like(params), 'effect_off':T.arm_params(params,'effect_off'),
          'recovery_off':T.arm_params(params,'recovery_off'), 'swap':T.arm_params(params,'swap')}
    for seed in (11,12,13): arms['random'+str(seed)]=C.random_like(params,seed)
    arms['blind']=params
    for arm, weights in arms.items():
        for game in GAMES:
            tick=time.monotonic()
            result=E.play(T.SOURCE,weights,game,BUDGET,seed=1,blind=arm=='blind',wall=90)
            rows.append({'split':'frozen_reporting','candidate':seal['chosen'],'arm':arm,'evaluation_seconds':time.monotonic()-tick,'result':result})
            save('RESULTS.json',rows)
    save('FINISH.json',{'utc':M.now(),'state':'completed','plays':len(rows),'maker_seconds':time.monotonic()-t0-sum(r['evaluation_seconds'] for r in rows),
                        'evaluation_seconds':sum(r['evaluation_seconds'] for r in rows), 'promoted':False,
                        'limitation':'pilot only; no full architecture/generalization/admission claim'})
except Exception as error:
    save('FAILURE.json',{'utc':M.now(),'error':repr(error),'charged_attempt':(ROOT/'AUTHOR'/'START.json').exists()})
    raise
