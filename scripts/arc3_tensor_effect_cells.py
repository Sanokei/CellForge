"""Fixed reusable tensor runtime. No maker-authored code or task identity is accepted."""
import copy
import hashlib
import inspect
import json
import pathlib
import numpy as np

SHAPES = {'effect': [8], 'recovery': [8], 'router': [4], 'rates': [2], 'click': [4]}

def validate_proposal(proposal):
    if not isinstance(proposal, dict) or set(proposal) != {'id', 'params'}:
        raise ValueError('numeric contract accepts only id and params; code/lookup fields forbidden')
    if proposal['id'] not in ('A', 'B'):
        raise ValueError('candidate identity must be A or B, never a task/sample key')
    params = proposal['params']
    if not isinstance(params, dict) or set(params) != set(SHAPES):
        raise ValueError('exact reusable tensor groups required; no identifiers or lookup tables')
    for key, shape in SHAPES.items():
        values = params[key]
        if not isinstance(values,list) or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) for v in values):
            raise ValueError('literal numeric scalars only: ' + key)
        arr = np.asarray(values)
        if arr.dtype.kind not in 'if' or list(arr.shape) != shape or not np.all(np.isfinite(arr)) or np.max(np.abs(arr)) > 20:
            raise ValueError('finite numeric tensor with declared shape/bounds required: ' + key)
    return {key: np.asarray(params[key], float).tolist() for key in SHAPES}

def features(stats, action, last, global_change):
    row = stats[action]
    return np.array([1.0, row[0], row[1], row[2] / (1.0 + row[2]), float(action == last), global_change, 1.0 / np.sqrt(1.0 + row[2]), row[3]])

def predict(params, feat):
    effect = 1.0 / (1.0 + np.exp(-np.clip(np.dot(params['effect'], feat), -30, 30)))
    noop = 1.0 / (1.0 + np.exp(-np.clip(np.dot(params['recovery'], feat), -30, 30)))
    return effect, noop

def act(params, mem, frame, avail, info):
    if 'stats' not in mem:
        mem['stats'] = np.zeros((max(avail) + 1, 4))
        mem['last'] = 0
        mem['global'] = 0.0
    stats = mem['stats']
    if max(avail) >= len(stats):
        grown = np.zeros((max(avail) + 1, 4))
        grown[:len(stats)] = stats
        stats = grown
        mem['stats'] = stats
    if 'prev' in mem:
        last = mem['last']
        changed = float(np.mean(frame != mem['prev']))
        noop = float(changed == 0.0)
        rates = 1.0 / (1.0 + np.exp(-np.clip(params['rates'], -20, 20)))
        stats[last, 0] = (1.0 - rates[0]) * stats[last, 0] + rates[0] * changed
        stats[last, 1] = (1.0 - rates[1]) * stats[last, 1] + rates[1] * noop
        stats[last, 2] += 1.0
        stats[last, 3] = noop * (stats[last, 3] + 1.0) / (2.0 + stats[last, 3])
        mem['global'] = (mem['global'] + changed) / 2.0
    scores = []
    for action in avail:
        feat = features(stats, action, mem['last'], mem['global'])
        effect, noop = predict(params, feat)
        route = np.array([effect, -noop, feat[6], -feat[2]])
        scores.append(float(np.dot(params['router'], route)))
    best = int(np.argmax(np.asarray(scores)))
    action = avail[best]
    step = info['step']
    click = params['click']
    x = int(np.abs(click[0] * step + click[1]) % 64)
    y = int(np.abs(click[2] * step + click[3]) % 64)
    mem['prev'] = frame.copy()
    mem['last'] = action
    return action, x, y

SOURCE = '\n\n'.join(inspect.getsource(f) for f in (features, predict, act))
RUNTIME_SHA256 = hashlib.sha256(SOURCE.encode()).hexdigest()

def arm_params(params, arm):
    result = copy.deepcopy(params)
    if arm == 'effect_off':
        result['effect'] = [0.0] * 8
    elif arm == 'recovery_off':
        result['recovery'] = [0.0] * 8
    elif arm == 'swap':
        result['effect'], result['recovery'] = result['recovery'], result['effect']
    return result

def seal_selection(root, candidates, scores, development_manifest, reporting_manifest):
    if set(development_manifest) & set(reporting_manifest):
        raise ValueError('development/reporting overlap')
    chosen = max(candidates, key=lambda key: (scores[key], key))
    seal = {'chosen': chosen, 'candidate_sha256': {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k,v in candidates.items()},
            'development': development_manifest, 'reporting': reporting_manifest, 'runtime_sha256': RUNTIME_SHA256,
            'selection_metric': 'development levels only; ties lexicographic, never reporting'}
    target = pathlib.Path(root) / 'SELECTION_SEALED.json'
    with target.open('x') as out:
        json.dump(seal, out, indent=2)
    return seal


def validate_release(raw, candidates, prompt, author_start, runtime_sha256):
    if runtime_sha256 != RUNTIME_SHA256:
        raise ValueError('undeclared inference runtime')
    if author_start.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError('prompt provenance mismatch')
    if author_start.get('model') != 'gpt-6.1-sol' or author_start.get('reasoning') != 'low' or author_start.get('auth') != 'ChatGPT subscription':
        raise ValueError('unverified maker route')
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError('missing literal author response')
    released = {p['id']: validate_proposal(p) for p in raw}
    if released != candidates:
        raise ValueError('saved weights differ from literal response')
    return True

def validate_revision(parent, child, parent_sha256, prompt_sha256, prompt=None):
    digest = hashlib.sha256(json.dumps(parent,sort_keys=True).encode()).hexdigest()
    if not isinstance(prompt,str) or not prompt.strip():
        raise ValueError('actual nonempty revision prompt required')
    prompt_digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    if parent_sha256 != digest or prompt_sha256 != prompt_digest:
        raise ValueError('parent or actual revision prompt provenance mismatch')
    new = validate_proposal({'id':'A','params':child})
    changes = sum(int(np.count_nonzero(np.asarray(parent[k]) != np.asarray(new[k]))) for k in SHAPES)
    if not changes:
        raise ValueError('no numerical weight changes')
    return changes
