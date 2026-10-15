"""Snapshot-bound numerical routing and separate development MoE competition."""
import copy
import json
import math
import numpy as np
from cellforge.construction.joint_weight_population import digest, predict_routed, expert_predictions
from cellforge.construction.red_queen_history import OpponentHistory


def literal_router(router, n):
    if set(router) != {'route.W', 'route.b'}:
        raise ValueError('complete literal router only; no slots or metadata')
    def leaves(v):
        if isinstance(v, list):
            return bool(v) and all(leaves(x) for x in v)
        return type(v) in (int, float) and math.isfinite(v)
    if not all(leaves(v) for v in router.values()):
        raise ValueError('finite literal numerical tensors required')
    w, b = np.asarray(router['route.W']), np.asarray(router['route.b'])
    if w.shape != (n, n) or b.shape != (n,):
        raise ValueError('router shape must align selected live cell order')
    return copy.deepcopy(router)


def assemble_snapshot(snapshot, assembly, core, parts):
    ids = assembly['cell_ids']
    if not ids or len(ids) != len(set(ids)) or ids != assembly['router_cell_order']:
        raise ValueError('explicit distinct ordered selected cells required')
    live = [cid for values in snapshot['live'].values() for cid in values]
    if len(live) != len(set(live)):
        raise ValueError('duplicate live cell membership')
    if [c['id'] for c in assembly['cells']] != ids:
        raise ValueError('assembly cells disagree with router order')
    joint = literal_router(assembly['router'], len(ids))
    for i, cid in enumerate(ids):
        c = snapshot['cells'][cid]
        if c['state'] != 'live' or cid not in live or c != assembly['cells'][i]:
            raise ValueError('only exact selected live snapshot cells may cooperate')
        if cid != digest({'structure': c['structure'], 'params': c['params']}):
            raise ValueError('cell hash mismatch')
        name, params = c['structure'], c['params']
        if name not in core.SHAPES or set(params) != set(core.SHAPES[name]):
            raise ValueError('unknown or slot-only cell contract')
        for key, shape in core.SHAPES[name].items():
            value = np.asarray(params[key])
            if value.shape != shape or value.dtype.kind not in 'fi' or not np.isfinite(value).all():
                raise ValueError('malformed literal cell')
        # Competence is executed again: copied flags cannot earn cooperation.
        earned = any(all(core.PREDICTORS[name](params, p['input'], False)[0] == p['output']
                         for p in row['task']['test'])
                     for rows in parts.values() for row in rows)
        if not earned:
            raise ValueError('selected cell has no executed development competence')
        joint.update({str(i).zfill(6)+':'+k: copy.deepcopy(v) for k,v in params.items()})
    return joint


def score_joint(joint, rows, core, *, partition='development', source='public-training-real'):
    router = {k:v for k,v in joint.items() if k.startswith('route.')}
    outcomes, details = {}, []
    consistent = false_positive = oracle_hits = first_hits = uniform_expected = 0
    physical = 0
    for row in rows:
        task = row['task']
        actual, oracle, first, demo_consistent, query_traces = [], [], [], [], []
        per_cell = None
        for pair in task['test']:
            view = {'demos':task['train'], 'input':pair['input']}
            prediction, trace = predict_routed(joint, router, view, core, True)
            predictions = expert_predictions(joint, pair['input'], core)
            physical += len(predictions) * (len(task['train']) + 2)
            matches = [p == pair['output'] for p in predictions]
            if per_cell is None: per_cell = [True] * len(matches)
            per_cell = [a and b for a,b in zip(per_cell,matches)]
            actual.append(prediction == pair['output'])
            oracle.append(any(matches))
            first.append(matches[0])
            demo_consistent.append(trace['features'][trace['chosen_expert']] == 0)
            query_traces.append(dict(trace, prediction=prediction, expected=pair['output'], oracle=any(matches)))
        outcomes[row['id']] = all(actual)
        consistent += int(all(demo_consistent))
        false_positive += int(all(demo_consistent) and not all(actual))
        oracle_hits += int(all(oracle)); first_hits += int(all(first))
        # True uniform single-expert selection, once per task, exact expectation.
        uniform_expected += sum(per_cell) / len(per_cell)
        details.append({'id':row['id'],'correct':all(actual),'queries':query_traces})
    return {'partition':partition,'manifest':digest(rows),'source':source,'outcomes':outcomes,
            'false_positive_given_consistency':false_positive/consistent if consistent else None,
            'oracle_query_upper_bound':oracle_hits,'zero_weight_first_cell':first_hits,
            'uniform_random_cell_expected_solves':uniform_expected,'traces':details,
            'physical_cell_predictions':physical}


def random_matched(parent, authored, seed):
    rng = np.random.default_rng(seed)
    out, positions = {}, {}
    for k,v in authored.items():
        old, new = np.asarray(parent[k],dtype=float), np.asarray(v,dtype=float)
        mask = old != new
        result = old.copy()
        sampled = rng.integers(-9,10,size=new.shape).astype(float)
        result[mask] = sampled[mask]
        out[k] = result.tolist()
        positions[k] = np.argwhere(mask).tolist()
    return out, positions


class MoECompetition:
    """A separate archive; scorer recomputes every candidate, including neutral edits."""
    def __init__(self, joint, core, parts, caps):
        self.core, self.parts, self.caps = core, copy.deepcopy(parts), caps
        self.parent = copy.deepcopy(joint)
        self.parent_scores = self.score(self.parent)
        self.history = OpponentHistory()
        self.history.register(graph_hash=digest(joint),family='grid_moe',structure_hash=digest(sorted(joint)),
                              provenance={'source':'selected live snapshot'},scores=self.parent_scores)
        self.events, self.charges = [], []
    def score(self, joint):
        return {k:score_joint(joint,rows,self.core) for k,rows in self.parts.items()}
    def propose(self, router, *, arm, provenance, resource_measure):
        import time
        self.charges.append({'arm':arm,'charged':True,'provenance':copy.deepcopy(provenance)})
        try:
            candidate = copy.deepcopy(self.parent)
            candidate.update(literal_router(router,len(self.parent['route.b'])))
            if arm == 'llm' and (not provenance.get('raw_response') or not provenance.get('prompt') or
                                provenance.get('response_tensors') != router):
                raise ValueError('actual author response and prompt must equal literal router')
            scores, resources = resource_measure(lambda:self.score(candidate))
            scores['caps_passed'] = all(resources[k] <= self.caps[k] for k in ('wall_seconds','traced_peak_bytes'))
            scores['fixtures_passed'] = True  # Complete numerical shape, execution and ordered snapshot checks.
            decision = self.history.admission(self.parent_scores,scores)
            # A gate that zero-weight deterministic selection replaces the router must fail.
            fresh = scores['fresh']
            if sum(fresh['outcomes'].values()) <= fresh['zero_weight_first_cell']:
                decision['accepted'] = False
                decision['reasons'].append({'gate':'router_zero_weight_no_loss'})
            result = {'hash':digest(candidate),'joint':candidate,'scores':scores,'resource':resources,
                      'admission':decision,'state':'admitted' if decision['accepted'] else 'rejected',
                      'parents':[digest(self.parent)],'arm':arm,'provenance':copy.deepcopy(provenance)}
            if decision['accepted']:
                self.history.register(graph_hash=result['hash'],family='grid_moe',structure_hash=digest(sorted(candidate)),
                                      provenance=provenance,scores=scores)
                self.parent, self.parent_scores = candidate, scores
        except (ValueError,KeyError,TypeError) as error:
            result = {'arm':arm,'state':'invalid','admission':{'accepted':False,'reasons':[str(error)]}}
        self.events.append(result)
        return result

