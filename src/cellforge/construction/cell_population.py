"""Independent literal-weight cells with charged, traceable development selection."""
import copy
import json
import math
import re
from cellforge.construction.joint_weight_population import digest, inherit_neighbors
from cellforge.construction.red_queen_history import OpponentHistory


def validate_literal(params):
    def walk(value):
        if isinstance(value, list):
            if not value:
                raise ValueError('empty tensor')
            for item in value:
                walk(item)
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('finite literal numerical tensors required')
    if not isinstance(params, dict) or not params or any(k.startswith('slot') for k in params):
        raise ValueError('weight cells required')
    for value in params.values():
        walk(value)


def _membership(structure, provenance, scores, family=None, niche=None, niche_manifest=None):
    return {'family': family or provenance.get('family') or structure,
            'niche': niche or provenance.get('niche') or structure,
            'niche_manifest': copy.deepcopy(niche_manifest or provenance.get('niche_manifest') or
                                           {k: scores[k]['manifest'] for k in ('fresh', 'archive', 'anchor')})}


def _inheritance(recipient, donor, receipt):
    """Recompute supported numerical crossover; a receipt never supplies its own parents."""
    if not isinstance(receipt, dict):
        raise ValueError('two-parent birth requires numerical inheritance receipt')
    if receipt.get('recipient_hash') != digest(recipient) or receipt.get('donor_hash') != digest(donor):
        raise ValueError('inheritance must bind both actual parent tensors')
    mapping = receipt.get('mapping', {})
    if isinstance(mapping, dict) and 'recipient' in mapping and 'donor' in mapping:
        rkey, dkey = mapping['recipient'], mapping['donor']
        rprefix = rkey.rsplit(':', 1)[0] + ':' if ':' in rkey else ''
        dprefix = dkey.rsplit(':', 1)[0] + ':' if ':' in dkey else ''
        inherited, actual = inherit_neighbors(recipient, donor, recipient_prefix=rprefix,
                                             donor_prefix=dprefix, router_row=receipt.get('router_row_from_donor'))
        if actual != receipt:
            raise ValueError('inheritance receipt differs from executed crossover')
    elif isinstance(mapping, dict) and mapping.get('tensor_sources'):
        # Whole homologous tensor transfers are directly replayable, including arbitrary families.
        sources = mapping['tensor_sources']
        if set(sources) != set(recipient) or set(sources.values()) != {'recipient', 'donor'}:
            raise ValueError('crossover must retain numerical material from both parents')
        inherited = {}
        for key, source in sources.items():
            value = (recipient if source == 'recipient' else donor).get(key)
            if value is None:
                raise ValueError('missing inherited tensor')
            inherited[key] = copy.deepcopy(value)
        if receipt.get('child_hash') != digest(inherited):
            raise ValueError('inherited child hash mismatch')
    else:
        raise ValueError('unsupported numerical inheritance mapping')
    return inherited


def _scalar_paths(params):
    result = {}
    def walk(path, value):
        if isinstance(value, list):
            for i, item in enumerate(value):
                walk(path+'['+str(i)+']', item)
        else:
            result[path] = value
    for key, value in params.items():
        walk(key, value)
    return result


def require_retained_material(recipient, inherited, applied, receipt, newly_authored):
    """A mutation may edit inherited tensors but cannot erase either contributor."""
    old, base, child = map(_scalar_paths, (recipient, inherited, applied))
    donor_paths = [path for path in base if path in old and base[path] != old[path]]
    retained = receipt.get('recipient_retained', [])
    if not retained:
        retained = [key for key, source in receipt.get('mapping', {}).get('tensor_sources', {}).items()
                    if source == 'recipient']
    recipient_paths = [path for path in base if any(path == key or path.startswith(key+'[') for key in retained)]
    if not any(path in child and child[path] == base[path] for path in donor_paths):
        raise ValueError('post-crossover mutation erased all distinctive donor numerics')
    if not any(path in child and path in old and child[path] == old[path] for path in recipient_paths):
        raise ValueError('post-crossover mutation erased all retained recipient numerics')
    if not isinstance(newly_authored, list) or not newly_authored or any(not isinstance(p, str) for p in newly_authored):
        raise ValueError('explicit authored numerical positions required')
    changed = [path for path in set(base) | set(child) if base.get(path) != child.get(path)]
    if any(not any(path == key or path.startswith(key+'[') for key in newly_authored) for path in changed):
        raise ValueError('applied numerical mutation exceeds declared authored positions')

def verify_author_receipt(provenance, expected, verifier):
    """Verify an independently recorded invocation through a trusted host callback.

    The callback receives (receipt, expected_bindings), and must check the
    invocation commitment against the host's external recorder, not the run's
    self-asserted files. Missing host verification fails closed.
    """
    receipt = provenance.get('author_receipt')
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('author receipt must bind actual parent, prompt and raw response')
    if type(receipt.get('returncode')) is not int or receipt['returncode'] != 0:
        raise ValueError('successful actual author exit receipt required')
    if not isinstance(receipt.get('invocation_commitment'), str) or not re.fullmatch('[0-9a-f]{64}', receipt['invocation_commitment']):
        raise ValueError('independent author invocation commitment required')
    if not callable(verifier) or verifier(copy.deepcopy(receipt), copy.deepcopy(expected)) is not True:
        raise ValueError('trusted independent author invocation verification required')

class CellPopulation:
    def __init__(self, *, author_model='gpt-6.1-sol', author_reasoning='low', author_verifier=None):
        if author_model != 'gpt-6.1-sol' or author_reasoning != 'low':
            raise ValueError('population requires requested gpt-6.1-sol author at low reasoning')
        self.author_model = author_model; self.author_reasoning = author_reasoning
        self.author_verifier = author_verifier
        self.cells = {}; self.live = {}; self.histories = {}; self.events = []; self.charges = []
        self.competition_iteration = 0

    def seed(self, structure, params, provenance, scores, *, family=None, niche=None, niche_manifest=None):
        validate_literal(params)
        cid = digest({'structure': structure, 'params': params})
        if cid in self.cells:
            raise ValueError('duplicate founder')
        membership = _membership(structure, provenance, scores, family, niche, niche_manifest)
        history_key = structure if membership['niche'] == structure else structure + ':' + digest(membership['niche'])
        membership['history_key'] = history_key
        history = self.histories.setdefault(history_key, OpponentHistory())
        history.register(graph_hash=cid, family=membership['family'], structure_hash=digest(structure),
                         provenance=provenance, scores=scores)
        self.cells[cid] = dict(id=cid, structure=structure, params=copy.deepcopy(params), parents=[],
                              provenance=copy.deepcopy(provenance), scores=copy.deepcopy(scores), state='live', **membership)
        self.live.setdefault(structure, []).append(cid)
        self.events.append({'event': 'seed', 'cell': cid, **membership})
        return cid

    def birth(self, structure, params, parents, provenance, scores, *, caps_passed, fixtures_passed, arm,
              family=None, niche=None, niche_manifest=None):
        charge = {'arm': arm, 'parents': copy.deepcopy(parents), 'child': None, 'charged': True}
        self.charges.append(charge)
        try:
            if not parents or len(parents) > 2 or len(set(parents)) != len(parents):
                raise ValueError('one or two distinct known live parents required')
            for parent in parents:
                if parent not in self.cells or self.cells[parent]['state'] != 'live' or parent not in self.live.get(self.cells[parent]['structure'], []):
                    raise ValueError('every parent must be a registered live mating partner')
            recipient = self.cells[parents[0]]
            validate_literal(params)
            if len(parents) == 2:
                inherited = _inheritance(recipient['params'], self.cells[parents[1]]['params'], provenance.get('inheritance'))
                if arm.startswith('llm'):
                    if params == inherited:
                        raise ValueError('post-crossover LLM mutation must change inherited numerical tensors')
                    if provenance.get('model') != self.author_model or provenance.get('reasoning') != self.author_reasoning:
                        raise ValueError('post-crossover author must match requested model and reasoning')
                    prompt = provenance.get('prompt')
                    if not isinstance(prompt, dict) or prompt.get('mutation_parent_hash') != digest(inherited):
                        raise ValueError('post-crossover author prompt must bind inherited numerical child')
                    raw = provenance.get('raw_response')
                    if isinstance(raw, str):
                        raw = json.loads(raw)
                    if not isinstance(raw, dict) or raw.get('tensors') != params:
                        raise ValueError('post-crossover raw response must equal applied literal tensors')
                    require_retained_material(recipient['params'], inherited, params, provenance['inheritance'],
                                              provenance.get('newly_authored'))
                    verify_author_receipt(provenance, {
                        'model': self.author_model, 'reasoning': self.author_reasoning,
                        'parent_ids': list(parents), 'parent_hashes': [digest(self.cells[p]['params']) for p in parents],
                        'mutation_parent_hash': digest(inherited), 'prompt_hash': digest(prompt),
                        'response_hash': digest(raw)}, self.author_verifier)
                    if provenance.get('mutation_parent_hash') != digest(inherited):
                        raise ValueError('post-crossover mutation must bind inherited child hash')
                    if not all(provenance.get(k) for k in ('model', 'newly_authored', 'typed_io')):
                        raise ValueError('post-crossover mutation requires author and typed IO provenance')
                elif inherited != params:
                    raise ValueError('crossover child differs from inherited tensors')
            if structure != recipient['structure'] and arm.startswith('llm'):
                if not all(provenance.get(k) for k in ('raw_response', 'model', 'newly_authored', 'typed_io')):
                    raise ValueError('structural repair requires charged author and typed IO provenance')
            cid = digest({'structure': structure, 'params': params})
            charge['child'] = cid
            if cid in self.cells:
                self.events.append({'event': 'rejected', 'cell': cid, 'reason': 'duplicate'})
                return {'accepted': False, 'reasons': ['duplicate']}
            membership = _membership(structure, provenance, scores, family or recipient['family'],
                                     niche or recipient['niche'], niche_manifest or recipient['niche_manifest'])
            h = self.histories[recipient.get('history_key', recipient['structure'])]
            result = h.admission(recipient['scores'], dict(scores, caps_passed=caps_passed, fixtures_passed=fixtures_passed))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.events.append({'event': 'invalid', 'family': structure, 'reason': str(error)})
            return {'accepted': False, 'invalid': True, 'reasons': [str(error)]}
        self.cells[cid] = dict(id=cid, structure=structure, params=copy.deepcopy(params), parents=copy.deepcopy(parents),
                              provenance=copy.deepcopy(provenance), scores=copy.deepcopy(scores), state='born', **membership)
        self.events.append({'event': 'birth', 'cell': cid, 'parents': copy.deepcopy(parents), **membership})
        if result['accepted']:
            history_key = structure if membership['niche'] == structure else structure + ':' + digest(membership['niche'])
            self.cells[cid]['history_key'] = history_key
            target_history = self.histories.setdefault(history_key, OpponentHistory())
            target_history.register(graph_hash=cid, family=membership['family'], structure_hash=digest(structure),
                                    provenance=provenance, scores=scores)
            self.cells[cid]['state'] = 'live'
            self.live[recipient['structure']].remove(parents[0])
            recipient['state'] = 'retired'
            self.live.setdefault(structure, []).append(cid)
            self.events.extend([{'event': 'admitted', 'cell': cid}, {'event': 'retired', 'cell': parents[0]}])
        else:
            self.cells[cid]['state'] = 'rejected'
            self.events.append({'event': 'rejected', 'cell': cid, 'admission': result})
        return result

    def assemble(self, router, ids=None):
        ids = list(ids) if ids is not None else sorted(cid for values in self.live.values() for cid in values)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('nonempty distinct selected cells required')
        for cid in ids:
            if cid not in self.cells or self.cells[cid]['state'] != 'live' or cid not in self.live.get(self.cells[cid]['structure'], []):
                raise ValueError('only live cells cooperate')
        # Cross-family primitives may proceed to joint evaluation; this is candidate eligibility only.
        complementary = len({self.cells[c]['family'] for c in ids}) > 1
        if not complementary and any(not any(v for k in ('fresh', 'archive', 'anchor')
                                              for v in self.cells[c]['scores'][k]['outcomes'].values()) for c in ids):
            raise ValueError('assembly requires measured cell competence')
        validate_literal(router)
        if len(router['route.b']) != len(ids) or len(router['route.W']) != len(ids) or any(len(r) != len(ids) for r in router['route.W']):
            raise ValueError('router must align selected live cells')
        return {'cell_ids': ids, 'cells': [copy.deepcopy(self.cells[c]) for c in ids], 'router': copy.deepcopy(router),
                'router_cell_order': ids, 'admitted': False, 'status': 'candidate_requires_joint_evaluation'}

    def compete(self, *, environments, representatives=None, evaluator=None, status_root=None):
        """Execute frozen development tasks without retiring niche representatives.

        evaluator(cell, task) returns a prediction. Missing executors/tasks are
        explicitly unscored; these receipts establish no accuracy or ARC evidence.
        """
        before = digest(self.snapshot())
        families = {}
        for ids in self.live.values():
            for cid in ids:
                families.setdefault(self.cells[cid]['family'], []).append(cid)
        if not families or not environments:
            raise ValueError('live families and development environments required')
        if representatives is None:
            representatives = {f: max(ids, key=lambda c: (sum(self.cells[c]['scores']['fresh']['outcomes'].values()), c))
                               for f, ids in families.items()}
        if set(representatives) != set(families) or any(c not in families[f] for f, c in representatives.items()):
            raise ValueError('one live representative from every eligible family required')
        rows = []
        for environment in environments:
            if environment.get('partition') != 'development':
                raise ValueError('competition uses development environments only')
            for family, cid in representatives.items():
                cell = self.cells[cid]
                input_type = cell['provenance'].get('input_type', 'grid')
                row = {'family': family, 'cell': cid, 'environment': environment['id'],
                       'manifest': environment.get('manifest'), 'correct': None, 'scored_tasks': 0}
                if environment.get('input_type', input_type) != input_type:
                    row.update(status='inapplicable', reason='input type incompatible with registered cell')
                elif evaluator is None or not environment.get('tasks'):
                    row.update(status='unscored', reason='executable evaluator and frozen tasks required')
                else:
                    if not environment.get('manifest'):
                        raise ValueError('executed competition requires frozen task manifest')
                    task_ids = [task.get('id') for task in environment['tasks']]
                    if any(not isinstance(tid, str) or not tid for tid in task_ids) or len(set(task_ids)) != len(task_ids):
                        raise ValueError('distinct frozen competition task identities required')
                    if any('expected' not in task for task in environment['tasks']):
                        raise ValueError('target-separated competition expectations required')
                    outcomes = {}
                    for task in environment['tasks']:
                        view = copy.deepcopy({k: v for k, v in task.items() if k not in ('id', 'expected', 'target', 'answer', 'output', 'niche')})
                        prediction = evaluator(copy.deepcopy(cell), view)
                        outcomes[task['id']] = prediction == task['expected']
                    row.update(status='executed', outcomes=outcomes, correct=sum(outcomes.values()),
                               scored_tasks=len(outcomes), task_hash=digest(environment['tasks']))
                rows.append(row)
        if any(row['status'] == 'executed' and row['scored_tasks'] > 0 for row in rows):
            self.competition_iteration += 1
        result = {'representatives': copy.deepcopy(representatives), 'registry_hash_before': before,
                  'execution_receipts': rows, 'competition_iteration': self.competition_iteration,
                  'accepted': {row['family'] for row in rows if row['status'] == 'executed' and row['scored_tasks'] > 0} == set(families),
                  'admission': {row['family'] for row in rows if row['status'] == 'executed' and row['scored_tasks'] > 0} == set(families),
                  'status': 'executed' if any(row['status'] == 'executed' for row in rows) else 'diagnostic-unscored',
                  'scientific_acceptance': False}
        self.events.append({'event': 'competition', **copy.deepcopy(result)})
        return result

    @classmethod
    def from_snapshot(cls, snapshot):
        p = cls(author_model=snapshot.get('author_model', 'gpt-6.1-sol'),
                author_reasoning=snapshot.get('author_reasoning', 'low'))
        p.cells = copy.deepcopy(snapshot['cells']); p.live = copy.deepcopy(snapshot['live'])
        p.events = copy.deepcopy(snapshot['events']); p.charges = copy.deepcopy(snapshot['charges'])
        p.histories = {k: OpponentHistory(v) for k, v in snapshot['histories'].items()}
        p.competition_iteration = snapshot.get('competition_iteration', 0)
        seen = []
        for structure, ids in p.live.items():
            for cid in ids:
                if cid not in p.cells or p.cells[cid]['state'] != 'live' or p.cells[cid]['structure'] != structure:
                    raise ValueError('invalid resumable live population')
                seen.append(cid)
        if len(seen) != len(set(seen)) or set(seen) != {c for c, v in p.cells.items() if v['state'] == 'live'}:
            raise ValueError('live registry must match all and only live cells')
        for cell in p.cells.values():
            for key, value in _membership(cell['structure'], cell['provenance'], cell['scores']).items():
                cell.setdefault(key, value)
        return p

    def snapshot(self):
        return copy.deepcopy({'cells': self.cells, 'live': self.live, 'events': self.events, 'charges': self.charges,
                              'histories': {k: v.records for k, v in self.histories.items()},
                              'competition_iteration': self.competition_iteration,
                              'author_model': self.author_model, 'author_reasoning': self.author_reasoning})








