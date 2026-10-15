"""Development-only circuit admission earned by executed numerical comparisons."""
import copy
import hashlib
import random
from pathlib import Path
from .joint_weight_population import digest


def _forward(cell, bits):
    for layer in cell['layers']:
        bits = [int(sum(w*x for w, x in zip(row, bits))+bias > 0)
                for row, bias in zip(layer['W'], layer['b'])]
    return bits


def _zero(value):
    return [_zero(x) for x in value] if isinstance(value, list) else 0


def _topology(spec):
    result = copy.deepcopy(spec)
    for node in result.get('nodes', []):
        if node.get('kind') == 'glue':
            node['params'] = {k: _zero(v) for k, v in node['params'].items()}
    return result


def _source_hash():
    root = Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in
        ('circuit_population.py', 'adhesion.py', 'typed_population.py'))).hexdigest()


class CircuitArchive:
    def __init__(self, baseline, baseline_outcomes, founder_ids, *, founder_weights, development=None):
        self.development = copy.deepcopy(development)
        self.founder_ids = list(founder_ids)
        if (not self.founder_ids or len(set(self.founder_ids)) != len(self.founder_ids)
                or set(founder_weights) != set(self.founder_ids)):
            raise ValueError('exact independently selected founder identities and literal weights required')
        from .cell_population import validate_literal
        for params in founder_weights.values():
            # Typed definitions contain named ports as well as literal tensors.
            if isinstance(params, dict) and set(params) == {'inputs', 'outputs', 'layers'}:
                from .typed_population import validate_threshold_net
                validate_threshold_net(params)
            else:
                validate_literal(params)
        self.founder_weights = copy.deepcopy(founder_weights)
        self.live = digest({'spec': baseline, 'founders': self.founder_weights})
        self.records = {self.live: {'spec': copy.deepcopy(baseline),
            'outcomes': dict(baseline_outcomes), 'state': 'bootstrap-control', 'parents': []}}
        self.history = [self.live]
        self.events, self.charges = [], []

    def _execute(self, spec, catalogue, development, label):
        from .adhesion import AdhesionCandidate
        candidate = AdhesionCandidate(spec, catalogue, _forward)
        rows, outcomes = [], {}
        for case in development['cases']:
            try:
                execution = candidate.execute(copy.deepcopy(case['request']))
                correct = execution['packet'] == case['target']
                rows.append({'id': case['id'], 'correct': correct, 'execution': execution})
            except (ValueError, TypeError, KeyError) as error:
                correct = False
                rows.append({'id': case['id'], 'correct': False, 'error': str(error)})
            outcomes[case['id']] = correct
        receipt = {'label': label, 'partition': 'development', 'source': development['source'],
            'source_hash': _source_hash(), 'manifest_hash': digest(development),
            'weights_hash': digest({'spec': spec, 'founders': catalogue}),
            'typed_contract': candidate.receipt, 'outcomes': outcomes, 'rows': rows}
        receipt['receipt_hash'] = digest(receipt)
        return receipt

    def _measure(self, proposal, execution, current):
        if not isinstance(execution, dict) or set(execution) != {'development', 'random_seed', 'provenance'}:
            raise ValueError('actual development execution and matched controls required')
        dev = execution['development']
        if self.development is None or dev != self.development:
            raise ValueError('execution must match constructor-frozen development environment')
        if (not isinstance(dev, dict) or set(dev) != {'partition', 'source', 'manifest', 'cases'}
                or dev['partition'] != 'development' or dev['source'] != 'public-training-real'
                or not isinstance(dev['manifest'], str) or not dev['manifest']
                or not isinstance(dev['cases'], list) or not dev['cases']):
            raise ValueError('frozen real development manifest required; toy truth tables cannot earn architecture admission')
        ids = []
        for case in dev['cases']:
            if not isinstance(case, dict) or set(case) != {'id', 'request', 'target'}:
                raise ValueError('exact development input/target records required')
            if not isinstance(case['id'], str) or not case['id']:
                raise ValueError('development identity required')
            ids.append(case['id'])
        if len(set(ids)) != len(ids):
            raise ValueError('distinct development cases required')
        provenance = execution['provenance']
        if (not isinstance(provenance, dict) or not provenance.get('model')
                or provenance.get('reasoning') not in ('low', 'medium', 'high', 'xhigh', 'max', 'ultra')
                or not isinstance(provenance.get('prompt'), dict)
                or provenance['prompt'].get('parent_hash') != current
                or provenance['prompt'].get('founder_hash') != digest(self.founder_weights)
                or provenance['prompt'].get('development_manifest_hash') != digest(dev)
                or provenance.get('parent_hash') != current
                or provenance.get('raw_response') != {'spec': proposal['spec']}):
            raise ValueError('actual LLM prompt and exact raw numerical spec bound to current parent required')
        from .adhesion import validate_adhesion
        contract = validate_adhesion(proposal['spec'], self.founder_weights)
        if set(proposal['spec']['parents']) != set(self.founder_ids) or len(self.founder_ids) < 2:
            raise ValueError('all distinct contributing founders must execute')
        if proposal['new_numeric_positions'] != contract['new_numeric_positions'] or not contract['new_numeric_positions']:
            raise ValueError('exact charged authored glue positions required')
        seed = execution['random_seed']
        if type(seed) is not int:
            raise ValueError('explicit independent random control seed required')
        rng = random.Random(seed)
        random_spec = copy.deepcopy(proposal['spec'])
        def draw(value):
            return [draw(v) for v in value] if isinstance(value, list) else rng.uniform(-9, 9)
        for node in random_spec['nodes']:
            if node['kind'] == 'glue':
                node['params'] = {k: draw(v) for k, v in node['params'].items()}
        receipts = [self._execute(proposal['spec'], self.founder_weights, dev, 'candidate'),
                    self._execute(self.records[current]['spec'], self.founder_weights, dev, 'parent'),
                    self._execute(random_spec, self.founder_weights, dev, 'random')]
        zero_spec = _topology(proposal['spec'])
        zero_catalogue = copy.deepcopy(self.founder_weights)
        for cell in zero_catalogue.values():
            for layer in cell['layers']:
                layer.update({k: _zero(v) for k, v in layer.items()})
        receipts.append(self._execute(zero_spec, zero_catalogue, dev, 'zero'))
        for founder in self.founder_ids:
            cat = copy.deepcopy(self.founder_weights)
            for layer in cat[founder]['layers']:
                layer.update({k: _zero(v) for k, v in layer.items()})
            receipts.append(self._execute(proposal['spec'], cat, dev, 'ablate-cell:'+founder))
        for node in proposal['spec']['nodes']:
            if node['kind'] == 'glue':
                spec = copy.deepcopy(proposal['spec'])
                for altered in spec['nodes']:
                    if altered['id'] == node['id']:
                        altered['params'] = {k: _zero(v) for k, v in altered['params'].items()}
                receipts.append(self._execute(spec, self.founder_weights, dev, 'ablate-glue:'+node['id']))
        receipts[2]['random_seed'] = seed
        receipts[2]['receipt_hash'] = digest({k: v for k, v in receipts[2].items() if k != 'receipt_hash'})
        return receipts

    def propose(self, proposal, *, valid=None, dev_outcomes=None, contribution=None, arm, execution=None):
        # Caller claims are retained for diagnostics, never execution evidence.
        current = self.live
        self.charges.append({'arm': arm, 'positions': proposal.get('new_numeric_positions', 0),
            'charged': True, 'proposal': digest(proposal)})
        cid = digest({'spec': proposal['spec'], 'founders': self.founder_weights})
        reasons, receipts = [], []
        outcomes = {}
        try:
            receipts = self._measure(proposal, execution, current)
            outcomes = receipts[0]['outcomes']
            if any('error' in row for receipt in receipts[:2] for row in receipt['rows']):
                reasons.append('invalid_typed_execution')
            if receipts[1]['outcomes'] != self.records[current]['outcomes']:
                reasons.append('parent_outcomes_do_not_reproduce')
            count = sum(outcomes.values())
            for receipt in receipts[1:]:
                if count <= sum(receipt['outcomes'].values()):
                    reasons.append('unearned_'+receipt['label'])
            if set(outcomes) != set(self.records[current]['outcomes']):
                reasons.append('development_manifest_changed')
        except (ValueError, TypeError, KeyError) as error:
            reasons.append('missing_or_invalid_execution_evidence')
            reasons.append(str(error))
        claimed = outcomes if receipts else (dev_outcomes or {})
        if any(v and not claimed.get(k, False) for hid in self.history
               for k, v in self.records[hid]['outcomes'].items()):
            reasons.append('historical_development_regression')
        if receipts and sum(outcomes.values()) <= sum(self.records[current]['outcomes'].values()):
            reasons.append('strict_development_gain')
        if cid in self.records:
            reasons.append('duplicate')
        accepted = not reasons
        if cid not in self.records:
            self.records[cid] = {'spec': copy.deepcopy(proposal['spec']), 'outcomes': dict(outcomes),
                'state': 'admitted' if accepted else 'rejected',
                'parents': {'circuit': current, 'cells': list(self.founder_ids)},
                'new_weight_positions': proposal.get('new_numeric_positions', 0),
                'execution_receipts': receipts,
                'provenance': copy.deepcopy(execution.get('provenance')) if isinstance(execution, dict) else None}
        self.events.append({'event': 'proposal', 'id': cid, 'parent': current,
                            'accepted': accepted, 'reasons': reasons})
        if accepted:
            self.records[current]['state'] = 'retired'
            self.live = cid
            self.history.append(cid)
            self.events.append({'event': 'admitted', 'id': cid, 'retired': current})
        return {'accepted': accepted, 'reasons': reasons, 'child': cid, 'parent': current}

    def snapshot(self):
        return copy.deepcopy({'live': self.live, 'founder_weights': self.founder_weights,
            'records': self.records, 'development': self.development, 'historical_opponents': self.history,
            'events': self.events, 'charges': self.charges})
