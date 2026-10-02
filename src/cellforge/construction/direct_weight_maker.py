"""Explicit LLM-authored cell/router numeric proposals; no optimizer or LLM API."""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
from cellforge.construction.maker_operator import MakerOperator
from cellforge.ir.profiles import check_profile


def descendants(graph, roots):
    seen = set()
    def visit(name):
        if name in seen:
            return
        seen.add(name)
        for child in graph.children(name):
            visit(child)
    for root in roots:
        visit(root)
    return seen


def parameter_scopes(graph):
    reachable = descendants(graph, graph.entries.values())
    routers = descendants(graph, [n for n in reachable if graph.defs[n].get('role') == 'router'])
    experts = descendants(graph, [n for n in reachable if graph.defs[n].get('role') == 'expert'])
    scopes = {}
    for name in reachable:
        if graph.defs[name]['kind'] != 'cell':
            continue
        scope = 'shared' if name in routers and name in experts else 'router' if name in routers else 'cell'
        for param in graph.params_of(name):
            scopes.setdefault(param, set()).add(scope)
    return scopes


def attach_affine_router_cells(parent, prefix='direct_router', entry=None):
    """Identity-initialized trainable microcells over existing demo-fit scores.

    This is minimal numeric calibration capacity, not a contextual router.
    Each score remains paired with its expert index and existing state contract.
    """
    graph = parent.copy()
    entry = entry or ('step' if 'step' in graph.entries else 'grid_step')
    root = graph.defs[graph.entries[entry]]
    dispatch = next(n for n in root['nodes'] if n['use'] == 'dispatch')
    node_id = dispatch['in']['scores'].split('.')[0]
    router_name = next(n['use'] for n in root['nodes'] if n['id'] == node_id)
    router = graph.defs[router_name]
    if router.get('role') != 'router' or any(n.startswith(prefix) for n in graph.defs):
        raise ValueError('fresh namespace and graph-owned router required')
    branches = dispatch['args']['branches']
    previous = router['outputs']['scores']
    scores = {}
    for index, branch in enumerate(branches):
        name = f'{prefix}_{index}'
        graph.params[name + '.W'] = np.ones((1, 1))
        graph.params[name + '.b'] = np.zeros(1)
        graph.defs[name + '_cell'] = {'kind': 'cell', 'act': 'identity',
            'params': {'W1': name + '.W', 'b1': name + '.b'}}
        graph.defs[name + '_cluster'] = {'kind': 'bundle', 'role': 'cluster',
            'inputs': ['x'], 'nodes': [{'id': 'cell', 'use': name + '_cell', 'in': {'x': '$x'}}],
            'outputs': {'score': 'cell.y'}}
        router['nodes'] += [
            {'id': name + '_slice', 'use': 'op:slice', 'in': {'x': previous},
             'args': {'axis': 0, 'start': index, 'stop': index + 1}},
            {'id': name + '_score', 'use': name + '_cluster', 'in': {'x': name + '_slice'}}]
        scores[f'x{index}'] = name + '_score.score'
    router['nodes'].append({'id': prefix + '_scores', 'use': 'op:concat', 'in': scores, 'args': {'axis': 0}})
    router['outputs']['scores'] = prefix + '_scores'
    graph.meta['direct_weight_router'] = {'parent_hash': parent.hash(),
        'router_definition': router_name, 'expert_order': list(branches),
        'parameters': [f'{prefix}_{i}.{suffix}' for i in range(len(branches)) for suffix in ('W', 'b')],
        'initialization': 'identity affine score calibration',
        'limitations': 'no new contextual features; no performance gain claimed'}
    graph.validate()
    check_profile(graph, graph.meta.get('profile', 'ladder-micro/1'))
    return graph


class DirectWeightMaker:
    """A strict numeric path around the existing audited maker transaction API.

    A serialized LLM proposal supplies literal tensors, not fitted values.
    Invalid/no-change proposals consume budget. Acceptance remains fail-closed.
    Return next_parent explicitly so rejected candidate artifacts are not adopted.
    """
    def __init__(self, ledger, *, budget, profile='ladder-micro/1', max_parameters=None):
        self.operator = MakerOperator(ledger, budget=budget, profile=profile)
        self.max_parameters = max_parameters

    def submit(self, parent, proposal, *, diagnostics, fixtures, evaluate=None, accept=None, admission_parent=None):
        proposal = json.loads(json.dumps(proposal, allow_nan=False))
        provenance = {'actor': proposal.get('actor'), 'model': proposal.get('model'),
                      'method': 'manual-llm-maker', 'scope': proposal.get('scope'),
                      'proposal_sha256': hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest(),
                      'rationale': proposal.get('rationale'), 'authoring_cost': proposal.get('authoring_cost', 'unknown'),
                      'weight_changes': []}
        ops = proposal.get('ops')
        try:
            if proposal.get('base') != parent.hash():
                raise ValueError('exact proposal base hash required')
            if proposal.get('scope') not in ('cell', 'router', 'joint') or not proposal.get('rationale'):
                raise ValueError('cell/router/joint scope and rationale required')
            if self.max_parameters is not None and sum(v.size for v in parent.params.values()) > self.max_parameters:
                raise ValueError('bounded numeric-maker parameter cap exceeded')
            scopes = parameter_scopes(parent)
            if not isinstance(ops, list) or not 1 <= len(ops) <= 64:
                raise ValueError('bounded explicit numeric operations required')
            seen = set()
            for op in ops:
                if set(op) != {'op', 'name', 'value'} or op['op'] != 'set_param':
                    raise ValueError('this path requires literal set_param values, not slot edits or fitting')
                name = op['name']
                if name in seen or name not in scopes:
                    raise ValueError('unique reachable cell parameter required')
                seen.add(name)
                if proposal['scope'] != 'joint' and scopes[name] != {proposal['scope']}:
                    raise ValueError('parameter does not belong exclusively to requested scope')
                old = parent.params[name]
                value = np.asarray(op['value'], dtype=float)
                if value.shape != old.shape or not np.isfinite(value).all():
                    raise ValueError('same-shape finite numeric tensor required')
                changed = int(np.count_nonzero(value != old))
                if changed:
                    provenance['weight_changes'].append({'name': name, 'changed_scalars': changed,
                        'old': old.tolist(), 'new': value.tolist(), 'scopes': sorted(scopes[name])})
            if not provenance['weight_changes']:
                raise ValueError('no actual numeric weight change')
        except Exception as error:
            provenance['numeric_rejection'] = f'{type(error).__name__}: {error}'
            ops = None  # existing MakerOperator charges and records invalid submission
        original = parent.hash()
        candidate, receipt = self.operator.submit(parent, diagnostics=diagnostics, ops=ops,
            provenance=provenance, fixtures=fixtures, evaluate=evaluate, accept=accept,
            admission_parent=admission_parent)
        if parent.hash() != original:
            raise ValueError('parent mutated during callback')
        return {'candidate': candidate, 'receipt': receipt,
                'next_parent': candidate if receipt['status'] == 'accepted' else parent,
                'weight_changes': copy.deepcopy(provenance['weight_changes'])}

    def assemble(self, parent, proposal, *, diagnostics, fixtures, resolver=None, evaluate=None, accept=None, admission_parent=None):
        """LLM decides readiness and composition through ordinary typed IR ops.

        No prescribed pairing, cell count, graduation threshold or auto-assembly.
        The caller supplies the trained cell evidence and the LLM's explicit
        circuit/expert plan. This does not claim the author model itself trains.
        """
        proposal = json.loads(json.dumps(proposal, allow_nan=False))
        provenance = {'actor': proposal.get('actor'), 'model': proposal.get('model'),
                      'method': 'manual-llm-maker', 'scope': 'authored-assembly',
                      'rationale': proposal.get('rationale'),
                      'readiness_evidence': proposal.get('readiness_evidence'),
                      'proposal_sha256': hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()}
        ops = proposal.get('ops')
        allowed = {'set_param', 'put_def', 'set_entry', 'set_state_dim', 'import_def', 'wrap', 'unwrap'}
        if (proposal.get('base') != parent.hash() or not proposal.get('rationale')
                or not proposal.get('readiness_evidence') or not isinstance(ops, list)
                or not 1 <= len(ops) <= 64 or any(op.get('op') not in allowed for op in ops)):
            provenance['assembly_rejection'] = 'explicit same-base LLM readiness decision and bounded IR operations required'
            ops = None
        candidate, receipt = self.operator.submit(parent, diagnostics=diagnostics, ops=ops,
            provenance=provenance, fixtures=fixtures, resolver=resolver, evaluate=evaluate, accept=accept,
            admission_parent=admission_parent)
        return {'candidate': candidate, 'receipt': receipt,
                'next_parent': candidate if receipt['status'] == 'accepted' else parent}

    def compare_repairs(self, child, *, maker_proposal, random_ops, diagnostics,
                        fixtures, evaluate, admission_parent):
        """Three charged arms on one frozen child and paired development gates.

        Inputs are frozen before evaluation. This runs supplied proposals, not
        an LLM service, and leaves winner selection to an independent caller.
        """
        frozen = json.loads(json.dumps({'maker': maker_proposal, 'random': random_ops}, allow_nan=False))
        used = sum(row['kind'] == 'attempt' for row in self.operator.rows)
        if self.operator.budget - used < 3:
            raise ValueError('three attempt budget units required')
        results = {}
        for arm, ops in (('breeding_only', []), ('random_repair', frozen['random'])):
            candidate, receipt = self.operator.submit(child, diagnostics=diagnostics,
                ops=ops, provenance={'actor': arm, 'model': 'none',
                    'method': 'non-llm-mutation-control', 'arm': arm},
                fixtures=fixtures, evaluate=evaluate, evaluate_unchanged=True,
                admission_parent=admission_parent)
            results[arm] = {'candidate': candidate, 'receipt': receipt,
                'next_parent': candidate if receipt['status'] == 'accepted' else child}
        results['maker_repair'] = self.submit(child, frozen['maker'],
            diagnostics=diagnostics, fixtures=fixtures, evaluate=evaluate,
            admission_parent=admission_parent)
        return results

