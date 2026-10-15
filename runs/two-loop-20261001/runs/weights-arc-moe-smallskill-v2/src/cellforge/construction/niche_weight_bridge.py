"""File-supervised niche proposer bridge for arbitrary graph artifacts.

Task adapters own domain semantics; every comparison starts from the same gene.
"""
import copy
import hashlib
import json
from pathlib import Path
import random

from .direct_weight_maker import DirectWeightMaker, parameter_scopes

SEEDS = (11, 22, 33)


def task_seed(seed, gene, round_index, purpose):
    return int(hashlib.sha256(json.dumps([seed, gene, round_index, purpose]).encode()).hexdigest()[:12], 16)


def numeric_proposal(parent, proposal):
    """Accept DirectWeightMaker ops or the original niche indexed-edit protocol."""
    if not isinstance(proposal, dict):
        proposal = {}
    result = dict(proposal)
    result.setdefault('base', parent.hash())
    result.setdefault('scope', 'joint')
    # Authorship must be supplied by the supervising human/LLM, never invented.
    if 'ops' not in result:
        try:
            tensors = {}
            for edit in result.get('edits', []):
                name = edit['param']
                value = tensors.setdefault(name, parent.params[name].copy())
                index = edit['index']
                if len(index) != value.ndim or any(type(i) is not int or not 0 <= i < n
                                                  for i, n in zip(index, value.shape)):
                    raise ValueError('invalid tensor index')
                value[tuple(index)] = edit['value']
            result['ops'] = [{'op': 'set_param', 'name': name, 'value': value.tolist()}
                             for name, value in tensors.items()]
        except (ValueError, KeyError, TypeError, IndexError):
            result['ops'] = None  # malformed edits still receive a charged receipt
    return result


def random_ops(parent, rng, count=1):
    names = sorted(parameter_scopes(parent))
    if not names:
        return []
    tensors = {}
    for _ in range(count):
        name = rng.choice(names)
        value = tensors.setdefault(name, parent.params[name].copy())
        if value.size:
            value.flat[rng.randrange(value.size)] = rng.choice((-2., -1., -.5, 0., .5, 1., 2.))
    return [{'op': 'set_param', 'name': n, 'value': v.tolist()} for n, v in tensors.items()]


def write_json(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(body, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def run_comparisons(root, genes, adapter, *, rounds=4, adapter_id='custom'):
    """Prepare outboxes and consume complete inboxes, safely resumable.

    adapter(graph, shown_seed, fresh_seed) returns diagnostics, fixtures,
    evaluate(graph)->aggregate metrics, accept(parent_metrics, child_metrics).
    The adapter must generate separate shown/fresh tasks and keep fresh labels
    out of diagnostics. Seeds are paired across all three arms.
    """
    root = Path(root)
    if type(rounds) is not int or rounds < 1 or not genes:
        raise ValueError('genes and positive rounds required')
    policy = {'seeds': list(SEEDS), 'rounds': rounds, 'adapter': adapter_id,
              'genes': {n: g.hash() for n, g in sorted(genes.items())},
              'arms': ['breeding_only', 'random_repair', 'maker_repair'],
              'charged_attempts_per_arm_per_gene_per_seed': rounds,
              'comparison': 'independent paired repairs of each frozen gene',
              'random_control': 'one random reachable numeric scalar edit per attempt',
              'source_branch': 'claude/redqueen-archive@61e6489'}
    policy_path = root / 'predeclared.json'
    if policy_path.exists():
        if json.loads(policy_path.read_text()) != policy:
            raise ValueError('predeclared experiment differs; use a fresh output directory')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('fresh experiment output required')
        write_json(policy_path, policy)
        for name, graph in genes.items():
            graph.save(root / 'genes' / name)
    pending = []
    for seed in SEEDS:
        for k in range(rounds):
            output = root / 'outbox' / str(seed) / f'round{k}.json'
            incoming = root / 'inbox' / str(seed) / f'round{k}.json'
            result_path = root / 'results' / str(seed) / f'round{k}.json'
            if result_path.exists():
                continue
            contexts = {}
            dump = {'seed': seed, 'round': k, 'targets': sorted(genes), 'niches': {}}
            for name, graph in sorted(genes.items()):
                contexts[name] = adapter(graph.copy(), task_seed(seed, name, k, 'shown'),
                                         task_seed(seed, name, k, 'fresh'))
                dump['niches'][name] = {'base': graph.hash(),
                    'champion_params': {n: v.tolist() for n, v in graph.params.items()},
                    'editable_parameters': sorted(parameter_scopes(graph)),
                    'diagnostics': contexts[name]['diagnostics'],
                    'counterexamples': contexts[name]['diagnostics'].get('counterexamples', [])}
            if output.exists():
                if json.loads(output.read_text()) != dump:
                    raise ValueError('adapter diagnostics changed on resume')
            else:
                write_json(output, dump)
            if not incoming.exists():
                pending.append(str(incoming))
                continue
            proposals = json.loads(incoming.read_text(encoding='utf-8'))
            if not isinstance(proposals, dict):
                raise ValueError('inbox must map target ids to proposals')
            # Freeze human-authored bytes before any evaluation.
            frozen_path = root / 'submitted' / str(seed) / f'round{k}.json'
            if frozen_path.exists():
                if json.loads(frozen_path.read_text()) != proposals:
                    raise ValueError('submitted inbox changed')
            else:
                write_json(frozen_path, proposals)
            rows = {}
            for name, graph in sorted(genes.items()):
                context = contexts[name]
                ledger = root / 'ledgers' / str(seed) / str(k) / f'{name}.jsonl'
                if ledger.exists():
                    raise ValueError('partial comparison detected; preserve receipts and start a fresh run')
                maker = DirectWeightMaker(ledger, budget=3, profile=graph.meta.get('profile', 'ladder-micro/1'))
                evaluate = context['evaluate']
                baseline = evaluate(graph.copy())
                accept = lambda metrics: context['accept'](baseline, metrics)
                results = {}
                for arm, ops in [('breeding_only', []), ('random_repair', random_ops(graph,
                        random.Random(task_seed(seed, name, k, 'random'))))]:
                    _, receipt = maker.operator.submit(graph, diagnostics=context['diagnostics'],
                        fixtures=context['fixtures'], ops=ops, evaluate=evaluate, accept=accept,
                        evaluate_unchanged=True, provenance={'actor': arm, 'model': 'none',
                        'method': 'non-llm-mutation-control', 'arm': arm})
                    results[arm] = receipt
                results['maker_repair'] = maker.submit(graph, numeric_proposal(graph, proposals.get(name)),
                    diagnostics=context['diagnostics'], fixtures=context['fixtures'],
                    evaluate=evaluate, accept=accept)['receipt']
                rows[name] = {'parent_hash': graph.hash(), 'baseline': baseline, 'arms': results,
                              'charged_per_arm': 1, 'fresh_seed': task_seed(seed, name, k, 'fresh')}
            write_json(result_path, rows)
    return {'pending_inboxes': pending, 'predeclared': str(policy_path)}
