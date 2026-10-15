# SPDX-License-Identifier: MIT-0
"""One counted maker rewire versus one uniform rewire, same authored bank.

Selection uses synthetic demonstration-pair reconstruction only. Fresh synthetic
queries and exposed native DEV queries are scored after winner hashes freeze.
"""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np

from cellforge.construction.maker_operator import MakerOperator
from cellforge.governance.ledger import Ledger
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor, pack_grid
from cellforge.reference.grid_curriculum import FAMILIES, generate_task, classify_train_families
from cellforge.reference.grid_false_positives import score_with_consistency
from build_grid_curriculum_portfolio import slot_transaction, random_control_transaction
from measure_grid_curriculum import measure, digest, write


def fixtures(graph):
    rows = []
    for grid, phase in (([[0, 2, 0], [4, 0, 7]], 0.), ([[9]], 1.)):
        x, _, h, w = pack_grid(grid)
        rows.append({'entry': graph.entries['grid_step'],
                     'inputs': {'input': x.tolist(), 'input_height': h.tolist(), 'input_width': w.tolist(),
                                'output': x.tolist(), 'output_height': h.tolist(), 'output_width': w.tolist(),
                                'phase': [[phase]], 'state': np.zeros((1, graph.meta['state_dim'])).tolist()},
                     'output_shapes': {'state': [1, graph.meta['state_dim']], 'height_logits': [1, 2, 30],
                                       'width_logits': [1, 2, 30], 'color_logits': [1, 2, 30, 30, 10]}})
    return rows


def train_fitness(graph, groups):
    predictor = GridPredictor(graph, backend='compiled')
    start = time.perf_counter(); cpu = time.process_time(); rows = {}
    for family, tasks in groups.items():
        pairs = correct = failures = 0
        for task in tasks:
            for demo in task['train']:
                pairs += 1
                try:
                    predictions = predictor(task['train'], demo['input'])
                    correct += int(demo['output'] in predictions)
                except (ValueError, KeyError, IndexError, TypeError):
                    failures += 1
        rows[family] = {'tasks': len(tasks), 'demonstration_pairs': pairs,
                        'reconstructed_pairs': correct, 'pair_accuracy': correct / pairs, 'failures': failures}
    return {'partition': 'development', 'evidence': 'train-pairs', 'families': rows,
            'macro_pair_accuracy': sum(row['pair_accuracy'] for row in rows.values()) / len(rows),
            'wall_seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'rss_scope': 'whole process high-water, not incremental model memory'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--parent', required=True)
    p.add_argument('--private', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); out = Path(args.out)
    if out.exists():
        raise ValueError('fresh output required')
    parent = Graph.load(args.parent); out.mkdir(); ledger = Ledger(out / 'ledger.jsonl')
    policy = {'protocol': 'maker-grid-slot-pilot/1', 'parent_hash': parent.hash(),
              'proposal_budget_per_arm': 1, 'control_seed': 662021,
              'allowed_changes': 'one put_def rewire of typed slot9; same shared authored bank, cells, parameters, state capacity',
              'maker_choice': 'bank_local_fill.gene', 'selection_tasks_per_family': 100,
              'selection_seed_base': 731000, 'validation_seed_base': 1151000, 'validation_tasks_per_family': 100,
              'selection_fitness': 'macro exact reconstruction on all three demonstration pairs of each synthetic task; query targets excluded',
              'acceptance': 'strict macro improvement and no family regression against common founder',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'maker_model': 'existing Codex conversational assistant; exact serving model not recorded; no separate model calls',
              'maker_cost': 'not measured, not zero',
              'proposal_timing': 'manual maker wiring choice predeclared from prior published diagnostics; this is not a new adaptive model-service call after receiving this run diagnostics',
              'prior_context_exposure': 'assistant saw published synthetic baseline query aggregates and native DEV200 aggregates; those are not unexposed final data; fresh synthetic validation below is scored only after selection freeze',
              'gene_authoring_scope': 'the new fill gene is shared by both arms; this tests one wiring choice, not an isolated effect of LLM gene authoring',
              'audit_evaluations': 0, 'final_evaluations': 0, 'gradient_steps': 0}
    write(out / 'policy.json', policy); ledger.append('policy_frozen', policy, 'operator')
    groups = {family: [generate_task(family, 731000 + j * 10000 + i) for i in range(100)]
              for j, family in enumerate(FAMILIES)}
    baseline = train_fitness(parent, groups)
    write(out / 'founder-train-fitness.json', baseline); ledger.append('founder_train_scored', baseline, 'operator')
    diagnostics = {'partition': 'development', 'evidence': 'train-pairs',
                   'generator_source': 'cellforge.reference.grid_curriculum; original synthetic examples',
                   'baseline_demonstration_reconstruction': baseline,
                   'instruction': 'use a shared authored gene for the simultaneous zero-only four-neighbor fill family; preserve existing supported families',
                   'context_exposure_policy_sha256': hashlib.sha256((out / 'policy.json').read_bytes()).hexdigest()}
    transactions = {'maker': slot_transaction(parent, 'bank_local_fill.gene'),
                    'random_control': random_control_transaction(parent, policy['control_seed'])}
    def accept(metrics):
        return metrics['macro_pair_accuracy'] > baseline['macro_pair_accuracy'] and all(
            metrics['families'][family]['pair_accuracy'] >= baseline['families'][family]['pair_accuracy']
            for family in FAMILIES)
    frozen = {'founder': parent}; receipts = {}
    for arm, transaction in transactions.items():
        operator = MakerOperator(out / (arm + '-maker-ledger.jsonl'), budget=1, profile='ladder-micro/1')
        candidate, receipt = operator.submit(parent, diagnostics=diagnostics, ops=transaction['ops'],
            provenance={'actor': 'Codex root' if arm == 'maker' else 'uniform typed-slot sampler',
                        'model': policy['maker_model'] if arm == 'maker' else 'none (NumPy RNG)',
                        'method': 'manual-llm-maker' if arm == 'maker' else 'non-llm-mutation-control',
                        'transaction': transaction}, fixtures=fixtures(parent),
            evaluate=lambda graph: train_fitness(graph, groups), accept=accept)
        receipts[arm] = receipt
        if candidate is not None:
            candidate.save(out / (arm + '-candidate'))
        selected = candidate if receipt['status'] == 'accepted' else parent
        operator.freeze(selected); frozen[arm] = selected
        selected.save(out / (arm + '-selected'))
        ledger.append('proposal_attempt_complete', {'arm': arm, 'receipt': receipt}, 'operator')
        write(out / 'selection-progress.json', receipts)
    write(out / 'winners-frozen.json', {arm: graph.hash() for arm, graph in frozen.items()})
    ledger.append('all_winners_frozen', {arm: graph.hash() for arm, graph in frozen.items()}, 'operator')
    # No further selection after this point. Fresh validation is not fed back.
    validation = {family: [generate_task(family, 1151000 + j * 10000 + i) for i in range(100)]
                  for j, family in enumerate(FAMILIES)}
    selection_fingerprints = {digest({'train': task['train'], 'test': task['test']})
                              for tasks in groups.values() for task in tasks}
    fresh_fingerprints = [digest({'train': task['train'], 'test': task['test']})
                          for tasks in validation.values() for task in tasks]
    if len(set(fresh_fingerprints)) != 800 or selection_fingerprints.intersection(fresh_fingerprints):
        raise ValueError('fresh synthetic validation has duplicate I/O')
    directory = Path(args.private) / 'arcagi2/data/training'
    paths = [path for key, path in sorted((hashlib.sha256(x.stem.encode()).hexdigest(), x)
             for x in directory.glob('*.json')) if int(key, 16) % 5 != 0][:200]
    if len(paths) != 200:
        raise ValueError('native DEV subset unavailable')
    native = [json.loads(path.read_bytes()) for path in paths]
    annotations = [classify_train_families(task['train']) for task in native]
    ledger.append('native_annotations_frozen', {'sha256': digest(annotations), 'tasks': 200}, 'operator')
    rows = []; native_rows = []; cache = {}
    for arm, graph in frozen.items():
        if graph.hash() not in cache:
            predictor = GridPredictor(graph, backend='compiled'); family_rows = []
            for family, tasks in validation.items():
                row, _ = measure(predictor, tasks); row['family'] = family; family_rows.append(row)
            native_row = score_with_consistency([(str(i), task) for i, task in enumerate(native)], predictor)
            native_row['family_membership_denominators'] = {family: sum(family in labels for labels in annotations)
                for family in FAMILIES}
            native_row['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            cache[graph.hash()] = (family_rows, native_row)
        family_rows, native_row = cache[graph.hash()]
        for row in family_rows:
            rows.append(dict(row, arm=arm, hash=graph.hash(), stats=graph.stats(), split='fresh-synthetic-validation'))
        native_rows.append(dict(native_row, arm=arm, hash=graph.hash(), stats=graph.stats()))
    result = {'policy': policy, 'baseline_train_fitness': baseline, 'proposal_receipts': receipts,
              'frozen_winners': {arm: graph.hash() for arm, graph in frozen.items()},
              'validation': rows, 'native': native_rows,
              'limitations': 'one maker/control proposal only; shared gene authoring not isolated; earlier aggregate exposure disclosed; no statistical LLM advantage, population mechanism, synthetic-to-hidden transfer or frontier claim'}
    ledger.append('frozen_winners_measured', result, 'operator'); result['ledger_entries_verified'] = ledger.verify()
    write(out / 'result.json', result)
    print(json.dumps({'statuses': {arm: row['status'] for arm, row in receipts.items()},
                      'native': [{'arm': row['arm'], 'tasks_solved': row['tasks_solved']} for row in native_rows]}))


if __name__ == '__main__':
    main()
