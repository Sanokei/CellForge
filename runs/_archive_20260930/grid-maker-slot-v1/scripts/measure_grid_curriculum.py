# SPDX-License-Identifier: MIT-0
"""Frozen graph measurement on original synthetic curricula and exposed ARC DEV.

Generators/annotations are evaluation-only. The graph receives demonstrations
and query pixels, never family labels, query targets or host-computed answers.
"""
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

from cellforge.governance.ledger import Ledger
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.arc_grid_operator import validate_grid
from cellforge.reference.grid_curriculum import FAMILIES, generate_task, classify_train_families


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def measure(predictor, tasks):
    tick = time.perf_counter(); cpu = time.process_time()
    solved = correct = invalid = inputs = node_execs = macs = 0
    masks = []; routing = {}
    for task in tasks:
        successes = []
        for query in task['test']:
            inputs += 1; stats = {}
            try:
                outputs = predictor(task['train'], query['input'])
                stats = predictor.last_exec or {}
                if not 1 <= len(outputs) <= 2:
                    raise ValueError('one or two attempts required')
                target = validate_grid(query['output'])
                success = any(validate_grid(x) == target for x in outputs)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                invalid += 1; success = False
            successes.append(success); correct += int(success)
            node_execs += stats.get('node_execs', 0); macs += stats.get('macs', 0)
            for site, choices in stats.get('route_counts', {}).items():
                totals = routing.setdefault(site, {})
                for choice, count in choices.items():
                    totals[choice] = totals.get(choice, 0) + count
        masks.append(all(successes)); solved += int(all(successes))
    wall = time.perf_counter() - tick; cpu = time.process_time() - cpu
    return {'tasks': len(tasks), 'tasks_solved': solved, 'solve_rate': solved / len(tasks),
            'test_inputs': inputs, 'test_inputs_correct': correct, 'invalid_outputs': invalid,
            'wall_seconds': wall, 'cpu_seconds': cpu, 'tasks_per_second': len(tasks) / wall,
            'cpu_seconds_per_task': cpu / len(tasks), 'node_execs': node_execs, 'macs': macs,
            'route_counts': routing, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'rss_scope': 'whole operator-process high-water to date, not incremental model allocation'}, masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', required=True); p.add_argument('--private', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--selection-per-family', type=int, default=200)
    p.add_argument('--validation-per-family', type=int, default=100)
    args = p.parse_args(); out = Path(args.out); inventory_path = Path(args.inventory)
    if out.exists() or not 1 <= args.selection_per_family <= 200 or not 1 <= args.validation_per_family <= 100:
        raise ValueError('fresh bounded output required')
    inventory = json.loads(inventory_path.read_text())
    models = [dict(entry, name=entry['family'] + '/' + entry['mode'])
              for entry in inventory['artifacts']]
    graphs = []
    for entry in models:
        graph = Graph.load(inventory_path.parent / entry['path'])
        if graph.hash() != entry['hash']:
            raise ValueError('frozen inventory changed')
        graphs.append(graph)
    directory = Path(args.private) / 'arcagi2/data/training'
    selected = [path for key, path in sorted((hashlib.sha256(path.stem.encode()).hexdigest(), path)
                for path in directory.glob('*.json')) if int(key, 16) % 5 != 0][:200]
    if len(selected) != 200:
        raise ValueError('expected exposed development subset unavailable')
    native = []; native_commit = hashlib.sha256(); labels = []
    for path in selected:
        raw = path.read_bytes(); task = json.loads(raw)
        native_commit.update(path.stem.encode() + b':' + hashlib.sha256(raw).digest())
        # Freeze labels from TRAIN PAIRS ONLY, before any query scoring.
        labels.append(classify_train_families(task['train']))
        native.append(task)
    synthetic = {}; seen = set(); family_seed_records = {}
    for split, count, base in (('selection', args.selection_per_family, 731000),
                               ('validation', args.validation_per_family, 941000)):
        synthetic[split] = {}
        for family_index, family in enumerate(FAMILIES):
            group = []; records = []
            for i in range(count):
                seed = base + family_index * 10000 + i
                task = generate_task(family, seed)
                fingerprint = digest({'train': task['train'], 'test': task['test']})
                if fingerprint in seen:
                    raise ValueError('duplicate synthetic I/O task within/across splits')
                seen.add(fingerprint); group.append(task)
                records.append({'seed': seed, 'task_sha256': fingerprint})
            synthetic[split][family] = group
            family_seed_records[split + '/' + family] = records
    out.mkdir(); ledger = Ledger(out / 'ledger.jsonl')
    source_paths = [Path(__file__), Path(__import__('cellforge.reference.grid_curriculum', fromlist=['x']).__file__)]
    policy = {'protocol': 'grid-dense-curriculum/1', 'models': models,
              'inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
              'source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
              'families': list(FAMILIES), 'synthetic_counts': {'selection': args.selection_per_family, 'validation': args.validation_per_family},
              'generator_seed_manifest_sha256': digest(family_seed_records),
              'native_tasks': 200, 'native_manifest_sha256': native_commit.hexdigest(),
              'native_family_annotation_sha256': digest(labels),
              'native_annotation': 'exact defined-generator membership on all TRAIN pairs only; overlapping families allowed, unclassified retained',
              'synthetic_exposure': 'original synthetic development selection/validation, disjoint seeds and exact I/O fingerprints; not sealed competition/final data',
              'model_access': 'only train demonstrations and query input; no family/target access',
              'selector_scope': 'authored reconstruction selector vs constant router, not learned amortized selector',
              'audit_evaluations': 0, 'final_evaluations': 0, 'gradient_steps': 0,
              'thread_environment': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}
    write(out / 'policy.json', policy); ledger.append('policy_frozen', policy, 'operator')
    write(out / 'original-synthetic-data.json', synthetic)
    write(out / 'generator-seeds.json', family_seed_records)
    (out / 'SYNTHETIC_DATA_LICENSE.txt').write_text(
        'SPDX-License-Identifier: CC0-1.0\n\n'
        'The original synthetic examples in original-synthetic-data.json are '
        'dedicated to the public domain under CC0 1.0 Universal.\n'
        'https://creativecommons.org/publicdomain/zero/1.0/\n\n'
        'No public ARC task contents are included in that synthetic file. '
        'This dedication does not apply to third-party ARC datasets, libraries '
        'or other artifacts. Competition eligibility is not established by this license.\n')
    rows = []; native_rows = []; started = time.perf_counter()
    for entry, graph in zip(models, graphs):
        predictor = GridPredictor(graph, backend='compiled')
        model = {'name': entry['name'], 'hash': graph.hash(), 'stats': graph.stats()}
        for split, groups in synthetic.items():
            for family, tasks in groups.items():
                row, _ = measure(predictor, tasks)
                row.update(model); row.update({'dataset': 'original synthetic', 'split': split, 'family': family})
                rows.append(row); ledger.append('synthetic_family_scored', row, 'operator')
                write(out / 'progress.json', {'synthetic': rows, 'native': native_rows})
        native_row, masks = measure(predictor, native)
        native_row.update(model); native_row.update({'dataset': 'ARC-AGI-2 public-training DEV', 'split': 'development'})
        by_family = {}
        for family in FAMILIES:
            indices = [i for i, annotation in enumerate(labels) if family in annotation]
            by_family[family] = {'tasks': len(indices), 'tasks_solved': sum(masks[i] for i in indices),
                                 'solve_rate': sum(masks[i] for i in indices) / len(indices) if indices else None}
        indices = [i for i, annotation in enumerate(labels) if not annotation]
        by_family['unclassified'] = {'tasks': len(indices), 'tasks_solved': sum(masks[i] for i in indices),
                                     'solve_rate': sum(masks[i] for i in indices) / len(indices) if indices else None}
        native_row['train_pair_family_membership'] = by_family
        native_rows.append(native_row); ledger.append('native_dev_scored', native_row, 'operator')
        write(out / 'progress.json', {'synthetic': rows, 'native': native_rows})
    result = {'policy': policy, 'synthetic': rows, 'native': native_rows,
              'seconds': time.perf_counter() - started,
              'limitations': 'frozen authored baselines only; no evolution or coevolving generators yet; synthetic/native family definitions differ in distribution; no hidden/final transfer or CPU saving presumed'}
    ledger.append('measurement_complete', result, 'operator')
    result['ledger_entries_verified'] = ledger.verify(); write(out / 'result.json', result)
    print(json.dumps({'models': len(models), 'synthetic_rows': len(rows), 'native_rows': len(native_rows), 'seconds': result['seconds']}))


if __name__ == '__main__':
    main()
