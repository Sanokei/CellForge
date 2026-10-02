"""Diagnostic-only expert ceilings. No query target enters inference or selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def decode(out, attempts):
    result = []
    for rank in range(attempts):
        h = np.asarray(out['height_logits'])[0, rank].argmax() + 1
        w = np.asarray(out['width_logits'])[0, rank].argmax() + 1
        result.append(np.asarray(out['color_logits'])[0, rank, :h, :w].argmax(-1).tolist())
    return result


def probe(graph, view, backend='compiled'):
    """Accept ONLY demos and query input; observe extra branches without feedback.

    Each expert executes with its OWN demonstration-adapted state slice and
    mapping bundle. Oracle targets are deliberately absent from this API.
    """
    from cellforge.runtime.grid import pack_grid
    from cellforge.runtime.interp import Interpreter
    from cellforge.runtime.compiled import CompiledInterpreter
    if backend not in ('compiled', 'reference'):
        raise ValueError('unknown execution backend')
    if set(view) != {'train', 'input'}:
        raise ValueError('inference view must contain train and input only')
    if not view['train'] or any(set(d) != {'input', 'output'} for d in view['train']):
        raise ValueError('complete demonstration pairs required')
    root_name = graph.entries['grid_step']
    root = graph.defs[root_name]
    first = next(n for n in root['nodes'] if n['use'] == 'dispatch')
    base = CompiledInterpreter if backend == 'compiled' else Interpreter

    class Observer(base):
        query = False
        captured = None

        def run_dispatch(self, parent, node, ins, path):
            # Preserve the actual lazy dispatch result before observing extras.
            scores = np.asarray(ins['scores']).copy()
            payload = {k: v for k, v in ins.items() if k != 'scores'}
            actual = super().run_dispatch(parent, node, dict(ins), path)
            if self.query and parent == root_name and node['id'] == first['id']:
                experts = []
                for index, branch in enumerate(node['args']['branches']):
                    out = self.run_bundle(branch, payload, '/diagnostic/' + branch)
                    shaped = {k: np.asarray(out[k]) for k in
                              ('height_logits', 'width_logits', 'color_logits')}
                    expected = {'height_logits': (1, 1, 30), 'width_logits': (1, 1, 30),
                                'color_logits': (1, 1, 30, 30, 10)}
                    if any(shaped[k].shape != shape for k, shape in expected.items()):
                        raise ValueError('unexpected expert attempt dimensions')
                    if any(not np.isfinite(v).all() for v in shaped.values()):
                        raise ValueError('nonfinite expert output')
                    experts.append({'index': index, 'branch': branch,
                                    'score': float(scores[0, index]),
                                    'output': decode(shaped, 1)[0]})
                self.captured = experts
            return actual

    runner = Observer(graph)
    state = np.zeros((1, graph.meta['state_dim']))
    items = list(view['train']) + [{'input': view['input']}]
    for index, item in enumerate(items):
        runner.query = index == len(view['train'])
        x, xm, xh, xw = pack_grid(item['input'])
        y, ym, yh, yw = ((np.zeros_like(v) for v in (x, xm, xh, xw))
                          if runner.query else pack_grid(item['output']))
        out = runner.run_bundle(root_name, {
            'input': x, 'input_mask': xm, 'input_height': xh, 'input_width': xw,
            'output': y, 'output_mask': ym, 'output_height': yh, 'output_width': yw,
            'phase': np.array([[float(runner.query)]]), 'index': np.array([[float(index)]]),
            'state': state}, '/grid')
        state = out['state']
    return {'actual': decode(out, graph.meta['grid_attempts']),
            'experts': runner.captured,
            'physical_root_steps': len(items),
            'extra_expert_query_executions': len(runner.captured)}


def score_outputs(observation, target):
    """Scoring boundary: oracle labels consumed ONLY after outputs are frozen."""
    return {'actual': target in observation['actual'],
            'oracle': any(e['output'] == target for e in observation['experts']),
            'correct_experts': [e['index'] for e in observation['experts']
                                if e['output'] == target]}


def summarize(records, arms):
    rows = []
    union = [r['scores']['left']['oracle'] or r['scores']['right']['oracle'] for r in records]
    routed_union = [r['scores']['left']['actual'] or r['scores']['right']['actual'] for r in records]
    for arm in arms:
        actual = [r['scores'][arm]['actual'] for r in records]
        oracle = [r['scores'][arm]['oracle'] for r in records]
        rows.append({'arm': arm, 'tasks': len(records), 'actual_top2': sum(actual),
                     'own_expert_oracle': sum(oracle),
                     'routing_misses': sum(o and not a for o, a in zip(oracle, actual)),
                     'parent_union_capability_missing': sum(p and not o for p, o in zip(union, oracle)),
                     'capability_outside_parent_union': sum(o and not p for p, o in zip(union, oracle)),
                     'actual_outside_own_oracle': sum(a and not o for a, o in zip(actual, oracle))})
    if any(row['actual_outside_own_oracle'] for row in rows):
        raise ValueError('oracle bound violated; observer incomplete')
    parent_oracle = [sum(r['scores'][side]['oracle'] for r in records) for side in ('left', 'right')]
    return {'rows': rows, 'combined_parent_expert_oracle': sum(union),
            'combined_parent_actual_union_four_attempts': sum(routed_union),
            'complementarity_over_best_parent_expert_oracle': sum(union) - max(parent_oracle),
            'parent_union_tasks_missed_by_both_routers': sum(p and not a for p, a in zip(union, routed_union)),
            'limitations': ['oracle uses target labels and unlimited expert choice; diagnostic only',
                            'actual arms each emit two guesses; parent actual union emits four',
                            'all candidates frozen; no report-based best-seed selection',
                            'single synthetic corpus, not native ARC or independent general intelligence',
                            'coverage loss identifies missing capability, not internal weight damage',
                            'separate observer work changes cost; not matched runtime comparison']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=3497000)
    p.add_argument('--per-family', type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('fresh output directory required')
    sys.path[:0] = [str(args.repo / 'src'), str(args.repo / 'scripts')]
    from cellforge.ir.graph import Graph
    from cellforge.runtime.grid import GridPredictor
    # Import lazy execution dependencies before recording source commitments.
    from cellforge.runtime.compiled import CompiledInterpreter
    from expanded_grid_genetics import crossover, mutate, compatible
    from measure_grid_expanded_capacity import family
    inventory_dir = args.repo / 'runs/arc-grid-genes/expanded-breeding-v2'
    inventory = json.loads((inventory_dir / 'INVENTORY.json').read_text())
    parents = {}
    for side, name in [('left', 'parent-crop-fill'), ('right', 'parent-object-adjacency-scaling')]:
        graph = Graph.load(inventory_dir / name)
        expected = next(row['hash'] for row in inventory['artifacts'] if row['name'] == name)
        if graph.hash() != expected:
            raise ValueError('parent commitment mismatch')
        parents[side] = graph
    compatible(parents['left'], parents['right'])
    graphs = dict(parents)
    for seed in range(4):
        graphs[f'cross-{seed}'] = crossover(parents['left'], parents['right'], seed)
        graphs[f'mutation-{seed}'] = mutate(parents['left' if seed % 2 == 0 else 'right'], seed)
    hashes = {name: g.hash() for name, g in graphs.items()}
    # Freeze every already-imported local dependency, not unrelated work files.
    source_paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).resolve().is_relative_to(args.repo.resolve()):
            source_paths.add(Path(path).resolve())
    commitments = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    policy = {'protocol': 'oracle-routing-diagnostic/1', 'seed': args.seed,
              'per_family': args.per_family, 'operator_seeds': list(range(4)),
              'graph_hashes_before_corpus': hashes, 'source_hashes': commitments,
              'scope': 'diagnostic target-label oracle only; no model selection or router update',
              'native_data_access': False, 'training': False, 'actual_attempts_per_arm': 2,
              'mutation_parent_rule': 'even seeds left, odd seeds right; four proposals each operator',
              'oracle_inference_schema': ['train', 'input'], 'frozen_before_task_generation': True}
    args.out.mkdir(parents=True)
    (args.out / 'policy.json').write_text(json.dumps(policy, indent=2))
    (args.out / 'source_diagnostic.py').write_bytes(Path(__file__).read_bytes())
    for name, graph in graphs.items():
        if graph.save(args.out / 'candidates' / name) != hashes[name]:
            raise ValueError('candidate export commitment mismatch')
    tasks = family(args.seed, args.per_family)
    if len(tasks) > 24:
        raise ValueError('bounded diagnostic only')
    predictors = {name: GridPredictor(g, backend='compiled') for name, g in graphs.items()}
    records = []
    start = time.perf_counter()
    for index, task in enumerate(tasks):
        for query in task['test']:
            # Separate inference views; query answers are never passed to probe.
            view = {'train': task['train'], 'input': query['input']}
            observations = {name: probe(g, view) for name, g in graphs.items()}
            for name, g in graphs.items():
                # Independent ordinary adapter parity on every scored query.
                if observations[name]['actual'] != predictors[name](view['train'], view['input']):
                    raise ValueError('ordinary prediction parity failure')
                if g.hash() != hashes[name]:
                    raise ValueError('source graph changed')
            scores = {name: score_outputs(obs, query['output']) for name, obs in observations.items()}
            records.append({'index': index, 'family': task.get('family', task.get('metadata', {}).get('family')),
                            'task_hash': digest(task), 'scores': scores, 'observations': observations})
        print(json.dumps({'completed_tasks': index + 1, 'total_tasks': len(tasks)}), flush=True)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in commitments.items()):
        raise ValueError('dependency changed concurrently; reject result')
    (args.out / 'observations.json').write_text(json.dumps(records, indent=2))
    result = {'policy': policy, 'summary': summarize(records, list(graphs)),
              'records': records, 'wall_seconds': time.perf_counter() - start,
              'ordinary_parity_calls': len(records) * len(graphs),
              'source_unchanged': True, 'graph_unchanged': True,
              'corpus_hash': digest(tasks)}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2))
    lines = ['# Oracle routing diagnostic', '',
             'Target-label oracle is diagnostic only. Candidates were frozen before task generation.', '',
             '| Arm | Actual top-2 | Own expert oracle | Routing misses | Parent-union capability missing |',
             '|---|---:|---:|---:|---:|']
    for row in result['summary']['rows']:
        lines.append(f"| {row['arm']} | {row['actual_top2']} | {row['own_expert_oracle']} | {row['routing_misses']} | {row['parent_union_capability_missing']} |")
    lines += ['', f"Combined parent expert oracle: {result['summary']['combined_parent_expert_oracle']}/{len(records)}.",
              f"Complementarity over best individual parent expert oracle: {result['summary']['complementarity_over_best_parent_expert_oracle']} tasks.",
              f"Parent-union tasks missed by both real routers: {result['summary']['parent_union_tasks_missed_by_both_routers']}.", '',
              'No report-label choice of winner, training, deployment, or native benchmark access.',
              'A child oracle below the parent union means coverage lost during inheritance; an own-oracle/actual gap means routing misses.',
              'Small synthetic diagnostic, four frozen proposals per operator, no population search or causal superiority claim.',
              'Expert probes retain each graph’s own association state, mapping bundle, and router score. All ordinary outputs verified independently.',
              'Pairing experts with that state/update/scoring contract is required for future weight-level breeding; no fixed-slot router optimization here.', '']
    (args.out / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
