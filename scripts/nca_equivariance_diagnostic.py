"""Training-only locality lower bounds for shared, masked 3x3 NCA updates.

No weights, predictions, optimization, model calls or new task data. Equal
radius-r input/mask neighborhoods force equal outputs after at most r updates.
Contradictory labels prove impossibility; unique signatures prove nothing.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from scripts.nca_capacity_diagnostic import load_frozen_panel, validate_task


def signature(grid, y, x, radius):
    if type(radius) is not int or radius < 0:
        raise ValueError('radius must be a nonnegative integer')
    h, w = len(grid), len(grid[0])
    return tuple(grid[v][u] if 0 <= v < h and 0 <= u < w else -1
                 for v in range(y-radius, y+radius+1)
                 for u in range(x-radius, x+radius+1))


def at_radius(tasks, radius):
    groups = defaultdict(list)
    per_task = {}
    for task in tasks:
        validate_task(task)
        per_task[task['task_id']] = {'conflict_classes': 0, 'pixel_error_lower_bound': 0}
        for index, pair in enumerate(task['train']):
            source, target = pair['input'], pair['output']
            if (len(source), len(source[0])) != (len(target), len(target[0])):
                continue
            for y, row in enumerate(target):
                for x, color in enumerate(row):
                    groups[signature(source, y, x, radius)].append(
                        {'task_id': task['task_id'], 'demo_index': index,
                         'y': y, 'x': x, 'required_color': color})
    conflicts, repeated, lower, witnesses = 0, 0, 0, []
    for key, occurrences in sorted(groups.items()):
        if len(occurrences) > 1:
            repeated += 1
        counts = Counter(o['required_color'] for o in occurrences)
        lower += len(occurrences) - max(counts.values())
        if len(counts) > 1:
            conflicts += 1
            if len(witnesses) < 3:
                a = occurrences[0]
                b = next(o for o in occurrences if o['required_color'] != a['required_color'])
                witnesses.append({'signature_sha256': hashlib.sha256(
                    json.dumps(key, separators=(',', ':')).encode()).hexdigest(),
                    'left': a, 'right': b})
        local = defaultdict(list)
        for occurrence in occurrences:
            local[occurrence['task_id']].append(occurrence['required_color'])
        for tid, values in local.items():
            c = Counter(values)
            per_task[tid]['pixel_error_lower_bound'] += len(values) - max(c.values())
            per_task[tid]['conflict_classes'] += int(len(c) > 1)
    return {'radius': radius, 'positions': sum(map(len, groups.values())),
            'unique_signatures': len(groups), 'repeated_signature_classes': repeated,
            'conflict_classes': conflicts, 'pixel_error_lower_bound': lower,
            'witnesses': witnesses, 'within_task': per_task}


def analyze(tasks):
    if not tasks or len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('nonempty panel with distinct task IDs required')
    for task in tasks:
        validate_task(task)
    sweep = [at_radius(tasks, r) for r in range(9)]
    sizes = []
    for task in tasks:
        for i, p in enumerate(task['train']):
            if (len(p['input']), len(p['input'][0])) != (len(p['output']), len(p['output'][0])):
                sizes.append({'task_id': task['task_id'], 'demo_index': i})
    compatible = [t for t in tasks if not any(x['task_id'] == t['task_id'] for x in sizes)]
    return {'schema': 'nca-locality-lower-bound/1',
            'total_tasks': len(tasks), 'total_demos': sum(len(t['train']) for t in tasks),
            'shape_incompatible_pairs': sizes,
            'compatible_task_ids': [t['task_id'] for t in compatible],
            'compatible_demos': sum(len(t['train']) for t in compatible),
            'minimum_shared_radius_not_ruled_out': next((s['radius'] for s in sweep if not s['conflict_classes']), None),
            'minimum_individual_radius_not_ruled_out': {
                t['task_id']: next((s['radius'] for s in sweep if not s['within_task'][t['task_id']]['conflict_classes']), None)
                for t in compatible},
            'sweep': sweep,
            'qualification': 'Necessary locality conditions only; no learned weights, capacity proof or held-out evaluation.'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks, provenance = load_frozen_panel(root,
        root/'reports/nca-capacity-20261008/INPUT_INVENTORY.json',
        root/'reports/nca-capacity-20261008/MATERIALIZATION.json')
    result = analyze(tasks)
    result['input_provenance'] = provenance
    result['diagnostic_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
