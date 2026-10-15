# SPDX-License-Identifier: MIT-0
"""DEV-only frozen-gene scoring with exact false-positive accounting."""
import argparse
import hashlib
import json
import resource
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_curriculum import generate_task
from cellforge.reference.grid_false_positives import score_with_consistency


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--inventory', required=True)
    p.add_argument('--fill', required=True); p.add_argument('--count', required=True)
    p.add_argument('--count-family', required=True); p.add_argument('--private', required=True)
    p.add_argument('--out', required=True); args = p.parse_args(); out = Path(args.out)
    if out.exists():
        raise ValueError('fresh output required')
    inventory_path = Path(args.inventory); inventory = json.loads(inventory_path.read_bytes())
    entries = [dict(row, name=row['family'] + '/' + row['mode'], path=inventory_path.parent / row['path'])
               for row in inventory['artifacts']]
    for name, path in (('authored-local-fill', args.fill), ('authored-count-strip', args.count)):
        graph = Graph.load(path)
        entries.append({'name': name, 'path': Path(path), 'hash': graph.hash()})
    graphs = []
    for entry in entries:
        graph = Graph.load(entry['path'])
        if graph.hash() != entry['hash']:
            raise ValueError('frozen artifact changed')
        graphs.append(graph)
    directory = Path(args.private) / 'arcagi2/data/training'
    paths = [path for key, path in sorted((hashlib.sha256(x.stem.encode()).hexdigest(), x)
             for x in directory.glob('*.json')) if int(key, 16) % 5 != 0][:200]
    if len(paths) != 200:
        raise ValueError('expected native DEV200 unavailable')
    native = []; manifest = hashlib.sha256()
    for path in paths:
        raw = path.read_bytes(); manifest.update(path.stem.encode() + b':' + hashlib.sha256(raw).digest())
        native.append((path.stem, json.loads(raw)))
    count_raw = Path(args.count_family).read_bytes(); count_family = json.loads(count_raw)['tasks']
    if len(count_family) != 300:
        raise ValueError('expected original count-family300')
    local_family = [generate_task('local_fill', 1371000 + i) for i in range(300)]
    out.mkdir(); ledger = Ledger(out / 'ledger.jsonl')
    policy = {'protocol': 'frozen-grid-gene-evidence/1', 'native_tasks': 200,
              'native_manifest_sha256': manifest.hexdigest(),
              'models': [{'name': row['name'], 'hash': row['hash']} for row in entries],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'consistency_source_sha256': hashlib.sha256(Path(__import__('cellforge.reference.grid_false_positives', fromlist=['x']).__file__).read_bytes()).hexdigest(),
              'count_family_sha256': hashlib.sha256(count_raw).hexdigest(),
              'synthetic_partition': 'first200 selection, last100 validation, frozen before scoring; no model edits/selection occur',
              'local_fill_seed_base': 1371000, 'audit_evaluations': 0, 'final_evaluations': 0,
              'public_evaluation': 'exposed for Claude library; not loaded or rerun here',
              'metric': 'exact complete demo reconstruction at a fixed output rank; per-query either rank exact matching; conditional false positives have null when denominator0',
              'primary_owner_rule': 'no gradients of any kind; no multiplicative-update mixer'}
    write(out / 'policy.json', policy); ledger.append('policy_frozen', policy, 'operator')
    rows = []
    for entry, graph in zip(entries, graphs):
        predictor = GridPredictor(graph, backend='compiled')
        groups = [('native-development', native)]
        if entry['name'] in ('authored-local-fill', 'authored-count-strip'):
            family = local_family if entry['name'] == 'authored-local-fill' else count_family
            for split, tasks in (('original-synthetic-selection', family[:200]),
                                 ('original-synthetic-validation', family[200:])):
                groups.append((split, [(str(i), task) for i, task in enumerate(tasks)]))
        for split, tasks in groups:
            row = score_with_consistency(tasks, predictor)
            row.update({'name': entry['name'], 'hash': graph.hash(), 'stats': graph.stats(), 'split': split,
                        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        'rss_scope': 'whole process high-water to date, not incremental model memory'})
            rows.append(row); ledger.append('frozen_graph_scored', row, 'operator')
            write(out / 'progress.json', rows)
    result = {'policy': policy, 'rows': rows,
              'limitations': 'authored narrow genes; no controlled maker/population improvement, no unseen-final or frontier superiority; consistency checks add separate diagnostic cost'}
    ledger.append('evidence_complete', result, 'operator'); result['ledger_entries_verified'] = ledger.verify()
    write(out / 'result.json', result)
    print(json.dumps({'rows': len(rows), 'native': [{key: row[key] for key in ('name', 'tasks_solved', 'demo_consistent_tasks', 'demo_consistent_but_wrong_tasks')}
                                                   for row in rows if row['split'] == 'native-development']}))


if __name__ == '__main__':
    main()
