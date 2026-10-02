"""Bounded authored OCR glyph-component crossover; no fitting or final access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph, array_hash
from cellforge.construction.transactions import apply_ops
from cellforge.governance.ledger import Ledger
from cellforge.reference.evaluator import evaluate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def crossover(recipient, donor, recipient_keys, donor_keys):
    """Replace digit rows only; preserve recipient alignment/blank/EOS logic."""
    if recipient.meta['pixel_contract'] != donor.meta['pixel_contract'] or recipient.meta['vocab'] != donor.meta['vocab']:
        raise ValueError('incompatible OCR contracts')
    ops, components = [], []
    for target, source in zip(recipient_keys, donor_keys):
        value = recipient.params[target].copy()
        source_value = donor.params[source]
        if value.shape[1:] != source_value.shape[1:] or value.shape[0] < 10 or source_value.shape[0] < 10:
            raise ValueError('incompatible digit glyph shapes')
        value[:10] = source_value[:10]
        ops.append({'op': 'set_param', 'name': target, 'value': value.tolist()})
        components.append({'recipient_param': target, 'donor_param': source,
                           'donor_array_hash': array_hash(source_value), 'rows': [0, 10],
                           'remainder': 'unchanged recipient rows'})
    child, _ = apply_ops(recipient, ops)
    child.validate()
    return child, ops, components


def prepare(repo, output):
    inventory = []
    for path in sorted((repo / 'runs').glob('*/exports/*/graph.json')):
        artifact = Graph.load(path.parent)
        card = path.parent / 'MODEL_CARD.json'
        inventory.append({'path': str(path.parent.relative_to(repo)), 'hash': artifact.hash(),
                          'entries': artifact.entries, 'pixel_contract': artifact.meta.get('pixel_contract'),
                          'family': 'ocr-raster' if {'encode', 'step'}.issubset(artifact.entries) else 'non-ocr-incompatible',
                          'existing_card': json.loads(card.read_text()) if card.exists() else None})
    branches = json.loads((repo / 'runs/p1/branches.json').read_text())
    graphs = [{'hash': Graph.load(p.parent).hash(), 'path': str(p.parent.relative_to(repo))}
              for p in sorted((repo / 'runs/p1/graphs').glob('*/graph.json'))]
    parents = {name: Graph.load(repo / 'runs/p1/exports' / name)
               for name in ['r1-iter-a', 'r1-iter-b', 'r1-oneshot']}
    candidates = []
    for name, graph in parents.items():
        graph.save(output / 'population' / name)
        candidates.append({'id': name, 'kind': 'parent', 'hash': graph.hash()})
    plans = [('b_glyph_a', 'r1-iter-b', 'r1-iter-a', ('tmpl.W1', 'tmpl.b1'), ('glyph.W1', 'glyph.b1')),
             ('b_glyph_oneshot', 'r1-iter-b', 'r1-oneshot', ('tmpl.W1', 'tmpl.b1'), ('tmpl.W', 'tmpl.b')),
             ('a_glyph_oneshot', 'r1-iter-a', 'r1-oneshot', ('glyph.W1', 'glyph.b1'), ('tmpl.W', 'tmpl.b'))]
    ledger = Ledger(output / 'ledger.jsonl')
    for name, base, donor, targets, sources in plans:
        child, ops, components = crossover(parents[base], parents[donor], targets, sources)
        child.save(output / 'population' / name)
        proposal = {'protocol': 'ocr-glyph-crossover/1', 'recipient': parents[base].hash(),
                    'donor': parents[donor].hash(), 'child_hash': child.hash(), 'components': components,
                    'feature_contract': 'authored row-major 15x12 glyph patches; digit-class row order 0..9',
                    'ops': ops, 'no_fitting': True}
        write(output / 'proposals' / (name + '.json'), proposal)
        ledger.append('ocr_crossover_proposal', proposal, 'ocr-breeder')
        candidates.append({'id': name, 'kind': 'child', 'hash': child.hash(), 'recipient': base, 'donor': donor})
    write(output / 'inventory.json', {'exports': inventory, 'p1_graphs': graphs, 'p1_branches': branches,
          'limitations': 'Raster OCR only. MNIST/MNIST1D classifiers are different input and output contracts; no reshape transfer.'})
    write(output / 'selection_plan.json', {'protocol': 'ocr-glyph-population-selection/1', 'candidates': candidates,
          'evaluation_count': 6, 'dataset': 'runs/p1/dev/dev_main-9ca8e4db0e71f0e2',
          'ranking': ['exact_match descending', 'cer ascending', 'failures ascending', 'macs_total ascending', 'parent before child on exact tie', 'id ascending'],
          'final_feedback_used_for_selection': False, 'main_branch_mutation': False,
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def score(output, dataset):
    plan = json.loads((output / 'selection_plan.json').read_text())
    results = []
    ledger = Ledger(output / 'ledger.jsonl')
    for candidate in plan['candidates']:
        graph = Graph.load(output / 'population' / candidate['id'])
        if graph.hash() != candidate['hash']:
            raise ValueError('candidate hash changed after predeclaration')
        result = candidate | {'metrics': evaluate(graph, dataset, keep_per_case=False)}
        write(output / 'results' / (candidate['id'] + '.json'), result)
        ledger.append('ocr_development_evaluation', result, 'ocr-breeder')
        results.append(result)
    results.sort(key=lambda r: (-r['metrics']['exact_match'], r['metrics']['cer'], r['metrics']['failures'],
                               r['metrics']['macs_total'], r['kind'] != 'parent', r['id']))
    write(output / 'selection.json', {'protocol': plan['protocol'], 'ranked': results, 'survivor': results[0],
                                    'claim': 'Development selection only; not a frontier comparison or AGI result.'})
    ledger.append('ocr_survivor_selected', {'id': results[0]['id'], 'hash': results[0]['hash']}, 'ocr-breeder')
    print(json.dumps([(r['id'],r['metrics']['exact_match'],r['metrics']['cer']) for r in results]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--repo', default='.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset')
    args = parser.parse_args()
    output = Path(args.output)
    if args.prepare:
        prepare(Path(args.repo), output)
    else:
        if args.dataset is None:
            parser.error('--dataset is required for evaluation')
        score(output, Path(args.dataset))


if __name__ == '__main__':
    main()
